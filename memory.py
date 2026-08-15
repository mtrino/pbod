import uuid
import sqlite3
from datetime import datetime
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType

from config import co, qdrant

# ---------------------------------------------------------
# HYBRID MEMORY PIPELINE
# ---------------------------------------------------------
class HybridMemory:
    def __init__(self, sqlite_path="memory.db", qdrant_collection="memory", context_limit=10):
        self.sqlite_path = sqlite_path
        self.qdrant_collection = qdrant_collection
        self.context_limit = context_limit

        # Initialize databases
        self._init_sqlite()
        self._init_qdrant()

    def _init_sqlite(self):
        """Sets up tables for State, Short-Term Session, and Episodic Ledger."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()

            # 1. Global State (Supervisor)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_state (
                    session_id TEXT PRIMARY KEY,
                    current_objective TEXT,
                    status TEXT,
                    last_routed_to TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Short-Term Session Memory (Strictly Chronological)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 3. Shared Episodic Ledger (Cross-Agent Collaboration)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_role TEXT,
                    event_summary TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _init_qdrant(self):
        """Sets up the vector store for Semantic Memory."""
        # Ensure collection exists
        if not qdrant.collection_exists(self.qdrant_collection):
            qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

            # 2. Create a payload index for the agent_role field
            qdrant.create_payload_index(
                collection_name=self.qdrant_collection,
                field_name="agent_role",
                field_schema=PayloadSchemaType.KEYWORD
            )

    # ==========================================
    # 1. SUPERVISOR LOGIC (SQLITE)
    # ==========================================
    def update_global_state(self, session_id: str, objective: str, status: str, routed_to: str):
        """Tracks what the board is currently working on."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO global_state (session_id, current_objective, status, last_routed_to)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    current_objective=excluded.current_objective,
                    status=excluded.status,
                    last_routed_to=excluded.last_routed_to,
                    updated_at=CURRENT_TIMESTAMP
            """, (session_id, objective, status, routed_to))
            conn.commit()

    def get_global_state(self, session_id: str):
        """Returns the current state given a session id"""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT current_objective, status, last_routed_to
                FROM global_state 
                WHERE session_id = ?
            """, (session_id, ))
            return cursor.fetchall()

    # ==========================================
    # 2. SESSION LOGIC (SQLITE)
    # ==========================================
    def log_session_message(self, session_id: str, role: str, content: str):
        """Logs a chat message and enforces the rolling context window per session."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO session_memory (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

            # Trim old context to save tokens
            cursor.execute(f"""
                DELETE FROM session_memory
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM session_memory
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, (session_id, session_id, self.context_limit))
            conn.commit()

    def get_session_context(self, session_id: str) ->list[dict]:
        """Retrieves the exact chronological chat history."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM session_memory WHERE session_id = ? order by timestamp ASC",
                (session_id, )
            )
            return [{"role": r[0], "content": r[1]} for r in cursor.fetchall()]

    # ==========================================
    # 3. EPISODIC LEDGER LOGIC (SQLITE)
    # ==========================================
    def publish_to_ledger(self, agent_role: str, summary: str):
        """Worker agents publish their completed tasks here."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO shared_ledger (agent_role, event_summary) VALUES (?, ?)", 
                (agent_role, summary)
            )
            conn.commit()

    def get_recent_ledger(self, limit: int = 3) -> list[str]:
        """Provides recent board actions to the next active agent."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT agent_role, event_summary FROM shared_ledger ORDER BY timestamp DESC LIMIT ?",
                (limit, )
            )
            return [f"[{r[0]}] {r[1]}" for r in cursor.fetchall()]

    # ==========================================
    # 4. ISOLATED SEMANTIC LOGIC (QDRANT)
    # ==========================================
    def learn_fact(self, agent_role: str, fact: str):
        """Embeds a fact and assigns metadata to enforce isolation boundaries."""
        vector = co.embed(
            model="embed-v4.0",
            input_type="search_document",
            texts=[fact],
            embedding_types=["float"]
        )
        qdrant.upsert(
            collection_name=self.qdrant_collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "agent_role": agent_role,
                        "fact": fact,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )

    def retrieve_specialist_knowledge(self, agent_role: str, query: str, limit: int = 3) -> list[str]:
        """Searches for relevant facts but STRICTLY filters by the active agent's role."""
        query_vector = co.embed(
            model="embed-v4.0",
            input_type="search_query",
            texts=[query],
            embedding_types=["float"]
        ).embeddings.float[0]

        search_results = qdrant.query_points(
            collection_name=self.qdrant_collection,
            query=query_vector,
            limit=limit,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="agent_role",
                        match=MatchValue(value=agent_role)
                    )
                ]
            )
        ).points
        return [hit.payload["fact"] for hit in search_results]