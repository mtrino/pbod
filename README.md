PBOD: Personal Board of Directors

PBOD is an intelligent, agentic orchestration framework that acts as your personal "Chief of Staff." Rather than a single chatbot, PBOD decomposes complex, multi-domain queries into structured execution plans, autonomously routing tasks to specialized agents (e.g., Financial Accountants, RAG-enabled Researchers) to synthesize precise insights.

🧠 Architecture

PBOD utilizes a Supervisor-Worker pattern built on a ReAct (Reasoning + Acting) loop.

The Supervisor: An intelligent router that breaks user queries into actionable, multi-step sub-tasks.

Worker Agents: Specialized experts capable of executing tool-specific operations (Gmail, Vector DBs, Financial APIs).

Hybrid Search: Implements both Dense (vector) and Sparse (BM25) search for high-fidelity information retrieval.

Native Orchestration: Built with pure Python orchestration for maximum control over reasoning chains and low framework overhead.

🚀 Roadmap: The Path to Full Autonomous Assistants

We are actively evolving the PBOD framework. We track our development progress below:

[ ] Observability & Explainability: Integration with Langfuse for full reasoning chains and cost analytics.

[ ] Expanded Agent Roster: Introducing specialists for legal contract analysis and biometric performance coaching.

[ ] Advanced Reranking: Implementing Cohere Rerank to optimize document retrieval relevance.

[ ] Persistent Memory: Transitioning from session-based SQLite to long-term episodic memory.

[ ] Human-in-the-Loop (HITL): Implementing tactical confirmation gates for high-stakes tool execution.

[ ] Enterprise Functionality: Caching for LLM calls, score-based agent routing, and metadata-driven filtering.

[ ] Containerization: Full Docker support for portable, robust deployment.

[ ] UI/UX: A comprehensive, dashboard-driven interface for monitoring agent activity.

🛠️ Getting Started

Prerequisites

Python 3.10+

Cohere API Key

Qdrant (for vector search)
