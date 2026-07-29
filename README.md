PBOD: Personal Board of Directors

PBOD is an intelligent, agentic orchestration framework designed to act as your personal "Chief of Staff." It decomposes complex, multi-domain queries into structured execution plans and autonomously routes tasks to specialized agents (e.g., Financial Accountants, RAG-enabled Researchers) to synthesize precise insights.

Architecture

PBOD is built on a ReAct (Reasoning + Acting) loop leveraging LiteLLM and Cohere Command R+. Rather than a static pipeline, it utilizes a "Chief of Staff" Supervisor agent that performs structured JSON routing to coordinate specialized worker agents with access to modular, local tools.

Key Components

The Supervisor: An intelligent router that decomposes user queries into actionable, multi-step sub-tasks.

Worker Agents: Specialized experts capable of performing tool-specific operations (Gmail, Vector DBs, Financial APIs).

Hybrid Search: Implements both Dense (vector) and Sparse (BM25) search for high-fidelity information retrieval.

Native Orchestration: Built with pure Python orchestration for maximum control over the agent's thought process and reduced framework overhead.

Roadmap: The Path to Full Fledged AI Ready Assistants

We are actively evolving the PBOD framework. Our development roadmap includes:

[ ] Observability & Explainability: Integration with Langfuse for full reasoning chains and cost analytics.

[ ] Expanded Agent Roster: Introducing specialists for legal contract analysis and biometric performance coaching.

[ ] Advanced Reranking: Implementing Cohere Rerank to optimize document retrieval relevance.

[ ] Persistent Memory: Moving from session-based SQLite memory to long-term episodic memory.

[ ] Human-in-the-Loop (HITL): Adding tactical gates for high-stakes tool execution (e.g., transactions).

[ ] Enterprise Functionality: Caching for LLM calls, score-based agent routing, and metadata-driven filtering.

[ ] Containerization: Full Docker support for portable deployment.

[ ] UI/UX: A comprehensive, dashboard-driven interface for monitoring agent activity.

Getting Started

Prerequisites

Python 3.10+

Cohere API Key

Qdrant (for vector search)
