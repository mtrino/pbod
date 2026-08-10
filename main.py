import uuid
import json
import importlib
from dotenv import load_dotenv

from agents.supervisor import supervisor
from core.embed_docs import upload_to_qdrant
from memory import HybridMemory

load_dotenv()

ALL_AGENTS = importlib.import_module("agents.workers")


def run_agent(user_question: str, session_id: str, memory: HybridMemory) -> str:
    """Takes a user question and orchestrates the multi-agent execution with memory."""
    print(f"\n[{session_id}] Orchestrating new request...")

    plan_obj = supervisor(user_question, session_id, memory)

    context_results = {}

    for step in plan_obj.plan:
        agent_name = step.agent.value

        specific_task = step.task
        print(f"\n[System] Handoff to {agent_name}: '{specific_task}'")

        agent = getattr(ALL_AGENTS, agent_name)

        response = agent(user_question=user_question, session_id=session_id, memory=memory)
        context_results[agent_name] = response

    print("\n=== Final Execution Results ===")
    print(json.dumps(context_results, indent=2))

    return context_results

if __name__ == "__main__":
    # Initialize the memory system for the entire application lifecycle
    pbod_memory = HybridMemory()

    # Check if there are any new resources to be embedded
    anynew = input("Do you need to embed any new documents in the resources folder? [Y/N] \n").strip().upper()
    if anynew == "Y":
        upload_to_qdrant()

    # Running the agent
    runagent = input("Do you want to run the agent? [Y/N] \n").strip().upper()
    if runagent == 'Y':
        # Create a single session ID for this conversation loop so short-term memory persists!
        current_session_id = f"session_{uuid.uuid4().hex[:8]}"
        print(f"\nStarted interactive session: {current_session_id}")

        question = input("\nWhat do you want to know? \n")
        while question and question.lower() not in ["exit", "quit"]:
            run_agent(question, session_id=current_session_id, memory=pbod_memory)
            question = input("\nWhat else do you want to know? (Type 'exit' to quit)\n")

    else:
        # Default test run
        test_session_id = f"session_{uuid.uuid4().hex[:8]}"
        run_agent(
            user_question="How much did I spend on my axis bank credit card in the last 3 days?", 
            session_id=test_session_id, 
            memory=pbod_memory
        )