import uuid
import json
import importlib
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

from agents.supervisor import supervisor
from core.embed_docs import upload_to_qdrant
from memory import HybridMemory
from config import langfuse_context

load_dotenv()

ALL_AGENTS = importlib.import_module("agents.workers")

@observe()
def run_agent(user_question: str, session_id: str, memory: HybridMemory) -> str:
    """Takes a user question and orchestrates the multi-agent execution with memory."""
    print(f"\n[{session_id}] Orchestrating new request...")

    # 3. SET THE ROOT TRACE ATTRIBUTES
    with propagate_attributes(session_id=session_id):

        # Is there already an active agent handling this conversation
        current_state = memory.get_global_state(session_id)

        active_agent_name = None
        if current_state and isinstance(current_state, dict):
            active_agent_name = current_state.get("routed_to")

        # If an agent is active, bypass the supervisor and sent directly
        if active_agent_name and hasattr(ALL_AGENTS, active_agent_name):
            print(f"\n[System] Direct routing to active agent: {active_agent_name}")

            agent = getattr(ALL_AGENTS, active_agent_name)
            response = agent(user_question=user_question, session_id=session_id, memory=memory)

            # Did the active agent flag this question as out of bounds
            if hasattr(response, 'out_of_scope') and response.out_of_scope:
                print(f"\n[System] {active_agent_name} flagged request as out of scope. Re-routing via supervisor...")
                # We set this to None so the supervisor block below triggers
                active_agent_name = None 
            else:
                # The agent handled it successfully. Extract message from Pydantic model (or string if local_agent).
                message = getattr(response, 'message', response)
                result = {active_agent_name: message}
                
                print("\n=== Final Execution Results ===")
                print(json.dumps({active_agent_name: message}, indent=2))
                
                return result

        # 3. SUPERVISOR ROUTING: Triggers if this is a new session, or if the active agent bailed out
        if not active_agent_name:
            plan_obj = supervisor(user_question, session_id, memory)
            context_results = {}

            for step in plan_obj.plan:
                agent_name = step.agent.value
                specific_task = step.task
                
                print(f"\n[System] Supervisor handoff to {agent_name}: '{specific_task}'")
                agent = getattr(ALL_AGENTS, agent_name)

                response = agent(user_question=user_question, session_id=session_id, memory=memory)

                # Edge case: If the newly assigned agent immediately rejects the task
                if hasattr(response, 'out_of_scope') and response.out_of_scope:
                    print(f"\n[System] Error: {agent_name} immediately rejected the supervisor's task.")
                    context_results[agent_name] = "Rejected: Out of scope."
                    break

                # Safely extract the message whether the agent returns a Pydantic WorkerResponse or a raw string
                message = getattr(response, 'message', response)
                context_results[agent_name] = message

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
            try:
                # Capture the dictionary returned by run_agent
                results = run_agent(question, session_id=current_session_id, memory=pbod_memory)
                
                # Display the response cleanly to the user
                print("\n" + "="*40)
                for agent_name, answer in results.items():
                    # Format the agent name beautifully (e.g., email_agent -> Email Agent)
                    clean_name = agent_name.replace("_", " ").title()
                    print(f"[{clean_name}]: {answer}")
                print("="*40)

            except Exception as e:
                print(f"\n[!] An error occurred during agent execution: {str(e)}")
                print("[!] The session is still active. You can ask another question.")

            question = input("\nWhat else do you want to know? (Type 'exit' to quit)\n")

    else:
        # Default test run
        test_session_id = f"session_{uuid.uuid4().hex[:8]}"
        print(f"\nStarting default test run with session: {test_session_id}")
        
        try:
            results = run_agent(
                user_question="How much did I spend on my axis bank credit card in the last 3 days?", 
                session_id=test_session_id, 
                memory=pbod_memory
            )
            
            print("\n" + "="*40)
            for agent_name, answer in results.items():
                clean_name = agent_name.replace("_", " ").title()
                print(f"[{clean_name}]: {answer}")
            print("="*40)
            
        except Exception as e:
            print(f"\n[!] Test run failed: {str(e)}")

    # 5. FLUSH TRACES ON EXIT
    # This ensures any pending traces are sent to Langfuse before the script dies
    print("\nShutting down. Flushing traces to Langfuse...")
    langfuse_context.flush()
    print("Done. Goodbye!")