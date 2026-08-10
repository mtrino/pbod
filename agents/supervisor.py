import json
from enum import Enum
from config import co
from pydantic import BaseModel, Field

from memory import HybridMemory

########################################################################
# Data Model
########################################################################

# Defining strict Enums
class AvailableAgent(str, Enum):
    EMAIL = "email_agent"
    LOCAL = "local_agent"

# Defining the exact expected structure
class RoutingStep(BaseModel):
    # The Enum physically prevents the LLM from inventing agent names
    agent: AvailableAgent = Field(description="The ssigned agent")
    task: str = Field(description="Detailed instruction for the agent. If assigning to the CA, extract the exact bank identifier (e.g., 'axis', 'hdfc').")

class SupervisorPlan(BaseModel):
    plan: list[RoutingStep]


########################################################################
# Agent definition
########################################################################

def supervisor(user_prompt: str, session_id: str, memory: HybridMemory, model_name: str = "command-a-03-2025") -> str:
    """Acts as the supervisor function, which routes requests to different agents based on subject."""
    print("\n Running request via supervisor.... ")

    # 1. MEMORY INJECTION: Log the start of the interaction
    memory.log_session_message(session_id, "user", user_prompt)
    memory.update_global_state(session_id, user_prompt, status="routing", routed_to="supervisor")

    # 2. MEMORY INJECTION: Fetch recent board activity to give the Supervisor context
    recent_activity = memory.get_recent_ledger(limit=3)
    ledger_context = "\n".join(recent_activity) if recent_activity else "No recent board activity"

    # Simplify the system prompt. No more begging for JSON syntax!
    system_prompt = f"""You are a Chief of Staff routing user requests.
        Your goal is to decide which agent should handle the request.
        Break down multi-part queries into sequential steps.

        === Recent Board Activity ===
        {ledger_context}

        Logic:
        - Choose EMAIL if the user asks about money, spending, banks, tax, or bills.
        - Choose LocalAgent if the user asks any definition of statistical concepts.
    """

    response = co.chat(
        model=model_name,
        messages=[{"role": "system", "content": system_prompt}, 
                  {"role": "user", "content": user_prompt}],
        response_format={
            "type": "json_object",
            "schema": SupervisorPlan.model_json_schema()
        }
    )

    decision_text = response.message.content[0].text

    # Parse into python object
    parsed_plan = SupervisorPlan.model_validate_json(decision_text)

    # 3. MEMORY INJECTION: Update state and ledger with the routing decision
    if parsed_plan.plan:
        first_assigned_agent = parsed_plan.plan[0].agent.value
        memory.update_global_state(session_id, user_prompt, status="handoff_pending", routed_to=first_assigned_agent)
        memory.publish_to_ledger("supervisor", f"Generated routing plan with {len(parsed_plan.plan)} steps. Handing off to {first_assigned_agent}.")

    return parsed_plan

if __name__ == "__main__":
    print(supervisor("How much did i spend yesterday"))
