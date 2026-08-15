import json
from enum import Enum
from config import co
from pydantic import BaseModel, Field
from langfuse import observe, propagate_attributes

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

# Creating a helper function for the llm call
# The as_type="generation" tag tells langfuse this is specifically an LLM call,
# unclocking prompt tracking, token counting and LLM latency metrics
@observe(as_type="generation")
def generate_routing_plan(model_name: str, system_prompt: str, user_prompt: str) -> str:
    # Langfuse automatically captures model_name, system_prompt, and user_prompt as inputs
    
    response = co.chat(
        model=model_name,
        messages=[{"role": "system", "content": system_prompt}, 
                  {"role": "user", "content": user_prompt}],
        response_format={
            "type": "json_object",
            "schema": SupervisorPlan.model_json_schema()
        }
    )

    # Langfuse automatically captures this return value as the output
    return response.message.content[0].text


@observe()
def supervisor(user_prompt: str, session_id: str, memory: HybridMemory, model_name: str = "command-a-03-2025") -> SupervisorPlan:
    # Langfuse automatically captures user_prompt, session_id, memory, and model_name as trace inputs
    
    print("\n Running request via supervisor.... ")
    
    # Propagate session_id and tags to all child operations (like generate_routing_plan)
    with propagate_attributes(session_id=session_id, tags=["supervisor", "routing"]):
        
        memory.log_session_message(session_id, "user", user_prompt)
        memory.update_global_state(session_id, user_prompt, status="routing", routed_to="supervisor")

        recent_activity = memory.get_recent_ledger(limit=3)
        ledger_context = "\n".join(recent_activity) if recent_activity else "No recent board activity"

        system_prompt = f"""You are a Chief of Staff routing user requests.
            Your goal is to decide which agent should handle the request.
            Break down multi-part queries into sequential steps.

            === Recent Board Activity ===
            {ledger_context}

            Logic:
            - Choose EMAIL if the user asks about money, spending, banks, tax, or bills.
            - Choose LocalAgent if the user asks any definition of statistical concepts.
        """

        # This automatically inherits the session_id and tags from the propagate block
        decision_text = generate_routing_plan(model_name, system_prompt, user_prompt)

        parsed_plan = SupervisorPlan.model_validate_json(decision_text)

        if parsed_plan.plan:
            first_assigned_agent = parsed_plan.plan[0].agent.value
            memory.update_global_state(session_id, user_prompt, status="handoff_pending", routed_to=first_assigned_agent)
            memory.publish_to_ledger("supervisor", f"Generated routing plan with {len(parsed_plan.plan)} steps. Handing off to {first_assigned_agent}.")

        # Langfuse automatically captures this return value as the trace output
        return parsed_plan

if __name__ == "__main__":
    print(supervisor("How much did i spend yesterday"))
