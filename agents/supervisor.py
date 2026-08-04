import json
from enum import Enum
from config import co
from pydantic import BaseModel, Field

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

def supervisor(user_prompt: str, model_name: str = "command-a-03-2025") -> str:
    """Acts as the supervisor function, which routes requests to different agents based on subject."""
    print("\n Running request via supervisor.... ")

    # Simplify the system prompt. No more begging for JSON syntax!
    system_prompt = """You are a Chief of Staff routing user requests.
        Your goal is to decide which agent should handle the request.
        Break down multi-part queries into sequential steps.

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

    return parsed_plan

if __name__ == "__main__":
    print(supervisor("How much did i spend yesterday"))
