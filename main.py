import json
import importlib
from dotenv import load_dotenv

from agents.supervisor import supervisor
from core.embed_docs import upload_to_qdrant

load_dotenv()

ALL_AGENTS = importlib.import_module("agents.workers")


def run_agent(user_question: str) -> str:
    """Takes an user question and does the needful. """
    plan_obj = supervisor(user_question)

    context_results = {}

    for step in plan_obj.plan:
        agent_name = step.agent.value
        agent = getattr(ALL_AGENTS, agent_name)
        response = agent(user_question)
        context_results[agent_name] = response

    print(context_results)

if __name__ == "__main__":
    # Check if there ane any new resources to be embedded
    # anynew = input("Do you need to embed any new documents in the resources folder? [Y/N] \n")
    # if anynew == "Y":
    #     upload_to_qdrant()

    # Running the agent
    # runagent = input("Do you want to run the agent? [Y/N] \n")
    # if runagent == 'Y':
    #     question = input("What do you want to know? \n")
    #     while question:
    #         run_agent(question)
    #         question = input()

    print(run_agent("How much did I spend on my axis bank credit card in the last 3 days?"))
