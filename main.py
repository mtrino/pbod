import json
import importlib
from dotenv import load_dotenv

from agents.supervisor import supervisor

load_dotenv()

ALL_AGENTS = importlib.import_module("agents.worker_agents")


def run_agent(user_question: str) -> str:
    """Takes an user question and does the needful. """
    plan = json.loads(supervisor(user_question))["plan"]

    context_results = {}
    for step in plan:
        agent_name = step["agent"]
        agent = getattr(ALL_AGENTS, agent_name)
        response = agent(user_question)
        context_results[agent_name] = response

    return context_results

if __name__ == "__main__":
    print(run_agent("How much did I spend yesterday on axis bank credit card"))
