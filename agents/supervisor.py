from config import co

AVAILABLE_AGENTS = ["ca_agent", "local_agent"]

def supervisor(user_prompt: str, model_name: str = "command-a-03-2025") -> str:
    """Acts as the supervisor function, which routes requests to different agents based on subject."""
    print("\n Running request via supervisor.... ")

    agents = " | ".join(AVAILABLE_AGENTS)

    system_prompt = f"""You are a Chief of Staff routing user requests.
        Your goal is to decide which agent should handle the request.
        Output valid json format only and no markdown backticks, no conversational text.

        Valid agents: {agents}

        Output only raw json in the following format:
        {{
            "plan": [
                {{"agent": "AgentName", "task": "What this agent needs to do"}},
                {{"agent": "AgentName", "task": "The next step"}}
            ]
        }}

        Logic:
        - Choose CharteredAccountant if the user asks about money, spending, banks, tax, or bills. Only use the bank name like Axix, HDFC as arguments.
        - Choose LocalAgent if the user asks any definition of statistical concepts.
        - and so on

        If the user query requires multiple agents, break down the query and call the agents using the query parts as necessary.
    """

    response = co.chat(
        model=model_name,
        messages=[{"role": "system", "content": system_prompt}, 
                  {"role": "user", "content": user_prompt}]
    )

    decision = response.message.content[0].text

    return decision
