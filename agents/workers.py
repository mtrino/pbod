import json
from dotenv import load_dotenv

load_dotenv()

from tools.tool_schemas import inbox_search, rag_search
from tools.registry import TOOLS_REGISTRY
from config import co

########################################################################
# HELPER FUNCTION
########################################################################

def _execute_agent_loop(system_prompt: str, user_question: str, tools: list, max_steps: int = 4) -> str:
    """The universal execution engine for all ReAct agents."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    for step in range(max_steps):

        response = co.chat(
            model="command-a-03-2025",
            messages=messages,
            tools=tools
        )

        if not response.message.tool_calls:
            return response.message.content[0].text

        messages.append(
            {"role": "assistant",
            "tool_calls": response.message.tool_calls,
            "tool_plan": response.message.tool_plan}
        )

        for tc in response.message.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
                result = TOOLS_REGISTRY[tc.function.name](**tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": str(e)})
                })

    return "Error: Agent reached maximum steps without arriving at an answer."
        

########################################################################
# EMAIL AGENT
########################################################################

def email_agent(user_question: str) -> str:
    """Takes an user question and finds the answer from the email."""
    system_prompt = (
        "You are a precise assistant. Use the tools to gather facts and to do any arithmetic. "
        "When you have enough information, stop calling tools and answer."
    )
    return _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[inbox_search]
    )
    
########################################################################
# LOCAL AGENT
########################################################################

def local_agent(user_question: str, max_steps: int = 4) -> str:
    """Takes a user question, finds the relevant chunks in a vector database, using those as context, generates the answer"""
    system_prompt = (
        "You are a precise assistant. "
        "Use the tools to gather facts and answer question regarding any statistice, Data Structures or any ML Interview. "
        "Only answer using the contexts retrieved. "
        "When you have enough information, stop calling tools and answer. "
        "If no context received, say I don't know instead of making things up."
    )
    return _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[rag_search]
    )

if __name__ == "__main__":
    email_agent("Summarize the last email I got from mudrex")
