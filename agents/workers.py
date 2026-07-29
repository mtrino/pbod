import json
from dotenv import load_dotenv

load_dotenv()

from tools.tool_schemas import email_tool, rag_search_tool
from tools.registry import TOOLS_REGISTRY
from config import co

def ca_agent(user_question: str, max_steps: int = 4) -> str:
    """Takes an user question and finds the answer from the email."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Use the tools to gather facts and to do any arithmetic. "
                "When you have enough information, stop calling tools and answer."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    for _ in range(max_steps):

        response = co.chat(
            model="command-a-03-2025",
            messages=messages,
            tools=[email_tool]
        )
        
        if not response.message.tool_calls:
            return response.message.content[0].text

        messages.append(
            {"role": "assistant",
            "tool_calls": response.message.tool_calls,
            "tool_plan": response.message.tool_plan}
        )

        for tc in response.message.tool_calls:
            tool_args = json.loads(tc.function.arguments)
            result = TOOLS_REGISTRY[tc.function.name](**tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

def local_agent(user_question: str, max_steps: int = 4) -> str:
    """Takes a user question, finds the relevant chunks in a vector database, using those as context, generates the answer"""
    
    messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. "
                    "Use the tools to gather facts and answer question regarding any statistice, Data Structures or any ML Interview. "
                    "Only answer using the contexts retrieved"
                    "When you have enough information, stop calling tools and answer. "
                    "If no context received, say I don't know instead of making things up."
                ),
            },
            {"role": "user", "content": user_question},
        ]

    for _ in range(max_steps):

        response = co.chat(
            model="command-a-03-2025",
            messages=messages,
            tools=[rag_search_tool]
        )

        if not response.message.tool_calls:
            return response.message.content[0].text

        messages.append(
            {"role": "assistant",
            "tool_calls": response.message.tool_calls,
            "tool_plan": response.message.tool_plan}
        )

        for tc in response.message.tool_calls:
            tool_args = json.loads(tc.function.arguments)
            result = TOOLS_REGISTRY[tc.function.name](**tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

    
