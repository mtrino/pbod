import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langfuse import observe, propagate_attributes

load_dotenv()

from tools.tool_schemas import inbox_search, rag_search
from tools.registry import TOOLS_REGISTRY
from config import co, langfuse_context
from memory import HybridMemory

########################################################################
# DATA DEFINITION
########################################################################

class WorkerResponse(BaseModel):
    message: str = Field(description="Your response to the user. Leave blank if handing off.")
    out_of_scope: bool = Field(description="Set to True ONLY if the user asks a question unrelated to your current task.")

########################################################################
# HELPER FUNCTION
########################################################################

@observe(as_type="generation")
def _call_llm(chat_kwargs: dict) -> object:

    # Update the current generation
    langfuse_context.update_current_generation(
        input=chat_kwargs.get("messages", []),
        model=chat_kwargs.get("model", "command-a-03-2025")
    )

    response = co.chat(**chat_kwargs)
    
    if not response.message.tool_calls:
        langfuse_context.update_current_generation(output=response.message.content[0].text)
    else:
        langfuse_context.update_current_generation(output=f"Tool Calls Requested: {[tc.function.name for tc in response.message.tool_calls]}")
         
    return response

@observe(name="ReAct Loop")
def _execute_agent_loop(system_prompt: str, user_question: str, tools: list, response_schema: dict = None, max_steps: int = 4) -> str:
    """The universal execution engine for all ReAct agents."""

    # Logging the starting state of the loop
    langfuse_context.update_current_span(
        input={"question": user_question, "tools": [t.get("name") for t in tools if isinstance(t, dict)]}
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    # Dynamically build the arguments for co.chat
    chat_kwargs = {
        "model": "command-a-03-2025",
        "messages": messages,
        "tools": tools
    }

    # If a schema is provided, enforce JSON output for the final answer
    if response_schema:
        chat_kwargs["response_format"] = {"type": "json_object", "schema": response_schema}

    for _ in range(max_steps):

        response = _call_llm(chat_kwargs)

        if not response.message.tool_calls:
            final_text = response.message.content[0].text
            langfuse_context.update_current_span(output=final_text)
            return final_text

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

    # Return a fallback JSON string if it fails
    fallback = json.dumps({"message": "Error: Max steps reached", "out_of_scope": False})
    langfuse_context.update_current_span(output=fallback)
    return fallback
        

########################################################################
# EMAIL AGENT
########################################################################

@observe()
def email_agent(user_question: str, session_id: str, memory: HybridMemory) -> str:
    """Takes an user question and finds the answer from the email."""
    print(f"\n[Email Agent] Processing task for session {session_id}...")

    # Update current span
    langfuse_context.update_current_span(input=user_question)

    # 1. Update State
    memory.update_global_state(session_id, user_question, status="processing", routed_to="email_agent")

    # 2. Retrieve Isolated Semantic Memory (Qdrant)
    specialist_facts = memory.retrieve_specialist_knowledge("email_agent", user_question)
    facts_context = "\n- ".join(specialist_facts) if specialist_facts else "No prior learned context."

    # 3. Retrieve Session History for Multi-Turn Memory
    recent_activity = memory.get_session_context(session_id=session_id)
    if recent_activity:
        formatted_history = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_activity[-5:]]
        session_context = "\n".join(formatted_history)
    else:
        session_context = "No previous conversation."

    # 4. Augment System Prompt
    system_prompt = f"""You are the Email Agent. Use the tools to gather facts and do arithmetic.
    Your domain is answering questions about emails, spending, and finances.

    === Recent Conversation History ===
    {session_context}
    
    === Your Private Knowledge Base ===
    {facts_context}
    
    CRITICAL SCOPE RULE:
    Evaluate the user's question in the context of the Recent Conversation History. 
    If the question is a vague follow-up to the previous topic (e.g., "What was the date?", "Delete that", "Who sent it?"), it is IN SCOPE. Do your best to answer it using tools and conversation history.
    Set the `out_of_scope` flag to true ONLY if the user explicitly changes the subject to something completely unrelated (like statistics, code, or weather).
    """

    final_answer_json = _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[inbox_search],
        response_schema=WorkerResponse.model_json_schema()
    )

    # Parse the JSON string back into our Pydantic object
    try:
        parsed_response = WorkerResponse.model_validate_json(final_answer_json)
    except Exception as e:
        print(f"Failed to parse agent output: {e}")
        parsed_response = WorkerResponse(message="I encountered an formatting error.", out_of_scope=False)

    # Handle out of scope scenarios
    if parsed_response.out_of_scope:
        print("\n[Email Agent] Request out of scope. Handing back to supervisor.")
        # Clear the active routing state so the main loop sends it back to the supervisor
        memory.update_global_state(session_id, user_question, status="routing", routed_to=None)
        langfuse_context.update_current_span(output="Rejected: Out of scope")
        return parsed_response

    # 5. Post-Execution Memory Updates (Session & Ledger)
    memory.log_session_message(session_id, "email_agent", parsed_response.message)
    memory.publish_to_ledger("email_agent", f"Processed email query: '{user_question[:40]}...'")
    memory.update_global_state(session_id, user_question, status="completed", routed_to="email_agent")

    langfuse_context.update_current_span(output=parsed_response.model_dump())
    return parsed_response
    
########################################################################
# LOCAL AGENT
########################################################################

@observe()
def local_agent(user_question: str, session_id: str, memory: HybridMemory) -> str:
    """Takes a user question, finds the relevant chunks in a vector database, using those as context, generates the answer"""
    print(f"\n[Local Agent] Processing task for session {session_id}...")

    # Update current span
    langfuse_context.update_current_span(input=user_question)
    
    # 1. Update State
    memory.update_global_state(session_id, user_question, status="processing", routed_to="local_agent")

    # 2. Retrieve Isolated Semantic Memory (Qdrant)
    specialist_facts = memory.retrieve_specialist_knowledge("local_agent", user_question)
    facts_context = "\n- ".join(specialist_facts) if specialist_facts else "No prior learned context."

    # 3. Retrieve Session History for Multi-Turn Memory
    recent_activity = memory.get_session_context(session_id=session_id)
    if recent_activity:
        formatted_history = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_activity[-5:]]
        session_context = "\n".join(formatted_history)
    else:
        session_context = "No conversation history"

    # 4. Augment System Prompt
    system_prompt = f"""You are a precise assistant. 
    Use the tools to gather facts and answer questions regarding any statistics, Data Structures or any ML Interview. 
    Only answer using the contexts retrieved. 
    When you have enough information, stop calling tools and answer. 
    If no context received, say I don't know instead of making things up.

    === Recent Conversation History ===
    {session_context}

    === Your Private Knowledge Base ===
    {facts_context}

    CRITICAL SCOPE RULE:
        Evaluate the user's question in the context of the Recent Conversation History. 
        If the question is a vague follow-up to the previous topic (e.g., "What was the date?", "Delete that", "Who sent it?"), it is IN SCOPE. Do your best to answer it using tools and conversation history.
        Set the `out_of_scope` flag to true ONLY if the user explicitly changes the subject to something completely unrelated (like statistics, code, or weather).
    """
    
    # 5. Execute ReAct Loop
    final_answer_json = _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[rag_search],
        response_schema=WorkerResponse.model_json_schema()
    )

    # Parse the JSON string back into our Pydantic object
    try:
        parsed_response = WorkerResponse.model_validate_json(final_answer_json)
    except Exception as e:
        print(f"Failed to parse agent output: {e}")
        parsed_response = WorkerResponse(message="I encountered an formatting error.", out_of_scope=False)

    # Handle out of scope scenarios
    if parsed_response.out_of_scope:
        print("\n[Local Agent] Request out of scope. Handing back to supervisor.")
        # Clear the active routing state so the main loop sends it back to the supervisor
        memory.update_global_state(session_id, user_question, status="routing", routed_to=None)
        langfuse_context.update_current_span(output="Rejected: Out of scope")
        return parsed_response

    # 6. Post-Execution Memory Updates (Session & Ledger)
    memory.log_session_message(session_id, "local_agent", parsed_response.message)
    memory.publish_to_ledger("local_agent", f"Answered technical/ML query: '{user_question[:40]}...'")
    memory.update_global_state(session_id, user_question, status="completed", routed_to="local_agent")

    langfuse_context.update_current_span(output=parsed_response.model_dump())
    return parsed_response


if __name__ == "__main__":
    email_agent("Summarize the last email I got from mudrex")
