import json
from dotenv import load_dotenv

load_dotenv()

from tools.tool_schemas import inbox_search, rag_search
from tools.registry import TOOLS_REGISTRY
from config import co
from memory import HybridMemory

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

def email_agent(user_question: str, session_id: str, memory: HybridMemory) -> str:
    """Takes an user question and finds the answer from the email."""
    print(f"\n[Email Agent] Processing task for session {session_id}...")

    # 1. Update State
    memory.update_global_state(session_id, user_question, status="processing", routed_to="email_agent")

    # 2. Retrieve Isolated Semantic Memory (Qdrant)
    specialist_facts = memory.retrieve_specialist_knowledge("email_agent", user_question)
    facts_context = "\n- ".join(specialist_facts) if specialist_facts else "No prior learned context."

    system_prompt = f"""You are a precise assistant. Use the tools to gather facts and to do any arithmetic. 
    When you have enough information, stop calling tools and answer.

    === Your Private Knowledge Base ===
    {facts_context}
    """

    final_answer = _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[inbox_search]
    )

    # 5. Post-Execution Memory Updates (Session & Ledger)
    memory.log_session_message(session_id, "email_agent", final_answer)
    memory.publish_to_ledger("email_agent", f"Processed email query: '{user_question[:40]}...'")
    memory.update_global_state(session_id, user_question, status="completed", routed_to="email_agent")

    return final_answer
    
########################################################################
# LOCAL AGENT
########################################################################

def local_agent(user_question: str, session_id: str, memory: HybridMemory, max_steps: int = 4) -> str:
    """Takes a user question, finds the relevant chunks in a vector database, using those as context, generates the answer"""
    print(f"\n[Local Agent] Processing task for session {session_id}...")

    # 1. Update State
    memory.update_global_state(session_id, user_question, status="processing", routed_to="local_agent")

    # 2. Retrieve Isolated Semantic Memory (Qdrant)
    # E.g., The agent might recall: "User is applying for Data Scientist roles."
    specialist_facts = memory.retrieve_specialist_knowledge("local_agent", user_question)
    facts_context = "\n- ".join(specialist_facts) if specialist_facts else "No prior learned context."

    # 3. Augment System Prompt with Qdrant Memory
    system_prompt = f"""You are a precise assistant. 
    Use the tools to gather facts and answer questions regarding any statistics, Data Structures or any ML Interview. 
    Only answer using the contexts retrieved. 
    When you have enough information, stop calling tools and answer. 
    If no context received, say I don't know instead of making things up.

    === Your Private Knowledge Base ===
    {facts_context}
    """
    
    # 4. Execute ReAct Loop
    final_answer = _execute_agent_loop(
        user_question=user_question, 
        system_prompt=system_prompt, 
        tools=[rag_search],
        max_steps=max_steps
    )

    # 5. Post-Execution Memory Updates (Session & Ledger)
    memory.log_session_message(session_id, "local_agent", final_answer)
    memory.publish_to_ledger("local_agent", f"Answered technical/ML query: '{user_question[:40]}...'")
    memory.update_global_state(session_id, user_question, status="completed", routed_to="local_agent")

    return final_answer

if __name__ == "__main__":
    email_agent("Summarize the last email I got from mudrex")
