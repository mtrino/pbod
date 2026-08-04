# Define the tool schema for the email tool
inbox_search = {
    "type": "function",
    "function": {
        "name": "search_inbox",
        "description": "Fetches email from specific senders (mostly transactional) from Gmail.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_names": {"type": "array"},
                "days_ago": {"type": "integer"},
                "keywords": {"type": "array"}
            },
            "required": ["company_names", "days_ago"]
        }
    }
}

# Define the tool schema for the rag search tool
rag_search = {
    "type": "function",
    "function": {
        "name": "search_db",
        "description": "Searches the vector database using hybrid search (semantic + keyword) to find relevant technical information",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "fetch": {"type": "integer"},
                "k": {"type": "integer"}
            },
            "required": ["question"]
        }
    }
}

