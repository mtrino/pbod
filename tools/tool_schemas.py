# Define the tool schema for the email tool
email_tool = {
    "type": "function",
    "function": {
        "name": "read_emails",
        "description": "Fetches email from specific senders (mostly transactional) from Gmail to check how much is spent.",
        "parameters": {
            "type": "object",
            "properties": {
                "senders": {"type": "array"},
                "days_ago": {"type": "integer"},
                "subject_filter": {"type": "array"}
            },
            "required": ["senders"]
        }
    }
}

# Define the tool schema for the rag search tool
rag_search_tool = {
    "type": "function",
    "function": {
        "name": "search",
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

