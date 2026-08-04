from tools._email_parser import search_inbox
from tools._rag_search import search_db

TOOLS_REGISTRY = {
    'search_inbox': search_inbox,
    'search_db': search_db
}
