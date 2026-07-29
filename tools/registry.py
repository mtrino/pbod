from tools._email_parser import read_emails
from tools._rag_search import search

TOOLS_REGISTRY = {
    'read_emails': read_emails,
    'search': search
}
