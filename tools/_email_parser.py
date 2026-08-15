import os
import pickle
import base64
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field, validate_call
from typing import List, Optional

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

########################################################################
# Data Model
########################################################################

class DynamicEmailSearch(BaseModel):
    company_names: List[str] = Field(description="Core names of the companies to search for (e.g., 'axis', 'kempower', 'nvidia'). Omit suffixes like 'bank' or 'ltd'.")
    days_ago: int = Field(description="How many days back to search")
    keywords: Optional[List[str]] = Field(default=None, description="Optional keywords to filter the content (e.g., 'spent', 'interview', 'statement')")

########################################################################
# Functions
########################################################################

def _get_gmail_service():
    """Initializes and returns the authenticated Gmail API service client."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('../keys/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def _extract_email_body(payload: dict) -> str:
    """Recursively or iteratively extracts clean text from multipart email payloads."""
    
    body_data = ""
    parts = payload.get('parts', [])
    
    if parts:
        for part in parts:
            mime_type = part.get('mimeType')
            if mime_type == 'text/html':
                body_data = part.get('body', {}).get('data', '')
                break
            elif mime_type == 'text/plain' and not body_data:
                body_data = part.get('body', {}).get('data', '')
            elif 'parts' in part:
                # Handle nested multipart structures
                body_data = _extract_email_body(part)
                if body_data:
                    break
    elif 'body' in payload and 'data' in payload['body']:
        body_data = payload['body']['data']

    if body_data:
        body_data += "=" * (-len(body_data) % 4)
        decoded = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
        raw_text = BeautifulSoup(decoded, "html.parser").get_text(separator=' ', strip=True)
        return raw_text.encode("ascii", "ignore").decode("ascii")
    
    return ""


# The Main Tool
@validate_call
def search_inbox(**kwargs) -> str:
    """
    Queries gmail dynamically based on llm inputs
    """
    args = DynamicEmailSearch(**kwargs)
    print(f"\n[PARSER: Building dynamic query for {args.company_names} over last {args.days_ago} days...]")
    
    # 1. Build the sender clause: {from:axis from:kempower}
    from_parts = " OR ".join([f'from:"{name}"' for name in args.company_names])
    from_clause = f"({from_parts})"
    
    query_parts = [from_clause, f"newer_than:{args.days_ago}d"]

    # 2. Add keywords if the LLM provided them: {spent interview statement}
    if args.keywords:
        keyword_clause = "{" + " ".join([kw for kw in args.keywords]) + "}"
        query_parts.append(keyword_clause)
        
    final_query = " ".join(query_parts)
    print(f"[EXECUTE GMAIL QUERY]: {final_query}")

    # 3. Fetch messages from Gmail API
    service = _get_gmail_service()
    results = service.users().messages().list(userId='me', q=final_query, maxResults=15).execute()
    messages = results.get('messages', [])

    if not messages:
        return f"No emails found matching query: {final_query}"

    # 4. Process and format the results
    combined_text = ""
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract headers safely
        raw_sender = next((d['value'] for d in headers if d['name'] == 'From'), "Unknown Sender")
        raw_subject = next((d['value'] for d in headers if d['name'] == 'Subject'), "No Subject")
        
        sender = raw_sender.encode("ascii", "ignore").decode("ascii")
        subject = raw_subject.encode("ascii", "ignore").decode("ascii").lower()
        
        # Extract body
        clean_text = _extract_email_body(payload)
        
        # Truncate extremely long emails to save context window tokens
        combined_text += f"--- From: {sender} | Subject: {subject} ---\n{clean_text[:3000]}\n\n"

    return combined_text

if __name__ == "__main__":
    print(search_inbox(**{'company_names': ['axis bank'], "days_ago": 3}))