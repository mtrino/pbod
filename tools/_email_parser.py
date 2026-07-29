import os
import pickle
import base64
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

ENTITY_MAPPER = {
    "Axis": "alerts@axis.bank.in",
    "HDFC": "alerts@hdfcbank.bank.in"
}

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
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
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

def read_emails(senders: list[str], days_ago: int = 3, subject_filter: list[str] = ["spent", "update"]) -> str:
    """Parser tool for email body."""

    print(f"\n[PARSER: Searching email (newer than {days_ago} days)...]")
    service = _get_gmail_service()

    # Cleaning the 

    # Creating the query
    mapped_emails = []
    for s in senders:
        if "axis" in s.lower():
            mapped_emails.append(ENTITY_MAPPER["Axis"])
        if "hdfc" in s.lower():
            mapped_emails.append(ENTITY_MAPPER["HDFC"]) 

    queries = [f"(from:{e}) newer_than:{days_ago}d" for e in mapped_emails]
    
    messages = []
    for q in queries:
        results = service.users().messages().list(userId='me', q=q, maxResults=10).execute()
        messages.extend(results.get('messages', []))

    if not messages:
        return "No recent spends found."

    combined_text = ""
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract and sanitize headers to pure ASCII
        raw_sender = next((d['value'] for d in headers if d['name'] == 'From'), "Banks")
        raw_subject = next((d['value'] for d in headers if d['name'] == 'Subject'), "No Subject")
        
        sender = raw_sender.encode("ascii", "ignore").decode("ascii")
        subject = raw_subject.encode("ascii", "ignore").decode("ascii").lower()

        if subject_filter and not any(keyword in subject for keyword in subject_filter):
            continue  # Skip this email if the subject doesn't match the filter
        
        # Extract body and sanitize to pure ASCII (removes fancy stars, bullet points, or ₹ signs that cause crashes)
        raw_clean_text = _extract_email_body(payload)
        clean_text = raw_clean_text.encode("ascii", "ignore").decode("ascii")
        
        # Truncate extremely long articles to save context window tokens
        combined_text += f"--- From: {sender} | Subject: {subject} ---\n{clean_text[:3000]}\n\n"

    return combined_text
