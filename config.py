import os
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from langfuse import get_client

load_dotenv()

langfuse_context = get_client()
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
bm25 = SparseTextEmbedding("Qdrant/bm25")

COLLECTION_NAME = "interview_prep"