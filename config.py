import os
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding

load_dotenv()

co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
bm25 = SparseTextEmbedding("Qdrant/bm25")

COLLECTION_NAME = "imp_docs"