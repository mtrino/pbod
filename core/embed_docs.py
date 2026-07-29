import os
import json
import cohere
import pymupdf4llm
import uuid
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from qdrant_client.models import Distance, SparseVector, SparseVectorParams, VectorParams, Modifier, PointStruct

from core._chunking_strategies import fixed_size_chunking

load_dotenv()

SYNC_FILE = "data/synced.json"
KB_PATH = "data/resources"
COLLECTION_NAME = "your_collection_name"
MAX_CHUNK_SIZE = 96

def _get_all_files(file_path: str) -> str:
    """Returns the list of all files iteratively in the path."""
    files = os.listdir(file_path)
    return [os.path.join(file_path, file) for file in files]

def _get_already_indexed_files(sync_metadata_path: str) -> str:
    """Gets the list of already indexed files. """

    if not os.path.exists(sync_metadata_path):
        return []
    
    with open(sync_metadata_path, "r") as f:
        files = f.read()
    return list(json.loads(files).keys())

def files_to_index(file_path: str, sync_metadata_path: str) -> str:
    """Returns the files that are not yet embedded in qdrant."""
    all_files = _get_all_files(file_path)
    synced_files = _get_already_indexed_files(sync_metadata_path)
    return list(set(all_files) - set(synced_files))

def _upload_file_to_qdrant(file_path: str):
    """Chunks, embeds and uploads a single pdf to qdrant db"""

    # Instantiating the clients
    co = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])
    qdrant = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'])
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

    # Converting the pdfs into markdowns
    md = pymupdf4llm.to_markdown(file_path)

    # Getting the chunks
    chunks = fixed_size_chunking(md, 512, 62)
    print(f"Number of chunks: {len(chunks)}")

    # Embedding the chunks
    dense = []
    for i in range(0, len(chunks), MAX_CHUNK_SIZE):
        emb = co.embed(
            model="embed-v4.0",
            input_type="search_document",
            texts=chunks[i:i+MAX_CHUNK_SIZE],
            embedding_types=["float"]
        ).embeddings.float
        dense.extend(emb)

    sparse = [
        SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        for sv in bm25.embed(chunks)
    ]

    # Check if a collection exists
    # If does not exist, we create a new collection
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=len(dense[0]), distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": dense[i], "sparse": sparse[i]},
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Upserted {len(points)} point to qdrant")
    print(f"Total point in collection: {qdrant.get_collection(COLLECTION_NAME).points_count}")

    # Updating chunked files metadata
    metadata = {file_path: str(datetime.now())}
    if os.path.exists(SYNC_FILE) and os.path.getsize(SYNC_FILE):
        with open(SYNC_FILE, 'r') as file:
            existing_metadata = json.load(file)
    else:
        existing_metadata = {}
    
    existing_metadata.update(metadata)
    with open(SYNC_FILE, 'w') as file:
        json.dump(existing_metadata, file)

def upload_to_qdrant():
    """Reads the files to be stored in the vector db, chunks, embeds and uploads them in qdrant"""
    files = files_to_index(KB_PATH, SYNC_FILE)
    for file in files:
        _upload_file_to_qdrant(file)
        print(f"File {file.split('/')[-1]} upserted to Qdrant")

if __name__ == "__main__":
    upload_to_qdrant()
