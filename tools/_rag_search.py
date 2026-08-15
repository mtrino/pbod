from qdrant_client.models import SparseVector, FusionQuery, Prefetch

from config import co, bm25, qdrant, COLLECTION_NAME


def search_db(question: str, fetch: int = 20, k: int = 10):
    """Searches the vector DB to generate relevant answers."""

    dense = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[question]
    ).embeddings.float[0]

    sv = next(iter(bm25.query_embed([question])))

    sparse = SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())

    candidates =  qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=fetch),
            Prefetch(query=sparse, using="sparse", limit=fetch),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    ).points

    if not candidates:
        return {"result": []}

    return {"result": [
        {"text": r.payload["text"], "score": r.score}
        for r in candidates
    ]}
