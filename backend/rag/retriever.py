"""
HaqDar — Retriever
Loads the pickle vector store and returns the best legal sections
for a user query using OpenAI embeddings + cosine similarity.
"""

import os
from dotenv import load_dotenv
from rag.vector_store import VectorStore, embed_text, VECTOR_STORE_PATH

load_dotenv()

# Cache the loaded store in memory so we don't reload on every query
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Lazy-load the vector store from pickle."""
    global _store
    if _store is None:
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {VECTOR_STORE_PATH}. "
                "Run 'python ingest/run_ingestion.py' first."
            )
        _store = VectorStore.load(VECTOR_STORE_PATH)
    return _store


def retrieve(query: str, top_k: int = 5, doc_type_filter: str = None) -> list[dict]:
    """
    Retrieve the top_k most relevant legal chunks for a user query.

    Args:
        query: The user's legal question.
        top_k: Number of results to return (default 5).
        doc_type_filter: Optional filter by doc_type (e.g. 'criminal', 'consumer').

    Returns:
        List of dicts with keys: text, law, doc_type, section_ref, score.
    """
    store = _get_store()

    if store.embeddings is None or len(store.documents) == 0:
        return []

    # Embed the query
    query_embedding = embed_text(query)

    # Search the vector store
    results = store.search(query_embedding, top_k=top_k)

    # Apply doc_type filter if specified
    if doc_type_filter:
        results = [r for r in results if r.get("doc_type") == doc_type_filter]

    return sorted(results, key=lambda x: -x["score"])
