"""
HaqDar — NumPy Vector Store
Stores document embeddings in a NumPy array, persisted via pickle.
Uses sklearn cosine_similarity for top-k search.
"""

import os
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_STORE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "haqdar_vectors.pkl")
)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    """Embed a single text using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    time.sleep(0.1)  # rate-limit safety
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single API call."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    time.sleep(0.1)
    return [item.embedding for item in response.data]


class VectorStore:
    """
    In-memory vector store backed by NumPy arrays.
    Supports add, search (cosine similarity), and pickle persistence.
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None   # shape: (n, dim)
        self.documents: list[dict] = []              # metadata + text for each chunk

    def add(self, embedding: list[float], document: dict):
        """Add a single embedding + document to the store."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])
        self.documents.append(document)

    def add_batch(self, embeddings: list[list[float]], documents: list[dict]):
        """Add a batch of embeddings + documents to the store."""
        vecs = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = vecs
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])
        self.documents.extend(documents)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find the top_k most similar documents using cosine similarity.
        Returns a list of dicts with document metadata + similarity score.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Get top_k indices (sorted descending by similarity)
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.15:  # threshold — legal text needs a low bar
                continue
            result = {**self.documents[idx], "score": round(score, 3)}
            results.append(result)

        return results

    def save(self, path: str = None):
        """Persist the vector store to a pickle file."""
        path = path or VECTOR_STORE_PATH
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Vector store saved to {path} ({len(self.documents)} chunks)")

    @classmethod
    def load(cls, path: str = None) -> "VectorStore":
        """Load a vector store from a pickle file."""
        path = path or VECTOR_STORE_PATH
        store = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            store.embeddings = data["embeddings"]
            store.documents = data["documents"]
            print(f"✓ Vector store loaded from {path} ({len(store.documents)} chunks)")
        else:
            print(f"⚠ No vector store found at {path} — starting empty")
        return store


def build_vector_store(chunks: list[dict]):
    """
    Embed all chunks using OpenAI and save to pickle.
    Processes in batches of 20 for efficiency.
    """
    store = VectorStore()

    BATCH_SIZE = 20
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        documents = [
            {
                "text": c["text"],
                "law": c["law"],
                "doc_type": c["doc_type"],
                "filename": c["filename"],
                "doc_id": c["doc_id"],
                "section_ref": c["section_ref"],
            }
            for c in batch
        ]

        store.add_batch(embeddings, documents)

    store.save()
    print(f"\n✓ Total stored: {len(store.documents)} chunks")
