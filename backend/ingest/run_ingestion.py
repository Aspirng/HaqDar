"""
HaqDar — Ingestion Pipeline
Run this script to load PDFs, chunk them, embed with OpenAI,
and save everything to backend/haqdar_vectors.pkl.

Usage:
    cd backend
    python ingest/run_ingestion.py
"""

import sys
import os
import time

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingest.loader import load_documents
from rag.vector_store import build_vector_store

if __name__ == "__main__":
    print("=" * 50)
    print("HaqDar — Ingestion Pipeline")
    print("=" * 50)

    t0 = time.time()

    print("\n[1/2] Loading and chunking documents...")
    chunks = load_documents()

    if not chunks:
        print("\n❌ No chunks loaded. Add PDFs to backend/documents/ and retry.")
        sys.exit(1)

    print(f"\n[2/2] Embedding {len(chunks)} chunks with OpenAI and saving to haqdar_vectors.pkl...")
    build_vector_store(chunks)

    print(f"\n✓ Done in {time.time() - t0:.0f}s")
