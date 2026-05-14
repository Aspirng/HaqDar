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

from ingest.loader import load_documents, DOCUMENTS_DIR
from rag.vector_store import VectorStore, embed_batch

if __name__ == "__main__":
    print("=" * 50)
    print("HaqDar — Incremental Ingestion Pipeline")
    print("=" * 50)

    t0 = time.time()

    # 1. Load existing VectorStore
    store = VectorStore.load()
    ingested_files = store.get_ingested_filenames()

    # 2. Scan for new PDFs
    all_pdfs = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    new_pdfs = [f for f in all_pdfs if f not in ingested_files]

    if not new_pdfs:
        print(f"\n[OK] Database is up to date. (Processed {len(all_pdfs)} files)")
        sys.exit(0)

    print(f"\n[1/3] Found {len(new_pdfs)} new PDF(s) to process:")
    for f in new_pdfs:
        print(f"  + {f}")

    # 3. Load chunks for new files
    print("\n[2/3] Extracting and chunking new documents...")
    new_chunks = load_documents(new_pdfs)

    if not new_chunks:
        print("\n[ERROR] No chunks were loaded from the new files.")
        sys.exit(1)

    # 4. Embed and add to store in batches
    print(f"\n[3/3] Embedding {len(new_chunks)} chunks with OpenAI...")
    
    BATCH_SIZE = 20
    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)
        
        # Prepare document metadata for store
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

    # 5. Save the updated store
    store.save()

    print(f"\n[OK] Ingestion complete! Total chunks now: {len(store.documents)}")
    print(f"[OK] Done in {time.time() - t0:.0f}s")
