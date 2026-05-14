"""
HaqDar — Document Loader
Extracts text from PDFs in backend/documents/ using pypdf,
cleans it, and returns a list of document dicts with metadata.
"""

import os
import re
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")

# Metadata for each PDF — maps filename → law name + doc_type
DOC_META = {
    "Pakistan Penal Code.pdf":                          {"law": "Pakistan Penal Code 1860",              "doc_type": "criminal"},
    "Code_of_criminal_procedure_1898.pdf":              {"law": "Code of Criminal Procedure 1898",       "doc_type": "procedure"},
    "PUB-15-000096.pdf":                                {"law": "Sindh Consumer Protection Act 2014",    "doc_type": "consumer"},
    "administrator42956bfa0c49fd1146daa6d1a5c9cd30.pdf": {"law": "Pakistan Code Compilation",            "doc_type": "general"},
}

# RecursiveCharacterTextSplitter — 500-char chunks with 100-char overlap
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove null bytes."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\x00", "", text)
    return text.strip()


def extract_section_ref(text: str) -> str:
    """Pull out e.g. 'Section 302' or 'S. 15A' from a chunk."""
    match = re.search(r"[Ss]ection\s+(\d+[A-Z]?)|[Ss]\.\s*(\d+[A-Z]?)", text)
    if match:
        return match.group(1) or match.group(2)
    return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception as e:
        print(f"  pypdf error: {e}")

    if len(text.strip()) < 300:
        print(f"  ⚠ Very little text extracted — PDF may be scanned/image-based")

    return text


def load_documents() -> list[dict]:
    """
    Load all PDFs from the documents directory, chunk them,
    and return a list of dicts with text + metadata.
    """
    all_chunks = []

    for filename, meta in DOC_META.items():
        pdf_path = os.path.join(DOCUMENTS_DIR, filename)

        if not os.path.exists(pdf_path):
            print(f"⚠ MISSING: {filename} — skipping")
            continue

        print(f"\nLoading: {filename}")
        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw)

        if not cleaned:
            print(f"  ❌ No text extracted from {filename}")
            continue

        chunks = DEFAULT_SPLITTER.split_text(cleaned)
        count = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append({
                "text": chunk,
                "law": meta["law"],
                "doc_type": meta["doc_type"],
                "filename": filename,
                "doc_id": f"{filename.replace('.pdf', '')}_{i}",
                "section_ref": extract_section_ref(chunk),
            })
            count += 1

        print(f"  ✓ {count} chunks")

    print(f"\n=== Total: {len(all_chunks)} chunks loaded ===")
    return all_chunks
