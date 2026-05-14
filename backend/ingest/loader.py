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

# RecursiveCharacterTextSplitter — 500-char chunks with 100-char overlap
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def get_law_name_from_filename(filename: str) -> str:
    """Derive a clean law name from a filename (e.g. 'Pakistan_Penal_Code.pdf' -> 'Pakistan Penal Code')."""
    name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
    # Capitalize each word
    return " ".join([w.capitalize() for w in name.split()])


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
        print(f"  [WARN] Very little text extracted — PDF may be scanned/image-based")

    return text


def load_single_document(filename: str) -> list[dict]:
    """
    Process a single PDF, chunk it, and return metadata dicts.
    """
    pdf_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.exists(pdf_path):
        print(f"⚠ NOT FOUND: {pdf_path}")
        return []

    law_name = get_law_name_from_filename(filename)
    
    print(f"\nProcessing: {filename} ({law_name})")
    raw = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw)

    if not cleaned:
        print(f"  ❌ No text extracted from {filename}")
        return []

    chunks = DEFAULT_SPLITTER.split_text(cleaned)
    document_chunks = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
        document_chunks.append({
            "text": chunk,
            "law": law_name,
            "doc_type": "legal", # Default to legal
            "filename": filename,
            "doc_id": f"{filename.replace('.pdf', '')}_{i}",
            "section_ref": extract_section_ref(chunk),
        })

    print(f"  [OK] Created {len(document_chunks)} chunks")
    return document_chunks


def load_documents(filenames: list[str] = None) -> list[dict]:
    """
    Load specific PDFs or all PDFs from documents/ if filenames is None.
    """
    all_chunks = []
    
    # If no filenames provided, scan the directory
    if filenames is None:
        filenames = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]

    for filename in filenames:
        chunks = load_single_document(filename)
        all_chunks.extend(chunks)

    print(f"\n=== Total: {len(all_chunks)} chunks loaded ===")
    return all_chunks
