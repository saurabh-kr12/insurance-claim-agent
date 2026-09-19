import os
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings


@dataclass
class RawPage:
    """One page of extracted text, before chunking."""
    claim_id: str
    source: str        # filename, e.g. "CLM-1001.pdf"
    page: int           # 1-indexed page number
    text: str


def infer_claim_id(filename: str, text: str) -> str:
    """
    Tries to find the Claim ID for a document.
    1) First, look for the pattern "CLM-####" inside the extracted text
       (most reliable, works even if the file gets renamed).
    2) Fall back to the filename (without extension) if nothing is found.
    """
    match = re.search(r"CLM-\d+", text)
    if match:
        return match.group(0)
    return os.path.splitext(filename)[0]


def load_pdf_pages(pdf_path: str) -> List[RawPage]:
    """Reads a single PDF file and returns one RawPage per page."""
    filename = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)

    pages: List[RawPage] = []
    full_text_so_far = ""  # used to help infer claim_id from page 1 onward

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        full_text_so_far += text
        claim_id = infer_claim_id(filename, full_text_so_far)
        pages.append(
            RawPage(claim_id=claim_id, source=filename, page=page_index, text=text)
        )

    return pages


def load_all_claim_pdfs(claims_dir: str = None) -> List[RawPage]:
    """Loads every .pdf in the claims directory."""
    claims_dir = claims_dir or settings.claims_data_dir
    if not os.path.isdir(claims_dir):
        raise FileNotFoundError(f"Claims directory not found: {claims_dir}")

    all_pages: List[RawPage] = []
    pdf_files = sorted(f for f in os.listdir(claims_dir) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print(f"WARNING: No PDF files found in {claims_dir}")

    for filename in pdf_files:
        pdf_path = os.path.join(claims_dir, filename)
        pages = load_pdf_pages(pdf_path)
        all_pages.extend(pages)
        print(f"Loaded {len(pages)} page(s) from {filename}")

    return all_pages


def chunk_pages(pages: List[RawPage]) -> List[Document]:
    """
    Splits each page's text into overlapping chunks using
    RecursiveCharacterTextSplitter, and wraps each chunk into a LangChain
    Document with metadata attached (claim_id, source, page).

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words -- so chunks stay semantically coherent
    instead of cutting mid-sentence whenever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for page in pages:
        if not page.text.strip():
            continue  # skip pages with no extractable text
        chunks = splitter.split_text(page.text)
        for chunk_index, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "claim_id": page.claim_id,
                        "source": page.source,
                        "page": page.page,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return documents


def load_and_chunk(claims_dir: str = None) -> List[Document]:
    """Convenience function: load all PDFs and return chunked Documents."""
    pages = load_all_claim_pdfs(claims_dir)
    documents = chunk_pages(pages)
    print(f"\nTotal pages loaded: {len(pages)}")
    print(f"Total chunks created: {len(documents)}")
    return documents


if __name__ == "__main__":
    # Manual test: python -m src.ingest
    docs = load_and_chunk()
    if docs:
        print("\n--- Sample chunk ---")
        print("Metadata:", docs[0].metadata)
        print("Content preview:", docs[0].page_content[:200])
