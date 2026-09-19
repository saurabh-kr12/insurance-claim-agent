import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings, get_embeddings

COLLECTION_NAME = "insurance_claims"


def get_vectorstore() -> Chroma:
    """
    Returns a Chroma vector store instance pointed at our persistent
    directory, using whichever embedding function is configured
    (HuggingFace local, by default).
    """
    embeddings = get_embeddings()
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def build_vectorstore(documents: List[Document], reset: bool = True) -> Chroma:
    """
    Embeds all given Documents and writes them into ChromaDB.

    reset=True (default): wipes any existing collection first, so re-running
    ingestion doesn't create duplicate chunks. Set False to add incrementally.
    """
    if reset:
        _reset_collection()

    vectorstore = get_vectorstore()
    if documents:
        vectorstore.add_documents(documents)
        print(f"Added {len(documents)} chunks to ChromaDB at {settings.chroma_persist_dir}")
    else:
        print("No documents to add.")
    return vectorstore


def _reset_collection() -> None:
    """Deletes the existing collection (if any) so ingestion is idempotent."""
    try:
        embeddings = get_embeddings()
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        existing.delete_collection()
        print("Cleared existing ChromaDB collection.")
    except Exception as e:
        # First run -- nothing to clear yet. Not an error.
        print(f"No existing collection to clear ({e}). Continuing.")


def similarity_search(
    query: str,
    k: Optional[int] = None,
    claim_id: Optional[str] = None,
) -> List[Document]:
    """
    Retrieves the top-k most relevant chunks for a query.
    If claim_id is provided, restricts the search to chunks from that claim
    only -- this is what lets tools like "summarize_claim('CLM-1001')" work.
    """
    vectorstore = get_vectorstore()
    k = k or settings.top_k

    search_kwargs = {}
    if claim_id:
        search_kwargs["filter"] = {"claim_id": claim_id}

    results = vectorstore.similarity_search(query, k=k, **search_kwargs)
    return results


def similarity_search_with_scores(
    query: str,
    k: Optional[int] = None,
    claim_id: Optional[str] = None,
):
    """Same as similarity_search but also returns a relevance score per chunk.
    Scores are logged to SQLite for LLMOps monitoring (Phase 7)."""
    vectorstore = get_vectorstore()
    k = k or settings.top_k

    search_kwargs = {}
    if claim_id:
        search_kwargs["filter"] = {"claim_id": claim_id}

    return vectorstore.similarity_search_with_relevance_scores(query, k=k, **search_kwargs)


def get_all_claim_ids() -> List[str]:
    """Returns the distinct list of claim_ids currently stored in the vector DB.
    Used by the Streamlit UI to populate a claim-selector dropdown."""
    vectorstore = get_vectorstore()
    raw = vectorstore.get(include=["metadatas"])
    claim_ids = {meta.get("claim_id") for meta in raw.get("metadatas", []) if meta.get("claim_id")}
    return sorted(claim_ids)


if __name__ == "__main__":
    # Manual test: python -m src.vectorstore
    from src.ingest import load_and_chunk

    docs = load_and_chunk()
    build_vectorstore(docs, reset=True)

    results = similarity_search("water damage kitchen", k=3)
    print("\n--- Top matches for 'water damage kitchen' ---")
    for r in results:
        print(r.metadata, "->", r.page_content[:100])
