"""
test_vectorstore.py
--------------------
Tests for src/vectorstore.py.

These tests build a small, isolated ChromaDB collection from a couple of
hand-written Documents (NOT the full PDF set) so they run fast and don't
depend on ingestion having been run first. They use a temporary directory
for Chroma's persistence so they never touch your real ./chroma_db data.
"""

import pytest
from langchain_core.documents import Document

from src import vectorstore as vs_module
from src.config import settings


@pytest.fixture()
def temp_chroma(tmp_path, monkeypatch):
    """Points ChromaDB at a temporary directory for the duration of the test."""
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_test"))
    docs = [
        Document(
            page_content="A burst pipe caused water damage in the kitchen.",
            metadata={"claim_id": "CLM-1002", "source": "CLM-1002.pdf", "page": 1},
        ),
        Document(
            page_content="A vehicle was stolen from a parking lot and later recovered.",
            metadata={"claim_id": "CLM-1003", "source": "CLM-1003.pdf", "page": 1},
        ),
    ]
    vs_module.build_vectorstore(docs, reset=True)
    return docs


def test_similarity_search_finds_relevant_chunk(temp_chroma):
    results = vs_module.similarity_search("water damage kitchen pipe", k=1)
    assert len(results) == 1
    assert results[0].metadata["claim_id"] == "CLM-1002"


def test_similarity_search_respects_claim_id_filter(temp_chroma):
    results = vs_module.similarity_search("claim details", k=5, claim_id="CLM-1003")
    assert all(r.metadata["claim_id"] == "CLM-1003" for r in results)


def test_get_all_claim_ids(temp_chroma):
    claim_ids = vs_module.get_all_claim_ids()
    assert set(claim_ids) == {"CLM-1002", "CLM-1003"}
