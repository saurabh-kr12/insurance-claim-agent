"""
test_ingest.py
--------------
Tests for src/ingest.py -- PDF loading, claim_id inference, and chunking.
These run against the synthetic sample PDFs, so make sure you've run:
    python scripts/generate_sample_claims.py
before running pytest.
"""

import os
import pytest

from src.ingest import (
    load_all_claim_pdfs,
    chunk_pages,
    infer_claim_id,
    RawPage,
)
from src.config import settings


def _sample_pdfs_exist() -> bool:
    return os.path.isdir(settings.claims_data_dir) and any(
        f.lower().endswith(".pdf") for f in os.listdir(settings.claims_data_dir)
    )


def test_infer_claim_id_from_text():
    text = "INSURANCE CLAIM RECORD\nClaim ID: CLM-1001\nPolicy Number: POL-88213"
    assert infer_claim_id("some_upload.pdf", text) == "CLM-1001"


def test_infer_claim_id_fallback_to_filename():
    text = "No claim id pattern present here."
    assert infer_claim_id("CLM-9999.pdf", text) == "CLM-9999"


@pytest.mark.skipif(not _sample_pdfs_exist(), reason="Sample PDFs not generated yet.")
def test_load_all_claim_pdfs_returns_pages():
    pages = load_all_claim_pdfs()
    assert len(pages) > 0
    assert all(isinstance(p, RawPage) for p in pages)
    assert all(p.claim_id.startswith("CLM-") for p in pages)


@pytest.mark.skipif(not _sample_pdfs_exist(), reason="Sample PDFs not generated yet.")
def test_chunk_pages_produces_documents_with_metadata():
    pages = load_all_claim_pdfs()
    documents = chunk_pages(pages)
    assert len(documents) > 0
    for doc in documents:
        assert "claim_id" in doc.metadata
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert len(doc.page_content) > 0


def test_chunk_pages_skips_empty_text():
    pages = [RawPage(claim_id="CLM-0000", source="empty.pdf", page=1, text="   ")]
    documents = chunk_pages(pages)
    assert documents == []
