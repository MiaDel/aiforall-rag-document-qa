"""
Unit tests for the Cross-Encoder reranking module (CrossEncoderReranker).
"""

import pytest
from datetime import datetime, timezone
from ingestion.pdf_loader import Document
from retrievers.reranker import CrossEncoderReranker


@pytest.fixture
def mock_retrieved_items():
    """
    Returns a mock list of retrieved documents.
    Note: The last item is the most relevant for an 'interest rate' query.
    """
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    return [
        {
            "document": Document(
                page_content="Repayment of the loan must occur in monthly installations.",
                metadata={
                    "doc_id": "doc_xyz",
                    "file_name": "terms.pdf",
                    "page_number": 2,
                    "chunk_id": "doc_xyz_p2_c0",
                    "section_title": "Repayment",
                    "source": "/workspace/terms.pdf",
                    "upload_timestamp": timestamp,
                    "total_pages": 3
                }
            ),
            "score": 0.9,
            "id": "doc_xyz_p2_c0"
        },
        {
            "document": Document(
                page_content="The loan interest rate is fixed at 5.5% annually.",
                metadata={
                    "doc_id": "doc_xyz",
                    "file_name": "terms.pdf",
                    "page_number": 1,
                    "chunk_id": "doc_xyz_p1_c0",
                    "section_title": "Interest Rates",
                    "source": "/workspace/terms.pdf",
                    "upload_timestamp": timestamp,
                    "total_pages": 3
                }
            ),
            "score": 0.4,
            "id": "doc_xyz_p1_c0"
        }
    ]


def test_reranking_order(mock_retrieved_items):
    """
    Verifies that CrossEncoderReranker correctly sorts the most relevant document first.
    """
    reranker = CrossEncoderReranker()
    
    # Query matching interest rate
    results = reranker.rerank("What is the interest rate?", mock_retrieved_items, top_k_after_rerank=2)
    
    assert len(results) == 2
    # The interest rate document should be reranked to position 0
    assert "interest rate" in results[0]["document"].page_content
    # Score should be normalized
    assert 0.0 <= results[0]["score"] <= 1.0
    assert results[0]["score"] > results[1]["score"]


def test_reranker_metadata_preservation(mock_retrieved_items):
    """
    Verifies that all required metadata keys are preserved in the reranked output.
    """
    reranker = CrossEncoderReranker()
    results = reranker.rerank("interest rate", mock_retrieved_items, top_k_after_rerank=1)
    
    assert len(results) == 1
    meta = results[0]["document"].metadata
    
    required_keys = {"doc_id", "file_name", "page_number", "chunk_id", "section_title", "source"}
    assert required_keys.issubset(meta.keys())
    assert meta["doc_id"] == "doc_xyz"
    assert meta["file_name"] == "terms.pdf"


def test_reranker_duplicate_removal(mock_retrieved_items):
    """
    Verifies that duplicate chunks (same chunk_id) are consolidated and not reranked twice.
    """
    reranker = CrossEncoderReranker()
    
    # Add a duplicate chunk of the first element
    duplicate_items = mock_retrieved_items + [mock_retrieved_items[0]]
    assert len(duplicate_items) == 3
    
    results = reranker.rerank("installments", duplicate_items, top_k_after_rerank=3)
    
    # Reranking should consolidate duplicates, returning at most 2 unique chunks
    assert len(results) == 2
    assert results[0]["id"] != results[1]["id"]


def test_reranker_empty_retrieval():
    """
    Verifies that passing empty candidate listings returns empty lists without crash.
    """
    reranker = CrossEncoderReranker()
    
    # 1. Empty list of documents
    results = reranker.rerank("interest", [], top_k_after_rerank=5)
    assert results == []
    
    # 2. Empty query string (should return original items truncated)
    mock_items = [
        {"document": Document(page_content="text", metadata={"chunk_id": "1"}), "score": 0.5}
    ]
    results_empty_query = reranker.rerank("   ", mock_items, top_k_after_rerank=1)
    assert len(results_empty_query) == 1


def test_reranker_invalid_input_handling():
    """
    Verifies that corrupted documents or chunks missing metadata keys are skipped gracefully.
    """
    reranker = CrossEncoderReranker()
    
    corrupted_items = [
        # Missing page content or document entirely
        {"document": None, "score": 0.8},
        # Missing chunk_id
        {
            "document": Document(page_content="Valid text content here", metadata={"doc_id": "1"}),
            "score": 0.5
        }
    ]
    
    results = reranker.rerank("interest", corrupted_items, top_k_after_rerank=5)
    assert len(results) == 0
