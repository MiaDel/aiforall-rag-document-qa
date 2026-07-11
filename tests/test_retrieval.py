"""
Unit and integration tests for the retrieval engine: DenseRetriever, BM25Retriever, and HybridRetriever.
"""

import pytest
from datetime import datetime, timezone
from config.settings import settings
from ingestion.pdf_loader import Document
from vectorstore.chroma_manager import ChromaManager
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever


@pytest.fixture(autouse=True)
def isolate_chroma_db(tmp_path):
    """
    Fixture to isolate ChromaDB persistent storage during retrieval testing.
    """
    original_persist_dir = settings.CHROMA_PERSIST_DIRECTORY
    settings.CHROMA_PERSIST_DIRECTORY = str(tmp_path / "test_retrieval_chroma_db")
    yield
    settings.CHROMA_PERSIST_DIRECTORY = original_persist_dir


@pytest.fixture
def populated_chroma_manager():
    """
    Pre-populates a ChromaManager collection with distinct test texts.
    """
    manager = ChromaManager(collection_name="test_retrieval_collection")
    manager.reset_database()
    
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    doc_id = "mock_doc_abc"
    
    # 3 distinct chunks
    docs = [
        Document(
            page_content="The interest rate on this loan is 5.5% fixed per annum.",
            metadata={
                "file_name": "terms.pdf",
                "doc_id": doc_id,
                "source": "/workspace/terms.pdf",
                "page_number": 1,
                "section_title": "Section 1: Interest Rate",
                "chunk_id": f"{doc_id}_p1_c0",
                "total_pages": 3,
                "upload_timestamp": timestamp
            }
        ),
        Document(
            page_content="Repayment terms dictate monthly installments due on the first day of each month.",
            metadata={
                "file_name": "terms.pdf",
                "doc_id": doc_id,
                "source": "/workspace/terms.pdf",
                "page_number": 2,
                "section_title": "Section 2: Repayment Installments",
                "chunk_id": f"{doc_id}_p2_c0",
                "total_pages": 3,
                "upload_timestamp": timestamp
            }
        ),
        Document(
            page_content="Late payment charges penalty fee is set at 2% of the overdue payment balance.",
            metadata={
                "file_name": "terms.pdf",
                "doc_id": doc_id,
                "source": "/workspace/terms.pdf",
                "page_number": 3,
                "section_title": "Section 3: Defaults & Penalties",
                "chunk_id": f"{doc_id}_p3_c0",
                "total_pages": 3,
                "upload_timestamp": timestamp
            }
        )
    ]
    
    manager.add_documents(docs)
    return manager


def test_dense_retrieval(populated_chroma_manager):
    """
    Verifies that DenseRetriever retrieves semantically relevant documents.
    """
    retriever = DenseRetriever(chroma_manager=populated_chroma_manager)
    
    # Query matching interest rates
    results = retriever.retrieve("interest rates", top_k=1)
    
    assert len(results) == 1
    assert "interest rate" in results[0]["document"].page_content
    assert results[0]["score"] > 0.0
    assert results[0]["score"] <= 1.0


def test_sparse_retrieval(populated_chroma_manager):
    """
    Verifies that BM25Retriever matches keyword tokens.
    """
    retriever = BM25Retriever(chroma_manager=populated_chroma_manager)
    
    # Query matching monthly repayments
    results = retriever.retrieve("monthly installments", top_k=1)
    
    assert len(results) == 1
    assert "repayment" in results[0]["document"].page_content.lower()
    assert results[0]["score"] == 1.0  # Normalized top match score is 1.0


def test_hybrid_retrieval(populated_chroma_manager):
    """
    Verifies HybridRetriever aggregates rankings using RRF and deduplicates.
    """
    retriever = HybridRetriever(chroma_manager=populated_chroma_manager)
    
    # Query combines concepts from chunk 2 and 3
    results = retriever.retrieve("monthly installments and late payment penalties", top_k=2)
    
    # Should get two results
    assert len(results) == 2
    
    # Deduplication check: ids must be unique
    chunk_ids = [res["id"] for res in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    
    # Score normalization check
    for res in results:
        assert 0.0 <= res["score"] <= 1.0


def test_retrieval_metadata_preservation(populated_chroma_manager):
    """
    Verifies that retrieved chunks retain the required metadata properties.
    """
    retriever = HybridRetriever(chroma_manager=populated_chroma_manager)
    results = retriever.retrieve("default penalty", top_k=1)
    
    assert len(results) == 1
    meta = results[0]["document"].metadata
    
    required_keys = {"doc_id", "file_name", "page_number", "chunk_id", "section_title", "source"}
    assert required_keys.issubset(meta.keys())
    assert meta["doc_id"] == "mock_doc_abc"
    assert meta["file_name"] == "terms.pdf"
    assert meta["page_number"] in [1, 2, 3]


def test_no_result_queries(populated_chroma_manager):
    """
    Verifies that queries with no keyword/semantic matches return an empty list.
    """
    # 1. Test sparse no-result queries
    bm25 = BM25Retriever(chroma_manager=populated_chroma_manager)
    results_sparse = bm25.retrieve("xyzqweasd123", top_k=2)
    assert len(results_sparse) == 0
    
    # 2. Test dense retriever threshold filter exclusion
    dense = DenseRetriever(chroma_manager=populated_chroma_manager)
    
    # Save original threshold and temporarily raise it high to force exclusion
    original_threshold = settings.SIMILARITY_THRESHOLD
    settings.SIMILARITY_THRESHOLD = 0.99
    
    results_dense = dense.retrieve("interest rates", top_k=2)
    assert len(results_dense) == 0
    
    # Restore original threshold
    settings.SIMILARITY_THRESHOLD = original_threshold
