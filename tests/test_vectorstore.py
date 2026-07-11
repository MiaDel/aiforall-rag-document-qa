"""
Unit tests for the embeddings generator (BGEEmbedder) and ChromaDB manager (ChromaManager).
"""

import os
import pytest
from datetime import datetime, timezone
from config.settings import settings
from ingestion.pdf_loader import Document
from embeddings.embedder import BGEEmbedder
from vectorstore.chroma_manager import ChromaManager


@pytest.fixture(autouse=True)
def isolate_chroma_db(tmp_path):
    """
    Fixture to isolate ChromaDB persistent storage during unit tests.
    Temporarily overrides the settings persist directory.
    """
    original_persist_dir = settings.CHROMA_PERSIST_DIRECTORY
    settings.CHROMA_PERSIST_DIRECTORY = str(tmp_path / "test_chroma_db")
    
    yield
    
    # Restore original configuration
    settings.CHROMA_PERSIST_DIRECTORY = original_persist_dir


def test_bge_embedder_dimensions():
    """
    Verifies BGEEmbedder singleton correctly returns a 1024 dimension vector.
    """
    embedder = BGEEmbedder()
    
    # Check singleton property
    embedder_second = BGEEmbedder()
    assert embedder is embedder_second
    
    # Check embedding dimension
    assert embedder.get_embedding_dimension() == 1024
    
    # Check output vector structures
    vec = embedder.embed_query("What is the loan rate?")
    assert isinstance(vec, list)
    assert len(vec) == 1024
    
    vecs = embedder.embed_documents(["document text 1", "document text 2"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024


def test_chroma_manager_lifecycle():
    """
    Verifies adding documents, statistics, deletion, and resetting in ChromaDB.
    """
    # Create database manager with a test-specific collection name
    manager = ChromaManager(collection_name="test_collection")
    
    # Verify starting state
    start_stats = manager.get_stats()
    assert start_stats["total_chunks"] == 0
    assert start_stats["unique_documents_count"] == 0
    
    # Generate mockup chunk documents
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    doc_id = "test_doc_123"
    
    docs = [
        Document(
            page_content="Interest rates for home loans are set at 6% fixed.",
            metadata={
                "file_name": "home_loan.pdf",
                "doc_id": doc_id,
                "source": "/path/to/home_loan.pdf",
                "page_number": 1,
                "section_title": "Terms",
                "chunk_id": f"{doc_id}_p1_c0",
                "total_pages": 2,
                "upload_timestamp": timestamp
            }
        ),
        Document(
            page_content="Interest rates for car loans are set at 8% floating.",
            metadata={
                "file_name": "home_loan.pdf",
                "doc_id": doc_id,
                "source": "/path/to/home_loan.pdf",
                "page_number": 2,
                "section_title": "Terms",
                "chunk_id": f"{doc_id}_p2_c0",
                "total_pages": 2,
                "upload_timestamp": timestamp
            }
        )
    ]
    
    # 1. Test addition
    success = manager.add_documents(docs)
    assert success is True
    
    # 2. Test statistics updates
    stats = manager.get_stats()
    assert stats["total_chunks"] == 2
    assert stats["unique_documents_count"] == 1
    assert doc_id in stats["unique_document_ids"]
    
    # 3. Test deletion by document ID
    deleted_count = manager.delete_document(doc_id)
    assert deleted_count == 2
    
    stats_after_delete = manager.get_stats()
    assert stats_after_delete["total_chunks"] == 0
    assert stats_after_delete["unique_documents_count"] == 0
    
    # 4. Test database resetting
    manager.add_documents(docs)
    assert manager.get_stats()["total_chunks"] == 2
    
    manager.reset_database()
    assert manager.get_stats()["total_chunks"] == 0
