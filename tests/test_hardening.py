"""
Integration and unit tests for system hardening, including error handling,
duplicate detection, idempotent indexing, persistence, and restart recovery.
"""

import os
import pytest
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime, timezone
from config.settings import settings
from ingestion.pdf_loader import PDFLoader, Document
from ingestion.cleaner import TextCleaner
from ingestion.metadata_builder import MetadataBuilder
from chunking.semantic_chunker import RecursiveCharacterSplitter
from chunking.metadata_manager import MetadataManager
from vectorstore.chroma_manager import ChromaManager


@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """
    Creates a temporary path for handling files.
    """
    return tmp_path


@pytest.fixture(autouse=True)
def isolate_chroma_db(tmp_path):
    """
    Fixture to isolate ChromaDB persistent storage during hardening tests.
    """
    original_persist_dir = settings.CHROMA_PERSIST_DIRECTORY
    settings.CHROMA_PERSIST_DIRECTORY = str(tmp_path / "test_hardening_chroma_db")
    yield
    settings.CHROMA_PERSIST_DIRECTORY = original_persist_dir


def test_empty_pdf(temp_workspace):
    """
    Verifies that trying to parse a 0-byte PDF file raises a ValueError.
    """
    empty_file = temp_workspace / "empty.pdf"
    empty_file.write_bytes(b"")  # 0 bytes
    
    with pytest.raises(ValueError, match="empty"):
        PDFLoader(empty_file)


def test_corrupted_pdf(temp_workspace):
    """
    Verifies that parsing a corrupted PDF raises a ValueError.
    """
    corrupt_file = temp_workspace / "corrupted.pdf"
    corrupt_file.write_text("THIS IS NOT A VALID PDF CONTENT")
    
    loader = PDFLoader(corrupt_file)
    with pytest.raises(ValueError, match="corrupted|Failed to parse"):
        loader.load()


def test_password_protected_pdf(temp_workspace):
    """
    Verifies that loading password-protected PDFs raises a ValueError.
    """
    protected_file = temp_workspace / "protected.pdf"
    
    # Create encrypted PDF using PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Classified Financial Report")
    doc.save(
        str(protected_file),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="admin_pass",
        user_pw="user_pass"
    )
    doc.close()
    
    loader = PDFLoader(protected_file)
    with pytest.raises(ValueError, match="password-protected"):
        loader.load()


def test_metadata_preservation():
    """
    Verifies that metadata has the exact required keys:
    doc_id, file_name, page_number, chunk_id, section_title, source.
    """
    parent_metadata = {
        "file_name": "test_loan.pdf",
        "source": "/workspace/data/uploads/test_loan.pdf",
        "page": 1,
        "section": "ARTICLE I: DEFINITIONS"
    }
    
    manager = MetadataManager()
    enriched = manager.enrich_chunk_metadata(
        parent_metadata=parent_metadata,
        chunk_index=0,
        total_pages=5
    )
    
    required_keys = {"doc_id", "file_name", "page_number", "chunk_id", "section_title", "source"}
    assert required_keys.issubset(enriched.keys())
    assert enriched["section_title"] == "ARTICLE I: DEFINITIONS"
    assert enriched["page_number"] == 1


def test_duplicate_upload_and_idempotence(temp_workspace):
    """
    Verifies that indexing the same document twice skips duplication and operates idempotently.
    """
    # 1. Create a dummy document
    dummy_file = temp_workspace / "loan_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "SECTION 1: INTEREST RATE\nThe interest rate is 6.5%.")
    doc.save(str(dummy_file))
    doc.close()
    
    # 2. Ingest, clean, build metadata, and chunk
    loader = PDFLoader(dummy_file)
    cleaner = TextCleaner()
    builder = MetadataBuilder()
    
    raw_pages = loader.load()
    cleaned = cleaner.clean_documents(raw_pages)
    
    # Generate unique document ID using file hashing
    doc_id = ChromaManager.calculate_file_hash(dummy_file)
    for p in cleaned:
        p.metadata["doc_id"] = doc_id
        
    enriched = builder.enrich_documents(cleaned)
    
    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    meta_manager = MetadataManager()
    
    chunks = []
    for page_doc in enriched:
        split_texts = splitter.split_text(page_doc.page_content)
        for idx, text in enumerate(split_texts):
            meta = meta_manager.enrich_chunk_metadata(page_doc.metadata, idx, len(enriched))
            chunks.append(Document(page_content=text, metadata=meta))
            
    # 3. Add to ChromaManager
    manager = ChromaManager(collection_name="idempotence_test")
    manager.reset_database()
    
    # First indexing (succeeds)
    success = manager.add_documents(chunks)
    assert success is True
    
    stats1 = manager.get_stats()
    assert stats1["total_chunks"] == len(chunks)
    assert stats1["unique_documents_count"] == 1
    
    # Second indexing without overwrite (should skip idempotently)
    success_duplicate = manager.add_documents(chunks, overwrite=False)
    assert success_duplicate is True  # skipped gracefully
    
    stats2 = manager.get_stats()
    assert stats2["total_chunks"] == len(chunks)  # Chunk count does not double
    
    # Overwrite index (deletes and re-adds)
    success_overwrite = manager.add_documents(chunks, overwrite=True)
    assert success_overwrite is True
    assert manager.get_stats()["total_chunks"] == len(chunks)


def test_collection_persistence_and_restart_recovery(temp_workspace):
    """
    Verifies that documents are written to the physical storage folder,
    and are recovered correctly upon manager re-initialization.
    """
    dummy_file = temp_workspace / "persist_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "SECTION 2: DEFAULT CONDITIONS\nDefault occurs after 90 days.")
    doc.save(str(dummy_file))
    doc.close()
    
    loader = PDFLoader(dummy_file)
    raw_pages = loader.load()
    
    doc_id = ChromaManager.calculate_file_hash(dummy_file)
    for p in raw_pages:
        p.metadata["doc_id"] = doc_id
        
    meta_manager = MetadataManager()
    chunks = [
        Document(
            page_content=raw_pages[0].page_content,
            metadata=meta_manager.enrich_chunk_metadata(raw_pages[0].metadata, 0, 1)
        )
    ]
    
    # Connect and store
    manager = ChromaManager(collection_name="recovery_test")
    manager.reset_database()
    manager.add_documents(chunks)
    
    # Verify files created in persistence directory
    persist_path = Path(settings.CHROMA_PERSIST_DIRECTORY)
    assert persist_path.exists()
    assert any(persist_path.iterdir())  # should contain sqlite files/collection directories
    
    # Verify stats
    assert manager.get_stats()["total_chunks"] == 1
    
    # Simulate restart by deleting client and re-instantiating
    del manager
    
    recovered_manager = ChromaManager(collection_name="recovery_test")
    stats_recovered = recovered_manager.get_stats()
    
    # Assert data successfully recovered from disk
    assert stats_recovered["total_chunks"] == 1
    assert doc_id in stats_recovered["unique_document_ids"]
