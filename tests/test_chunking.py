"""
Unit tests for the chunking pipeline: RecursiveCharacterSplitter, SemanticChunker, and MetadataManager.
"""

import pytest
from datetime import datetime, timezone
from ingestion.pdf_loader import Document
from chunking.metadata_manager import MetadataManager
from chunking.semantic_chunker import RecursiveCharacterSplitter, SemanticChunker


def test_chunk_creation():
    """
    Verifies that RecursiveCharacterSplitter successfully splits text larger than chunk_size.
    """
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    text = "This is a very long string that should be split into multiple smaller chunks."
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50


def test_chunk_overlap():
    """
    Verifies that the character overlap is correctly preserved between chunks.
    """
    # Using simple separators so we can check exact overlaps
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=12, separators=[" "])
    text = "word1 word2 word3 word4 word5"
    chunks = splitter.split_text(text)
    
    # Check that overlap characters from chunk 1 appear in chunk 2
    assert len(chunks) > 1
    # Example overlap: "word1 word2" (len 11), next is "word2 word3" (overlap is word2)
    assert "word2" in chunks[0]
    assert "word2" in chunks[1]


def test_metadata_preservation():
    """
    Verifies that MetadataManager generates and preserves all mandatory keys:
    file_name, document_id, source, page_number, chunk_id, total_pages, and upload_timestamp.
    """
    manager = MetadataManager()
    
    parent_metadata = {
        "file_name": "agreement.pdf",
        "source": "/path/to/agreement.pdf",
        "page": 3,
        "section": "Section 5: Indemnity"
    }
    
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    enriched = manager.enrich_chunk_metadata(
        parent_metadata=parent_metadata,
        chunk_index=1,
        total_pages=10,
        timestamp=timestamp
    )
    
    # Mandatory field checks
    assert enriched["file_name"] == "agreement.pdf"
    assert enriched["source"] == "/path/to/agreement.pdf"
    assert enriched["page_number"] == 3
    assert enriched["section_title"] == "Section 5: Indemnity"
    assert enriched["total_pages"] == 10
    assert enriched["upload_timestamp"] == timestamp
    assert "doc_id" in enriched
    assert enriched["chunk_id"] == f"{enriched['doc_id']}_p3_c1"


def test_empty_document():
    """
    Verifies that splitting empty or blank documents behaves gracefully and doesn't crash.
    """
    splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)
    
    # 1. Test empty string
    chunks = splitter.split_text("")
    assert len(chunks) == 1
    assert chunks[0] == ""
    
    # 2. Test semantic chunker fallback on empty page Document
    semantic_chunker = SemanticChunker(max_chunk_size=100)
    doc = Document(page_content="", metadata={"file_name": "empty.pdf", "page": 1})
    
    semantic_chunks = semantic_chunker.chunk_document(doc, total_pages=1)
    # Should handle empty content gracefully and return an empty or single empty chunk
    assert len(semantic_chunks) <= 1
