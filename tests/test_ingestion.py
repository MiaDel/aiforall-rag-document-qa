"""
Unit tests for the ingestion pipeline: PDFLoader, TextCleaner, and MetadataBuilder.
"""

import os
import pytest
import fitz  # PyMuPDF
from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from ingestion.cleaner import TextCleaner
from ingestion.metadata_builder import MetadataBuilder


@pytest.fixture
def dummy_pdf_path(tmp_path) -> str:
    """
    Creates a temporary dummy PDF for testing ingestion.
    """
    pdf_file = tmp_path / "test_loan_doc.pdf"
    
    # Generate PDF using PyMuPDF
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "SECTION 1. GENERAL LOAN TERMS\n"
        "This is the first page of the loan agreement.\n"
        "The loan interest- rate is set at 5.5% per annum.\n"
        "The bor-\nrower agrees to all terms."
    )
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "ARTICLE II. REPAYMENT SCHEDULE\n"
        "Monthly payments are due on the 1st of each calendar month.\n"
        "Late payments will incur a 2% fee."
    )
    
    doc.save(str(pdf_file))
    doc.close()
    
    return str(pdf_file)


def test_pdf_loader(dummy_pdf_path):
    """
    Verifies that PDFLoader correctly extracts text and populates page metadata.
    """
    loader = PDFLoader(dummy_pdf_path)
    docs = loader.load()
    
    assert len(docs) == 2
    assert "SECTION 1. GENERAL LOAN TERMS" in docs[0].page_content
    assert "ARTICLE II. REPAYMENT SCHEDULE" in docs[1].page_content
    
    # Verify metadata fields
    for idx, doc in enumerate(docs):
        assert doc.metadata["file_name"] == "test_loan_doc.pdf"
        assert doc.metadata["page"] == idx + 1
        assert doc.metadata["section"] == "Unknown"


def test_text_cleaner():
    """
    Verifies TextCleaner removes extra whitespace, hyphens, and formats newlines.
    """
    cleaner = TextCleaner()
    
    # Test hyphen rejoining and spacing
    raw_text = "The bor-\nrower is respon- \nsible for pay-\n    ment."
    cleaned = cleaner.clean_text(raw_text)
    
    assert "borrower" in cleaned
    assert "responsible" in cleaned
    assert "payment" in cleaned
    assert "pay-\n    ment" not in cleaned
    
    # Test extra newlines collapsing
    newline_text = "Line 1\n\n\n\nLine 2"
    cleaned_newlines = cleaner.clean_text(newline_text)
    assert "\n\n" in cleaned_newlines
    assert "\n\n\n" not in cleaned_newlines


def test_metadata_builder(dummy_pdf_path):
    """
    Verifies that MetadataBuilder detects section changes across document pages.
    """
    loader = PDFLoader(dummy_pdf_path)
    cleaner = TextCleaner()
    builder = MetadataBuilder()
    
    raw_docs = loader.load()
    cleaned_docs = cleaner.clean_documents(raw_docs)
    enriched_docs = builder.enrich_documents(cleaned_docs)
    
    assert len(enriched_docs) == 2
    
    # Check page 1 section detection
    assert enriched_docs[0].metadata["section"] == "SECTION 1. GENERAL LOAN TERMS"
    # Check page 2 section detection
    assert enriched_docs[1].metadata["section"] == "ARTICLE II. REPAYMENT SCHEDULE"
