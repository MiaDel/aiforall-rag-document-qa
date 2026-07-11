"""
=========================================================
File Name : pdf_loader.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module handles loading and parsing of PDF documents.
It integrates PyMuPDF (fitz) for fast text extraction and
pdfplumber for markdown-like table extraction. Includes
validations for corrupted, protected, or empty PDFs.

Technologies:
- PyMuPDF (fitz)
- pdfplumber
- Pydantic
=========================================================
"""

import os
import logging
from typing import List, Dict, Any, Union
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel, Field

# Setup logger for ingestion
logger = logging.getLogger(__name__)


class Document(BaseModel):
    """
    Standardized document schema compatible with core RAG and LangChain.

    Responsibilities:
    1. Store page text content.
    2. Maintain metadata dictionary (source, page, file_name, section).
    """
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PDFLoader:
    """
    Extracts text and table information from PDF documents with robust error handling.

    Responsibilities:
    1. Validate file format and size.
    2. Check password encryption.
    3. Extract text sequentially page-by-page.
    4. Extract tables and convert them to markdown representations.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initializes the PDFLoader with a file path.

        Workflow:
        1. Resolve absolute file path.
        2. Verify file existence.
        3. Check file extension suffix.
        4. Validate non-empty file size.

        Parameters:
            file_path (str | Path):
                Absolute or relative path to the PDF file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a PDF or is 0 bytes.
        """
        self.file_path: Path = Path(file_path).resolve()
        
        # Verify file exists
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found at: {self.file_path}")
        
        # Verify PDF extension suffix
        if self.file_path.suffix.lower() != ".pdf":
            logger.error(f"Unsupported file type: {self.file_path.name}. Must be a PDF.")
            raise ValueError(f"Unsupported file format '{self.file_path.suffix}'. Only PDFs are supported.")

        # Verify file size is not zero
        if self.file_path.stat().st_size == 0:
            logger.error(f"Empty file: {self.file_path.name} has 0 bytes.")
            raise ValueError(f"File '{self.file_path.name}' is empty (0 bytes).")

    def load(self) -> List[Document]:
        """
        Extracts content from all pages of the PDF.
        Integrates text from PyMuPDF and tables from pdfplumber.

        Workflow:
        1. Open the PDF using PyMuPDF (fitz) and validate encryption/pages.
        2. Iterate over pages and extract plain text.
        3. Open with pdfplumber, extract tables, and format them as markdown text.
        4. Concatenate text and formatted tables.
        5. Build page-level metadata list and return Documents.

        Parameters:
            None

        Returns:
            List[Document]:
                List of Document objects representing each page.

        Raises:
            ValueError: For password protection, empty text, or corruption.
            RuntimeError: For extraction-specific failures.
        """
        documents: List[Document] = []
        file_name: str = self.file_path.name
        
        logger.info(f"Opening PDF document for extraction: {file_name}")
        
        try:
            # 1. Open with PyMuPDF to parse pages
            try:
                doc_fitz = fitz.open(self.file_path)
            except Exception as fitz_err:
                logger.error(f"Failed to open PDF '{file_name}' (possibly corrupted): {str(fitz_err)}")
                raise ValueError(f"Failed to parse PDF '{file_name}'. The file may be corrupted.") from fitz_err

            # Check password protection encryption
            if doc_fitz.is_encrypted:
                doc_fitz.close()
                logger.error(f"Password protected file: {file_name}")
                raise ValueError(f"Failed to load PDF '{file_name}': File is password-protected.")

            # Validate total pages count
            total_pages: int = len(doc_fitz)
            if total_pages == 0:
                doc_fitz.close()
                logger.error(f"PDF has 0 pages: {file_name}")
                raise ValueError(f"Failed to load PDF '{file_name}': File contains 0 pages.")

            total_characters: int = 0
            
            # 2. Open with pdfplumber for table parsing
            with pdfplumber.open(self.file_path) as doc_plumber:
                for page_idx in range(total_pages):
                    page_num: int = page_idx + 1
                    logger.debug(f"Processing page {page_num}/{total_pages} of {file_name}")
                    
                    # Extract regular text from PyMuPDF
                    page_fitz = doc_fitz[page_idx]
                    page_text: str = page_fitz.get_text("text") or ""
                    
                    # Extract tables using pdfplumber
                    page_plumber = doc_plumber.pages[page_idx]
                    
                    tables: List[Any] = []
                    try:
                        tables = page_plumber.extract_tables()
                    except Exception as tbl_err:
                        logger.warning(f"Table extraction failed on page {page_num} for '{file_name}': {str(tbl_err)}")
                    
                    table_str_list: List[str] = []
                    if tables:
                        for idx, table in enumerate(tables):
                            cleaned_table: List[str] = []
                            for row in table:
                                if row is not None:
                                    cleaned_row: List[str] = [str(cell).strip() if cell is not None else "" for cell in row]
                                    if any(cleaned_row):
                                        cleaned_table.append(" | ".join(cleaned_row))
                            
                            if cleaned_table:
                                table_markdown: str = f"\n\n[Table {idx + 1}]:\n" + "\n".join(cleaned_table) + "\n"
                                table_str_list.append(table_markdown)
                    
                    # Combine regular text and formatted tables
                    combined_content: str = page_text
                    if table_str_list:
                        combined_content += "".join(table_str_list)
                    
                    total_characters += len(combined_content.strip())
                    
                    # Build basic metadata at page level
                    metadata: Dict[str, Any] = {
                        "source": self.file_path.as_posix(),
                        "file_name": file_name,
                        "page": page_num,
                        "section": "Unknown"
                    }
                    
                    documents.append(Document(page_content=combined_content, metadata=metadata))
            
            doc_fitz.close()
            
            # Validate extracted text content length is not zero
            if total_characters == 0:
                logger.error(f"No readable text extracted from {file_name}.")
                raise ValueError(f"Failed to load PDF '{file_name}': No readable text content found in document.")
                
            logger.info(f"Successfully extracted {len(documents)} pages from {file_name} ({total_characters} characters).")
            return documents
            
        except ValueError as val_err:
            # Re-raise explicit validation errors directly
            raise val_err
        except Exception as e:
            logger.exception(f"Unexpected error extracting text from PDF {file_name}: {str(e)}")
            raise RuntimeError(f"Unexpected parser failure on '{file_name}': {str(e)}") from e
