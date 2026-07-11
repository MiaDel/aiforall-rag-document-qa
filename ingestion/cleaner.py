"""
=========================================================
File Name : cleaner.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module cleans and normalizes text extracted from PDF
documents. It handles whitespaces, collapses redundant
newlines, rejoins hyphenated broken words, and filters
non-printable character symbols.

Technologies:
- Python Standard re Library
=========================================================
"""

import re
import logging
from typing import List
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw text data to improve retrieval accuracy and embedding quality.

    Responsibilities:
    1. Normalize line endings.
    2. Rejoin hyphenated words split across lines.
    3. Normalize whitespaces and remove redundant empty lines.
    4. Remove non-printable control characters.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans a string by removing formatting artifacts, normalizing spaces,
        and fixing split words.

        Workflow:
        1. Replace carriage returns to normalize line endings.
        2. Resolve split hyphenation at line breaks.
        3. Compress consecutive blank lines and multiple spaces.
        4. Remove unprintable ASCII/unicode control codes.

        Parameters:
            text (str):
                Raw string extracted from PDF page.

        Returns:
            str:
                Cleaned and normalized string.
        """
        if not text:
            return ""

        # Normalize line endings (replace \r\n with \n)
        cleaned: str = text.replace("\r\n", "\n")

        # Rejoin words hyphenated at line breaks (e.g., "agree-  \n ment" -> "agreement")
        cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned)

        # Split lines and strip whitespaces from each line
        lines: List[str] = [line.strip() for line in cleaned.split("\n")]

        # Remove empty lines that are redundant (collapse multiple newlines into maximum of 2 newlines)
        rejoined: str = "\n".join(lines)
        rejoined = re.sub(r"\n{3,}", "\n\n", rejoined)

        # Normalize whitespace (collapse multiple spaces/tabs into a single space, except newlines)
        rejoined = re.sub(r"[ \t]+", " ", rejoined)

        # Remove non-printable control characters (excluding newlines and tabs)
        rejoined = "".join(ch for ch in rejoined if ch.isprintable() or ch in "\n\t")

        return rejoined.strip()

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Cleans the page_content of each Document in place.

        Workflow:
        1. Iterate over each Document page.
        2. Clean the page_content using clean_text.
        3. Reassemble Document object preserving metadata.

        Parameters:
            documents (List[Document]):
                List of raw PDF page Documents to clean.

        Returns:
            List[Document]:
                List of cleaned Documents.
        """
        logger.info(f"Cleaning {len(documents)} document pages...")
        cleaned_docs: List[Document] = []
        for doc in documents:
            cleaned_content: str = self.clean_text(doc.page_content)
            # Retain existing metadata, but clean content
            cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
        
        logger.info("Document cleaning completed successfully.")
        return cleaned_docs
