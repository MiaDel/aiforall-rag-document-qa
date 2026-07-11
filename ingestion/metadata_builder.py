"""
=========================================================
File Name : metadata_builder.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module extracts structural section headings from page
contents using regular expression patterns, allowing the
system to enrich page metadata dynamically.

Technologies:
- Python Standard re Library
=========================================================
"""

import re
import logging
from typing import List, Optional, Dict, Any
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """
    Enriches page documents with structural metadata, including section identification.

    Responsibilities:
    1. Scan page lines for structural markers.
    2. Maintain active section title states.
    3. Inject section names to metadata.
    """

    # Common section header patterns in loan documents:
    # E.g., "Section 2.1", "SECTION 4", "ARTICLE III", "1. DEFINITIONS", "Clause 10.5"
    SECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"^\s*(?:SECTION|section)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE),
        re.compile(r"^\s*(?:ARTICLE|article)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE),
        re.compile(r"^\s*(?:CLAUSE|clause)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE),
        re.compile(r"^\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9\s,\-\(\)]+)$"),  # E.g., "1.1 Definitions" or "2.3.1 Interest Rate"
        re.compile(r"^\s*([A-Z\s\-]{4,25})$")  # Short all-caps headings like "REPAYMENT TERMS", "DEFAULT"
    ]

    def __init__(self):
        """
        Initializes the MetadataBuilder.

        Parameters:
            None

        Returns:
            None
        """
        self.current_section: str = "Introduction"

    def detect_section(self, line: str) -> Optional[str]:
        """
        Tests if a line matches any common section heading pattern.

        Workflow:
        1. Strip whitespace from target line.
        2. Verify length is under 100 characters.
        3. Test against patterns in SECTION_PATTERNS.
        4. Return matched line or None.

        Parameters:
            line (str):
                A single text line to evaluate.

        Returns:
            Optional[str]:
                The matched section name or None.
        """
        line_stripped: str = line.strip()
        if not line_stripped or len(line_stripped) > 100:  # Heading is rarely very long
            return None

        for pattern in self.SECTION_PATTERNS:
            match = pattern.match(line_stripped)
            if match:
                # Return the whole matched line or clean it as the section label
                return line_stripped
        return None

    def enrich_documents(self, documents: List[Document]) -> List[Document]:
        """
        Iterates over pages, detects section changes, and updates metadata in-place.
        Maintains the last detected section across pages to ensure continuous mapping.

        Workflow:
        1. Reset active section tracking state.
        2. Iterate over Documents list.
        3. Parse the first 5 lines of the page text.
        4. Test lines for section indicators and update self.current_section.
        5. Map section key directly to page metadata copy.

        Parameters:
            documents (List[Document]):
                List of page-level Documents.

        Returns:
            List[Document]:
                List of metadata-enriched Documents.
        """
        logger.info(f"Enriching metadata for {len(documents)} document pages...")
        enriched_docs: List[Document] = []
        
        # Reset state for new ingestion batch
        self.current_section = "Introduction"

        for doc in documents:
            content: str = doc.page_content
            lines: List[str] = content.split("\n")
            
            # Check the first few lines of the page for section headers
            # Often sections start at the top of a page.
            for line in lines[:5]:
                detected: Optional[str] = self.detect_section(line)
                if detected:
                    self.current_section = detected
                    logger.debug(f"Detected section change: '{self.current_section}' on page {doc.metadata.get('page')}")
                    break

            # Shallow copy and update metadata
            updated_metadata: Dict[str, Any] = doc.metadata.copy()
            updated_metadata["section"] = self.current_section
            
            enriched_docs.append(Document(page_content=content, metadata=updated_metadata))

        logger.info("Document metadata enrichment completed.")
        return enriched_docs
