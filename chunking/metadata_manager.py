"""
=========================================================
File Name : metadata_manager.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module manages metadata generation, preservation, and
formatting for text chunks. It guarantees document hash
tracking, page numbering, and deterministic chunk ID
mapping.

Technologies:
- Python Standard hashlib Library
- Python Standard datetime Library
=========================================================
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Handles chunk-level metadata generation, alignment, and preservation.

    Responsibilities:
    1. Generate deterministic document hashes.
    2. Generate unique page-chunk composite identifiers.
    3. Ensure required keys exist (doc_id, page_number, chunk_id).
    """

    @staticmethod
    def generate_document_id(file_name: str, source_path: str) -> str:
        """
        Generates a unique deterministic ID for a document based on its name and path.

        Workflow:
        1. Clean path to POSIX format.
        2. Combine filename and path.
        3. Extract SHA256 hex digest.

        Parameters:
            file_name (str):
                The name of the file.
            source_path (str):
                The absolute path to the file.

        Returns:
            str:
                A deterministic hash string.
        """
        # Convert path to posix format to ensure cross-platform hash stability
        posix_path: str = str(source_path).replace("\\", "/")
        combined: str = f"{file_name}:{posix_path}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def generate_chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
        """
        Generates a unique chunk ID.

        Workflow:
        1. Format document ID, page number, and chunk index.
        2. Join with structural separators.

        Parameters:
            doc_id (str):
                The document's unique identifier.
            page_number (int):
                The page number (1-indexed).
            chunk_index (int):
                The index of the chunk on the page (0-indexed).

        Returns:
            str:
                A unique chunk identifier string.
        """
        return f"{doc_id}_p{page_number}_c{chunk_index}"

    def enrich_chunk_metadata(
        self,
        parent_metadata: Dict[str, Any],
        chunk_index: int,
        total_pages: int,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enriches and formats the metadata dict for a single text chunk.

        Workflow:
        1. Extract page metadata elements.
        2. Resolve or generate parent document ID.
        3. Formulate chunk composite identifier.
        4. Populate and return enriched schema fields.

        Parameters:
            parent_metadata (Dict[str, Any]):
                The metadata from the parent page document.
            chunk_index (int):
                Index of this chunk on the page.
            total_pages (int):
                Total number of pages in the parent PDF.
            timestamp (Optional[str]):
                ISO format upload timestamp.

        Returns:
            Dict[str, Any]:
                Dict containing complete enriched metadata with exact keys.
        """
        file_name: str = parent_metadata.get("file_name", "Unknown")
        source: str = parent_metadata.get("source", "Unknown")
        page_number: int = parent_metadata.get("page", 1)
        section_title: str = parent_metadata.get("section", "Unknown")
        
        # Deterministic Document ID
        doc_id: Optional[str] = parent_metadata.get("doc_id") or parent_metadata.get("document_id")
        if not doc_id:
            doc_id = self.generate_document_id(file_name, source)

        # Unique Chunk ID
        chunk_id: str = self.generate_chunk_id(doc_id, page_number, chunk_index)

        # ISO Timestamp
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Construct final dict matching user requirements exactly
        enriched: Dict[str, Any] = {
            "doc_id": doc_id,
            "file_name": file_name,
            "page_number": page_number,
            "chunk_id": chunk_id,
            "section_title": section_title,
            "source": source,
            "total_pages": total_pages,
            "upload_timestamp": timestamp
        }

        return enriched
