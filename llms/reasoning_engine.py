"""
Module for Multi-Document Reasoning and evidence synthesis across multiple PDFs.
Analyzes contexts, highlights comparisons, detects conflicts, and packages evidence chunks.
"""

import logging
from typing import List, Dict, Any
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class MultiDocumentReasoningEngine:
    """
    Synthesizes and compares evidence retrieved from multiple distinct PDF loan documents.
    """

    @staticmethod
    def identify_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts source files and pages from chunks.
        """
        sources = []
        seen = set()
        for item in retrieved_chunks:
            doc = item.get("document")
            if doc:
                f_name = doc.metadata.get("file_name", "Unknown")
                p_num = doc.metadata.get("page_number", 1)
                combo = f"{f_name}:{p_num}"
                if combo not in seen:
                    seen.add(combo)
                    sources.append({"file": f_name, "page": p_num})
        return sources

    @staticmethod
    def extract_evidence_snippets(retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts list of raw text statements representing the physical evidence base.
        """
        return [item["document"].page_content.strip() for item in retrieved_chunks if item.get("document")]

    @classmethod
    def compile_comparative_context(cls, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Group retrieved evidence fragments by document name to enable the LLM
        to compare contracts or identify conflicts cleanly.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in retrieved_chunks:
            doc = item.get("document")
            if doc:
                f_name = doc.metadata.get("file_name", "Unknown")
                if f_name not in grouped:
                    grouped[f_name] = []
                grouped[f_name].append(item)

        context_str = ""
        for doc_name, items in grouped.items():
            context_str += f"\n=== Document: {doc_name} ===\n"
            for idx, item in enumerate(items):
                p_num = item["document"].metadata.get("page_number", 1)
                sec_title = item["document"].metadata.get("section_title", "General")
                context_str += f"[Page {p_num} - Section: {sec_title}]\n"
                context_str += f"{item['document'].page_content.strip()}\n\n"
        return context_str
