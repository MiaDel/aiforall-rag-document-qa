"""
Module for formatting citations and packaging metadata for LangChain integrations.
"""

import logging
from typing import List, Dict, Any, Tuple
from langchain_core.runnables import RunnableSerializable
from langchain_core.documents import Document as LCDocument

logger = logging.getLogger(__name__)


class LangChainCitationFormatter:
    """
    Utility formatting document references and metadata for structured RAG outputs.
    """

    @staticmethod
    def compile_citations(retrieved_docs: List[LCDocument]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Parses unique sources and calculates mock confidence based on retrieval.
        """
        sources = []
        seen = set()
        
        for doc in retrieved_docs:
            meta = doc.metadata
            f_name = meta.get("file_name", "Unknown")
            p_num = meta.get("page_number", 1)
            
            combo = f"{f_name}:{p_num}"
            if combo not in seen:
                seen.add(combo)
                sources.append({
                    "file": f_name,
                    "page": p_num
                })

        return sources

    @staticmethod
    def package_response(
        answer: str,
        retrieved_docs: List[LCDocument],
        provider_name: str = "LangChain Chain"
    ) -> Dict[str, Any]:
        """
        Standardizes RAG response with answer text, source files, pages, and context snippet links.
        """
        sources = []
        evidence = []
        chunk_refs = []
        seen = set()

        for doc in retrieved_docs:
            meta = doc.metadata
            f_name = meta.get("file_name", "Unknown")
            p_num = meta.get("page_number", 1)
            chunk_id = meta.get("chunk_id", "Unknown")
            
            combo = f"{f_name}:{p_num}"
            if combo not in seen:
                seen.add(combo)
                sources.append({
                    "file": f_name,
                    "page": p_num
                })
            
            evidence.append(doc.page_content.strip())
            chunk_refs.append(meta)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": 1.0 if len(sources) > 0 else 0.0,
            "evidence": evidence,
            "provider_used": provider_name,
            "tokens_used": 0,  # Chain-level mock or default
            "chunk_references": chunk_refs
        }
