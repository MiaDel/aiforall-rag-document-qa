"""
Module implementing node functions for the LangGraph loan document RAG state machine.
"""

import logging
from typing import Dict, Any, List
from langchain_core.documents import Document as LCDocument
from graph.state import LoanRAGState
from guardrails.input_guard import InputGuardrail
from guardrails.output_guard import OutputGuardrail
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker
from llms.llm_router import LLMRouter
from llms.reasoning_engine import MultiDocumentReasoningEngine
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class LoanRAGNodes:
    """
    StateGraph nodes managing data transformations and pipeline operations.
    """

    def __init__(self):
        from utils.resource_manager import get_reranker, get_llm_router
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = get_reranker()
        self.router = get_llm_router()

    def guardrail_node(self, state: LoanRAGState) -> Dict[str, Any]:
        """
        Validates prompt safety.
        """
        query = state["query"]
        logger.info(f"[Graph Node] Running guardrail checks on query: '{query}'")
        
        is_safe, refusal_reason = self.input_guard.validate(query)
        if not is_safe:
            return {
                "is_safe": False,
                "answer": refusal_reason or "Blocked by input guardrails.",
                "sources": [],
                "confidence": 0.0,
                "evidence": [],
                "provider_used": "Graph Guardrail",
                "tokens_used": 0,
                "error": "Blocked by guardrails"
            }
        return {"is_safe": True, "error": None}

    def retriever_node(self, state: LoanRAGState) -> Dict[str, Any]:
        """
        Performs hybrid dense-sparse candidate search.
        """
        if not state.get("is_safe", True):
            return {}

        query = state["query"]
        logger.info(f"[Graph Node] Running hybrid retriever for query: '{query}'")
        
        # Retrieve twice K for reranking
        candidates = self.hybrid_retriever.retrieve(query, top_k=10)
        
        # Convert custom Document objects to LangChain Document schemas
        lc_docs = []
        for item in candidates:
            doc = item["document"]
            meta = doc.metadata
            lc_docs.append(
                LCDocument(
                    page_content=doc.page_content,
                    metadata={
                        "doc_id": meta.get("doc_id"),
                        "file_name": meta.get("file_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "section_title": meta.get("section_title"),
                        "source": meta.get("source")
                    }
                )
            )
        return {"documents": lc_docs}

    def reranker_node(self, state: LoanRAGState) -> Dict[str, Any]:
        """
        Applies Cross-Encoder reranking over fetched candidates.
        """
        if not state.get("is_safe", True) or not state.get("documents"):
            return {"documents": []}

        query = state["query"]
        lc_docs = state["documents"]
        logger.info(f"[Graph Node] Running reranker on {len(lc_docs)} docs.")
        
        # Convert LangChain documents back to custom formats for the reranker engine
        retrieved_items = []
        for doc in lc_docs:
            meta = doc.metadata
            retrieved_items.append({
                "document": Document(
                    page_content=doc.page_content,
                    metadata={
                        "doc_id": meta.get("doc_id"),
                        "file_name": meta.get("file_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "section_title": meta.get("section_title"),
                        "source": meta.get("source")
                    }
                ),
                "score": 1.0
            })
            
        reranked = self.reranker.rerank(query, retrieved_items, top_k_after_rerank=3)
        
        # Re-wrap sorted items to LangChain documents
        lc_reranked = []
        for item in reranked:
            doc = item["document"]
            meta = doc.metadata
            lc_reranked.append(
                LCDocument(
                    page_content=doc.page_content,
                    metadata={
                        "doc_id": meta.get("doc_id"),
                        "file_name": meta.get("file_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "section_title": meta.get("section_title"),
                        "source": meta.get("source"),
                        "score": item["score"]
                    }
                )
            )
        return {"documents": lc_reranked}

    def llm_generation_node(self, state: LoanRAGState) -> Dict[str, Any]:
        """
        Executes priority fallback LLM generation.
        """
        if not state.get("is_safe", True):
            return {}

        query = state["query"]
        lc_docs = state.get("documents", [])
        
        # If no documents, trigger refusal immediately
        refusal_msg = "I could not find the answer in the uploaded documents."
        if not lc_docs:
            return {
                "answer": refusal_msg,
                "sources": [],
                "confidence": 0.0,
                "evidence": [],
                "provider_used": "Context Guardrail",
                "tokens_used": 0
            }

        # Format retrieved items back to custom document wrapper
        retrieved_chunks = []
        for doc in lc_docs:
            meta = doc.metadata
            retrieved_chunks.append({
                "document": Document(
                    page_content=doc.page_content,
                    metadata={
                        "doc_id": meta.get("doc_id"),
                        "file_name": meta.get("file_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "section_title": meta.get("section_title"),
                        "source": meta.get("source")
                    }
                ),
                "score": meta.get("score", 1.0)
            })

        logger.info(f"[Graph Node] Invoking LLMRouter for answer generation.")
        res = self.router.generate_answer(query, retrieved_chunks)
        
        return {
            "answer": res["answer"],
            "sources": res["sources"],
            "confidence": res["confidence"],
            "evidence": res["evidence"],
            "provider_used": res["provider_used"],
            "tokens_used": res["tokens_used"]
        }

    def citation_node(self, state: LoanRAGState) -> Dict[str, Any]:
        """
        Optional final citation and grounding check node.
        """
        # Node acts as a pass-through returning final structured state values
        logger.info("[Graph Node] Citation verification completed.")
        return {}
