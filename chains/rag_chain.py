"""
Module implementing the RAG execution chain using LangChain (LCEL).
Combines retrieval and LLM router fallbacks.
"""

import logging
from typing import Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document as LCDocument
from chains.retrieval_chain import LangChainRAGRetriever
from chains.citation_chain import LangChainCitationFormatter
from llms.llm_router import LLMRouter
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class LangChainRAGChain:
    """
    RAG chain implementation utilizing LangChain Expression Language (LCEL).
    """

    def __init__(self, top_k: int | None = None):
        if top_k is not None:
            self.retriever = LangChainRAGRetriever(top_k_after_rerank=top_k)
        else:
            self.retriever = LangChainRAGRetriever()
        from utils.resource_manager import get_llm_router
        self.router = get_llm_router()

    def _route_generation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapts LangChain output documents back to LLMRouter generation format.
        """
        query = inputs["query"]
        lc_docs: List[LCDocument] = inputs["documents"]
        
        # Translate LangChain documents back to custom RAG documents
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
                # Assign default score for fusion passing
                "score": 1.0
            })
            
        logger.info(f"LangChain RAG chain: passing {len(retrieved_chunks)} documents to LLMRouter.")
        
        # Invoke our fallback generation router (which enforces input/output guardrails)
        result = self.router.generate_answer(query=query, retrieved_chunks=retrieved_chunks)
        return result

    def get_chain(self):
        """
        Assembles and returns the LCEL runnable chain.
        """
        # Define chain: 
        # 1. Fetch relevant documents under the key 'documents'
        # 2. Keep the query intact under 'query'
        # 3. Route both values to the generation function
        chain = (
            {
                "documents": self.retriever,
                "query": RunnablePassthrough()
            }
            | RunnableLambda(self._route_generation)
        )
        return chain
