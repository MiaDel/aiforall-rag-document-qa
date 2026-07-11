"""
Module implementing a custom LangChain Retriever.
Wraps the custom HybridRetriever and CrossEncoderReranker into the LCEL ecosystem.
"""

import logging
from typing import List, Dict, Any
from pydantic import Field, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker
from config.settings import settings

logger = logging.getLogger(__name__)


class LangChainRAGRetriever(BaseRetriever):
    """
    LangChain wrapper for the custom Hybrid + Cross-Encoder retrieval pipeline.
    """
    top_k_before_rerank: int = Field(default=10)
    top_k_after_rerank: int = Field(default=settings.TOP_K)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[LCDocument]:
        """
        Executes hybrid retrieval followed by neural cross-encoder reranking,
        mapping outputs to standard LangChain Document schemas.
        """
        logger.info(f"Executing LangChain retriever for query: '{query}'")
        try:
            from utils.resource_manager import get_reranker
            # Lazily load and use cached instances
            hybrid_retriever = HybridRetriever()
            reranker = get_reranker()
            
            # 1. Hybrid dense-sparse search
            candidates = hybrid_retriever.retrieve(query, top_k=self.top_k_before_rerank)
            
            # 2. Neural reranking
            reranked = reranker.rerank(
                query=query,
                retrieved_items=candidates,
                top_k_after_rerank=self.top_k_after_rerank
            )
            
            # 3. Translate to LangChain Documents
            lc_documents = []
            for item in reranked:
                doc = item["document"]
                meta = doc.metadata
                
                lc_documents.append(
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
                
            logger.info(f"LangChain retriever completed. Returned {len(lc_documents)} docs.")
            return lc_documents

        except Exception as e:
            logger.error(f"Failed during LangChain retrieval execution: {str(e)}")
            return []
