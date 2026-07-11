"""
Module for neural reranking of retrieved documents using a Cross-Encoder.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to evaluate prompt-chunk pairs.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from config.settings import settings
from ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Singleton class managing the CrossEncoder reranker model.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = settings.RERANKER_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        logger.info(f"Loading Cross-Encoder model '{self.model_name}' on device '{self.device}'...")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device, cache_folder="./models")
            logger.info("Cross-Encoder model loaded successfully.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model '{self.model_name}': {str(e)}")
            # We do not raise an exception, allowing fallback graceful degradation
            self._initialized = False

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Sigmoid activation to map logits to [0.0, 1.0].
        """
        return 1.0 / (1.0 + np.exp(-x))

    def rerank(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        top_k_after_rerank: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved items based on Cross-Encoder scoring.

        Parameters:
            query: The user prompt.
            retrieved_items: List of retrieved dicts containing 'document' and 'score'.
            top_k_after_rerank: Number of final items to return.

        Returns:
            Reranked list of dicts.
        """
        k = top_k_after_rerank or settings.TOP_K
        
        # 1. Handle empty inputs
        if not retrieved_items:
            logger.info("No documents provided for reranking.")
            return []
            
        if not query.strip():
            logger.warning("Empty query received for reranking. Returning items unchanged.")
            return retrieved_items[:k]

        # 2. Check model loading state (Fallback routing)
        if not self.model:
            logger.warning("Reranking model not loaded. Skipping reranking and returning original order.")
            return retrieved_items[:k]

        try:
            # 3. Deduplicate candidates by chunk_id
            seen_ids = set()
            unique_items = []
            for item in retrieved_items:
                doc = item.get("document")
                if not doc or not isinstance(doc, Document):
                    logger.warning("Corrupted document block in candidates. Skipping.")
                    continue
                
                meta = doc.metadata
                chunk_id = meta.get("chunk_id")
                if not chunk_id:
                    logger.warning("Document chunk missing chunk_id metadata. Skipping.")
                    continue
                    
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_items.append(item)
            
            if not unique_items:
                return []

            # 4. Predict relevance scores
            # Pairs: [(query, doc1), (query, doc2), ...]
            pairs = [[query, item["document"].page_content] for item in unique_items]
            
            logger.debug(f"Predicting relevance scores for {len(pairs)} pairs...")
            raw_scores = self.model.predict(pairs)
            
            # Convert single float outputs (if predict returns raw float instead of list)
            if isinstance(raw_scores, (int, float)):
                raw_scores = [raw_scores]

            # 5. Format and normalize output scores
            reranked_items = []
            for idx, raw_score in enumerate(raw_scores):
                item = unique_items[idx]
                meta = item["document"].metadata
                
                # Apply sigmoid normalization to logit scores
                normalized_score = float(self._sigmoid(raw_score))
                
                # Ensure all metadata fields are preserved/fixed if corrupted
                doc = Document(
                    page_content=item["document"].page_content,
                    metadata={
                        "doc_id": meta.get("doc_id", "Unknown"),
                        "file_name": meta.get("file_name", "Unknown"),
                        "page_number": meta.get("page_number", 1),
                        "chunk_id": meta.get("chunk_id", "Unknown"),
                        "section_title": meta.get("section_title", "Unknown"),
                        "source": meta.get("source", "Unknown"),
                        "upload_timestamp": meta.get("upload_timestamp", ""),
                        "total_pages": meta.get("total_pages", 1)
                    }
                )
                
                reranked_items.append({
                    "document": doc,
                    "score": normalized_score,
                    "id": meta.get("chunk_id")
                })

            # 6. Sort by reranked score descending
            reranked_items.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Successfully reranked {len(reranked_items)} items. Returning top {min(k, len(reranked_items))}.")
            return reranked_items[:k]

        except Exception as e:
            logger.exception(f"Unexpected failure during reranking: {str(e)}")
            # Fail gracefully: return original list truncated
            return retrieved_items[:k]
