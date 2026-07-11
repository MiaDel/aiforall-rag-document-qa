"""
Module for Hybrid Retrieval combining Dense Semantic Search and BM25 Lexical Search.
Integrates results using Reciprocal Rank Fusion (RRF) and normalizes combined relevance.
"""

import logging
from typing import List, Dict, Any
from config.settings import settings
from vectorstore.chroma_manager import ChromaManager
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Ensemble retriever using Reciprocal Rank Fusion (RRF) to merge dense and sparse matches.
    """

    def __init__(
        self,
        chroma_manager: ChromaManager | None = None,
        rrf_k: int = 60
    ):
        """
        Initializes the HybridRetriever.

        Parameters:
            chroma_manager: ChromaManager database connector.
            rrf_k: RRF rank smoothing constant (defaults to standard 60).
        """
        from utils.resource_manager import get_chroma_manager
        self.chroma_manager = chroma_manager or get_chroma_manager()
        self.dense_retriever = DenseRetriever(chroma_manager=self.chroma_manager)
        self.bm25_retriever = BM25Retriever(chroma_manager=self.chroma_manager)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k hybrid matches combining dense and sparse indices.

        Parameters:
            query: User prompt.
            top_k: Number of fused results to yield (defaults to settings.TOP_K).

        Returns:
            List of dicts containing 'document' (Document), 'score' (float).
        """
        import time
        start_time = time.time()
        
        k = top_k or settings.TOP_K
        if not query.strip():
            logger.warning("Empty query received in HybridRetriever.")
            return []

        try:
            # 1. Retrieve candidates from both sources
            # Fetch twice the requested top_k from each to ensure high-recall fusion
            candidate_k = k * 2
            
            logger.debug(f"Executing dense candidate retrieval (K={candidate_k})...")
            dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
            
            logger.debug(f"Executing sparse candidate retrieval (K={candidate_k})...")
            sparse_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

            if not dense_results and not sparse_results:
                logger.info(f"No candidates found in hybrid search for: '{query}'")
                return []

            # 2. Reciprocal Rank Fusion (RRF) scoring
            # RRF_score = Sum( 1 / (rrf_k + rank) )
            rrf_scores = {}
            doc_map = {}

            # Process dense ranks
            for rank, item in enumerate(dense_results, start=1):
                doc = item["document"]
                chunk_id = doc.metadata["chunk_id"]
                doc_map[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

            # Process sparse ranks (automatically merges duplicate entries by chunk_id)
            for rank, item in enumerate(sparse_results, start=1):
                doc = item["document"]
                chunk_id = doc.metadata["chunk_id"]
                doc_map[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

            # 3. Sort by fused RRF scores descending
            sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            # 4. Normalize RRF scores to [0.0, 1.0] for similarity comparison
            max_rrf = sorted_candidates[0][1]
            min_rrf = sorted_candidates[-1][1]
            rrf_range = max_rrf - min_rrf

            fused_items = []
            for chunk_id, rrf_score in sorted_candidates:
                if rrf_range == 0.0:
                    normalized_score = 1.0 if max_rrf > 0.0 else 0.0
                else:
                    normalized_score = (rrf_score - min_rrf) / rrf_range
                
                fused_items.append({
                    "document": doc_map[chunk_id],
                    "score": normalized_score,
                    "id": chunk_id
                })

            # 5. Crop to requested top_k limit
            fused_items = fused_items[:k]
            
            elapsed = time.time() - start_time
            logger.info(f"[Timing] Retrieval time: {elapsed:.4f}s")
            print(f"[Timing] Retrieval time: {elapsed:.4f}s")
            logger.info(f"Hybrid RRF retrieve merged {len(fused_items)} chunks for query: '{query}'")
            return fused_items

        except Exception as e:
            logger.exception(f"Error during hybrid retrieval: {str(e)}")
            raise RuntimeError(f"Hybrid retrieval failure: {str(e)}") from e
