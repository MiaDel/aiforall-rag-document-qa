"""
Module for Sparse Lexical Retrieval using BM25.
"""

import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from config.settings import settings
from ingestion.pdf_loader import Document
from vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Retrieves document chunks using sparse BM25 keyword matching.
    """

    def __init__(self, chroma_manager: ChromaManager | None = None):
        """
        Initializes the BM25Retriever with a ChromaManager instance.
        """
        from utils.resource_manager import get_chroma_manager
        self.chroma_manager = chroma_manager or get_chroma_manager()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple alphanumeric tokenizer that lowercases input text.
        """
        return re.findall(r"\b\w{2,}\b", text.lower())

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k chunks based on BM25 lexical relevance.

        Parameters:
            query: User prompt.
            top_k: Number of chunks to retrieve (defaults to settings.TOP_K).

        Returns:
            List of dicts containing 'document' (Document), 'score' (float).
        """
        k = top_k or settings.TOP_K
        if not query.strip():
            logger.warning("Empty query received in BM25Retriever.")
            return []

        try:
            # 1. Fetch all documents from ChromaDB
            db_docs = self.chroma_manager.collection.get(include=["documents", "metadatas"])
            
            if not db_docs or not db_docs.get("ids"):
                logger.info("Chroma database is empty. BM25 retrieval skipped.")
                return []

            ids = db_docs["ids"]
            documents = db_docs["documents"]
            metadatas = db_docs["metadatas"]
            total_docs = len(ids)

            # 2. Tokenize corpus
            tokenized_corpus = [self.tokenize(doc) for doc in documents]
            
            # 3. Fit BM25
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 4. Score query
            tokenized_query = self.tokenize(query)
            if not tokenized_query:
                logger.warning(f"No tokens found in search query: '{query}'")
                return []

            scores = bm25.get_scores(tokenized_query)
            
            max_score = float(max(scores))
            min_score = float(min(scores))
            score_range = max_score - min_score

            # If all scores are 0, there are no matches
            if max_score <= 0.0:
                logger.info(f"No BM25 matches found for query: '{query}'")
                return []

            # 5. Compile and normalize scores
            scored_items = []
            for i in range(total_docs):
                raw_score = float(scores[i])
                if raw_score <= 0.0:
                    continue  # Skip chunks with zero keyword matching

                # Min-Max Normalization to [0.0, 1.0]
                if score_range == 0.0:
                    normalized_score = 1.0
                else:
                    normalized_score = (raw_score - min_score) / score_range

                meta = metadatas[i]
                doc = Document(
                    page_content=documents[i],
                    metadata={
                        "doc_id": meta.get("doc_id"),
                        "file_name": meta.get("file_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "section_title": meta.get("section_title"),
                        "source": meta.get("source"),
                        "upload_timestamp": meta.get("upload_timestamp"),
                        "total_pages": meta.get("total_pages")
                    }
                )
                
                scored_items.append({
                    "document": doc,
                    "score": normalized_score,
                    "id": ids[i]
                })

            # 6. Sort and get top_k
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            top_items = scored_items[:k]

            logger.info(f"BM25 retrieve matched {len(top_items)} chunks for query: '{query}'")
            return top_items

        except Exception as e:
            logger.exception(f"Error during BM25 retrieval: {str(e)}")
            raise RuntimeError(f"BM25 retrieval failure: {str(e)}") from e
