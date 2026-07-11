"""
Module for Dense Semantic Retrieval querying ChromaDB.
"""

import logging
from typing import List, Dict, Any
from config.settings import settings
from ingestion.pdf_loader import Document
from embeddings.embedder import BGEEmbedder
from vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves document chunks using vector similarity matching.
    """

    def __init__(self, chroma_manager: ChromaManager | None = None):
        """
        Initializes the DenseRetriever with a ChromaManager instance.
        """
        from utils.resource_manager import get_chroma_manager, get_embedding_model
        self.chroma_manager = chroma_manager or get_chroma_manager()
        self.embedder = get_embedding_model()

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k closest chunks using dense embeddings.

        Parameters:
            query: The user prompt.
            top_k: Number of chunks to retrieve (defaults to settings.TOP_K).

        Returns:
            List of dicts containing 'document' (Document), 'score' (float).
        """
        k = top_k or settings.TOP_K
        if not query.strip():
            logger.warning("Empty query received in DenseRetriever.")
            return []

        try:
            # 1. Embed query
            query_vector = self.embedder.embed_query(query)
            
            # 2. Search Chroma
            results = self.chroma_manager.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Check if empty results
            if not results or not results.get("ids") or not results["ids"][0]:
                logger.info(f"No semantic matches found for query: '{query}'")
                return []

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            retrieved_items = []
            for i in range(len(ids)):
                # Chroma distance is cosine distance (1 - cos_sim)
                # Convert to Cosine Similarity score in range [0, 1]
                distance = distances[i]
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
                
                # Check similarity threshold filter
                if similarity_score < settings.SIMILARITY_THRESHOLD:
                    logger.debug(f"Chunk {ids[i]} similarity {similarity_score:.4f} is below threshold. Filtering out.")
                    continue

                meta = metadatas[i]
                
                # Reconstruct standardized Document object
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
                
                retrieved_items.append({
                    "document": doc,
                    "score": similarity_score,
                    "id": ids[i]
                })

            logger.info(f"Dense retrieve matched {len(retrieved_items)} chunks for query: '{query}'")
            return retrieved_items

        except Exception as e:
            logger.error(f"Error during dense retrieval: {str(e)}")
            raise RuntimeError(f"Dense retrieval failure: {str(e)}") from e
