"""
=========================================================
File Name : embedder.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module manages vector embedding generations using local
SentenceTransformer model BAAI/bge-large-en-v1.5. It implements
the singleton pattern to prevent memory leaks and double loading.

Technologies:
- Sentence Transformers
- PyTorch
- NumPy
=========================================================
"""

import logging
import torch
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    Singleton wrapper for sentence-transformers local BGE embedding execution.

    Responsibilities:
    1. Load SentenceTransformer weights exactly once.
    2. Convert document lists to dense vector representation.
    3. Format search queries using recommended BGE prefixes.
    """
    _instance: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "BGEEmbedder":
        """
        Creates or returns the active singleton instance.

        Parameters:
            *args: Variable arguments list.
            **kwargs: Keyword arguments dict.

        Returns:
            BGEEmbedder:
                The singleton BGEEmbedder instance.
        """
        if not cls._instance:
            cls._instance = super(BGEEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initializes the model on CPU or GPU device using local fallback paths.
        """
        if self._initialized:
            return

        import os
        from pathlib import Path

        # Check if model was pre-saved to /app/models during Docker build
        local_model_dir = Path("./models").resolve()
        if local_model_dir.exists() and any(local_model_dir.iterdir()):
            self.model_name: str = str(local_model_dir)
            logger.info(f"Found pre-downloaded local model at '{self.model_name}'")
        else:
            self.model_name: str = settings.EMBEDDING_MODEL_NAME

        # Automatically detect device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        
        try:
            # Load model directly from local directory if present
            self.model: SentenceTransformer = SentenceTransformer(
                self.model_name, 
                device=self.device
            )
            logger.info("Embedding model loaded successfully.")
            self._initialized = True
        except Exception as e:
            logger.exception(f"Failed to load sentence transformer model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}") from e

    # def __init__(self) -> None:
    #     """
    #     Initializes the model on CPU or GPU device.

    #     Workflow:
    #     1. Gating check if already initialized.
    #     2. Set GPU/CPU device based on CUDA availability.
    #     3. Load SentenceTransformer from models directory.

    #     Parameters:
    #         None

    #     Returns:
    #         None

    #     Raises:
    #         RuntimeError: If model weights fail to load.
    #     """
    #     if self._initialized:
    #         return

    #     self.model_name: str = settings.EMBEDDING_MODEL_NAME
    #     # Automatically detect device
    #     self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #     logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        
    #     try:
    #         self.model: SentenceTransformer = SentenceTransformer(self.model_name, device=self.device, cache_folder="./models")
    #         logger.info("Embedding model loaded successfully.")
    #         self._initialized = True
    #     except Exception as e:
    #         logger.exception(f"Failed to load sentence transformer model '{self.model_name}': {str(e)}")
    #         raise RuntimeError(f"Embedding model initialization failed: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document strings.
        No retrieval prefix is required for document encoding under BGE guidelines.

        Workflow:
        1. Validate non-empty list input.
        2. Call sentence_transformers.encode in batches.
        3. Normalize embeddings for cosine distance checks.

        Parameters:
            texts (List[str]):
                List of document strings to embed.

        Returns:
            List[List[float]]:
                List of float lists representing dense vectors.

        Raises:
            RuntimeError: If vector generation fails.
        """
        if not texts:
            return []
            
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts...")
            embeddings: np.ndarray = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            # Convert numpy array to list of lists of floats
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise RuntimeError(f"Document embedding failed: {str(e)}") from e

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        Appends the recommended BGE query prefix for retrieval:
        'Represent this sentence for searching relevant passages: '

        Workflow:
        1. Prefix query with the mandatory BGE search prompt.
        2. Generate normalized dense vector.

        Parameters:
            query (str):
                The search query string.

        Returns:
            List[float]:
                List of floats representing the dense query vector.

        Raises:
            RuntimeError: If vector generation fails.
        """
        if not query:
            return []
            
        try:
            # BGE large v1.5 requires query instruction for retrieval tasks
            query_prefix: str = "Represent this sentence for searching relevant passages: "
            prefixed_query: str = f"{query_prefix}{query}"
            
            logger.debug(f"Generating query embedding...")
            embedding: np.ndarray = self.model.encode(
                prefixed_query,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise RuntimeError(f"Query embedding failed: {str(e)}") from e

    def get_embedding_dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model.
        For bge-large-en-v1.5, this is 1024.

        Parameters:
            None

        Returns:
            int:
                Dimension length.
        """
        return self.model.get_embedding_dimension()
