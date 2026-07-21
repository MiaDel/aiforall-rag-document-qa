"""
=========================================================
File Name : chroma_manager.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module manages connection, indexing, updates, deletions,
and stats retrieval operations for persistent vector store ChromaDB.
It checks for duplicate doc_ids using file hashes (SHA-256)
to guarantee idempotent document indexing.

Technologies:
- ChromaDB
- Sentence Transformers (BGE embeddings)
- Python Standard hashlib Library
=========================================================
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
import chromadb
from config.settings import settings
from ingestion.pdf_loader import Document
from embeddings.embedder import BGEEmbedder

logger = logging.getLogger(__name__)


class ChromaManager:
    """
    Handles connection, duplicate checking, document upserts, deletions, and queries for ChromaDB.

    Responsibilities:
    1. Connect to Chroma persistent client store.
    2. Idempotently insert text chunk vectors.
    3. Generate document SHA-256 file hashes.
    4. Delete matching documents from vector database.
    5. Retrieve total counts and collection status values.
    """

    def __init__(self, collection_name: Optional[str] = None) -> None:
        """
        Initializes the Chroma Persistent client and gets or creates the collection.

        Workflow:
        1. Resolve database persistent directory path.
        2. Instantiate chromadb.PersistentClient.
        3. Get cached embedding model instance.
        4. Recreate or fetch collection.

        Parameters:
            collection_name (Optional[str]):
                Custom collection name to load.

        Returns:
            None

        Raises:
            RuntimeError: If connection to ChromaDB persistent store fails.
        """
        self.persist_dir: Path = Path(settings.CHROMA_PERSIST_DIRECTORY).resolve()
        self.collection_name: str = collection_name or settings.COLLECTION_NAME
        
        logger.info(f"Connecting to persistent Chroma DB at '{self.persist_dir}'...")
        
        try:
            # Ensure the directory is created
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.client: chromadb.PersistentClient = chromadb.PersistentClient(path=str(self.persist_dir))
            from utils.resource_manager import get_embedding_model
            self.embedder = get_embedding_model()
            
            # Safely recreate or get collection
            self._init_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB persistent client: {str(e)}")
            raise RuntimeError(f"ChromaDB connection failure: {str(e)}") from e

    def _init_collection(self) -> None:
        """
        Initializes the collection in ChromaDB.

        Workflow:
        1. Query client to create or retrieve collection with cosine metric space.

        Parameters:
            None

        Returns:
            None

        Raises:
            RuntimeError: If collection initialization fails.
        """
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Connected to Chroma collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error creating/fetching collection '{self.collection_name}': {str(e)}")
            raise RuntimeError(f"Failed to initialize Chroma collection: {str(e)}") from e

    @staticmethod
    def calculate_file_hash(file_path: Union[Path, str]) -> str:
        """
        Calculates SHA-256 hash of a file for duplicate detection.

        Workflow:
        1. Resolve absolute file path.
        2. Scan file in bytes blocks.
        3. Compute SHA-256 hex digest.

        Parameters:
            file_path (Path | str):
                Path to the target file.

        Returns:
            str:
                SHA-256 hex string.
        """
        path: Path = Path(file_path).resolve()
        if not path.exists():
            # Generate deterministic fallback hash based on name if file is deleted
            return hashlib.sha256(path.name.encode()).hexdigest()
            
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256.update(byte_block)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed calculating hash for {path.name}: {str(e)}")
            # Fallback to simple name hashing
            return hashlib.sha256(path.name.encode()).hexdigest()

    def is_document_indexed(self, doc_id: str) -> bool:
        """
        Checks if a document with this doc_id is already present in ChromaDB.

        Workflow:
        1. Query the collection using doc_id metadata constraint.
        2. Verify if results ids count is greater than zero.

        Parameters:
            doc_id (str):
                Unique document identifier.

        Returns:
            bool:
                True if document exists.
        """
        try:
            results = self.collection.get(where={"doc_id": doc_id}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error querying Chroma for document existence: {str(e)}")
            return False

    def add_documents(self, documents: List[Document], overwrite: bool = False) -> bool:
        """
        Idempotently inserts document chunks. Checks for duplicate doc_id (file hash).

        Workflow:
        1. Inspect parent document id and metadata.
        2. Skip or delete index if duplicate doc_id is detected.
        3. Generate dense vectors using BGEEmbedder.
        4. Upsert elements to Chroma DB in batches.

        Parameters:
            documents (List[Document]):
                List of chunk-level Document objects.
            overwrite (bool):
                If True, replaces existing document index instead of skipping.

        Returns:
            bool:
                True if indexing succeeded or was skipped.
        
        Raises:
            RuntimeError: If embedding generation or storage fails.
        """
        if not documents:
            logger.warning("No documents provided to index.")
            return False

        # Gather document properties from the first chunk
        first_doc: Document = documents[0]
        doc_id: Optional[str] = first_doc.metadata.get("doc_id")
        file_name: str = first_doc.metadata.get("file_name", "Unknown")
        
        if not doc_id:
            logger.error(f"Cannot index document {file_name}: Missing 'doc_id' metadata.")
            raise ValueError(f"Metadata check failed: 'doc_id' is missing from chunks.")

        # Duplicate detection check
        if self.is_document_indexed(doc_id):
            if not overwrite:
                logger.warning(f"Document '{file_name}' (ID: {doc_id}) is already indexed. Skipping.")
                return True  # Skipped idempotently
            else:
                logger.info(f"Overwriting index for document '{file_name}' (ID: {doc_id})...")
                self.delete_document(doc_id)

        try:
            texts: List[str] = [doc.page_content for doc in documents]
            ids: List[str] = [doc.metadata["chunk_id"] for doc in documents]
            metadatas: List[Dict[str, Any]] = [doc.metadata for doc in documents]
            
            # Generate embeddings locally via BGEEmbedder
            try:
                embeddings: List[List[float]] = self.embedder.embed_documents(texts)
            except Exception as emb_err:
                logger.error(f"Embedding generation failure: {str(emb_err)}")
                raise RuntimeError(f"Embedding generation failed: {str(emb_err)}") from emb_err
            
            # Print diagnostics
            print("Chunks:", len(documents))
            print("Embeddings:", len(embeddings))
            print("Collection count before:", self.collection.count())
            
            logger.info(f"Storing {len(documents)} chunks in ChromaDB...")
            
            # Write to Chroma in batches
            batch_size: int = 100
            for i in range(0, len(documents), batch_size):
                end_idx: int = min(i + batch_size, len(documents))
                self.collection.upsert(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    documents=texts[i:end_idx]
                )
            
            print("Collection count after:", self.collection.count())
            print("Persist directory:", self.persist_dir)
            
            logger.info(f"Successfully indexed document '{file_name}' ({len(documents)} chunks).")
            return True
        except Exception as e:
            logger.exception(f"Failed to add document '{file_name}' to ChromaDB: {str(e)}")
            raise RuntimeError(f"Chroma insertion failed: {str(e)}") from e

    def delete_document(self, doc_id: str) -> int:
        """
        Deletes all chunks associated with a specific document ID.

        Workflow:
        1. Fetch collection ids matching doc_id.
        2. Call delete command on collection using doc_id metadata query.

        Parameters:
            doc_id (str):
                The document hash to delete.

        Returns:
            int:
                Number of chunks deleted.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            logger.info(f"Requesting deletion of document ID: {doc_id}")
            
            results = self.collection.get(where={"doc_id": doc_id})
            chunks_to_delete: int = len(results.get("ids", []))
            
            if chunks_to_delete > 0:
                self.collection.delete(where={"doc_id": doc_id})
                logger.info(f"Successfully deleted {chunks_to_delete} chunks for doc: {doc_id}")
            else:
                logger.warning(f"No chunks found in database for document ID: {doc_id}")
                
            return chunks_to_delete
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id} from Chroma: {str(e)}")
            raise RuntimeError(f"Chroma deletion failed: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves database collection statistics.

        Workflow:
        1. Get collection count value.
        2. Query all document metadata items.
        3. Collect set of unique doc_id entries.

        Parameters:
            None

        Returns:
            Dict[str, Any]:
                Dict containing totals for chunks and unique documents.
        """
        try:
            total_chunks: int = self.collection.count()
            
            unique_docs: Set[str] = set()
            if total_chunks > 0:
                results = self.collection.get(include=["metadatas"])
                for meta in results.get("metadatas", []):
                    if meta and "doc_id" in meta:
                        unique_docs.add(meta["doc_id"])
            
            return {
                "total_chunks": total_chunks,
                "unique_documents_count": len(unique_docs),
                "unique_document_ids": list(unique_docs)
            }
        except Exception as e:
            logger.error(f"Error fetching Chroma collection statistics: {str(e)}")
            return {"total_chunks": 0, "unique_documents_count": 0, "unique_document_ids": []}

    def reset_database(self) -> None:
        """
        Resets and clears the current collection.

        Workflow:
        1. Fetch list of all active unique doc_ids.
        2. Call delete_document sequentially.

        Parameters:
            None

        Returns:
            None

        Raises:
            RuntimeError: If resetting collection fails.
        """
        try:
            logger.warning("Resetting collection in ChromaDB...")
            stats: Dict[str, Any] = self.get_stats()
            for doc_id in stats["unique_document_ids"]:
                self.delete_document(doc_id)
            logger.info("Chroma collection reset successfully.")
        except Exception as e:
            logger.error(f"Failed resetting collection: {str(e)}")
            raise RuntimeError(f"Reset failed: {str(e)}") from e
