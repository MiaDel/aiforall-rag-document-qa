"""
=========================================================
File Name : semantic_chunker.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module contains text splitting and chunking algorithms.
It includes a structure-aware RecursiveCharacterSplitter,
and a similarity-based SemanticChunker that divides page
sentences based on cosine distances of their embeddings.

Technologies:
- NumPy
- Regex (re)
- Python standard logger
=========================================================
"""

import re
import logging
from typing import List, Callable, Dict, Any, Optional
import numpy as np
from ingestion.pdf_loader import Document
from chunking.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class RecursiveCharacterSplitter:
    """
    Splits text recursively using a list of separators.
    Preserves document structure by splitting on paragraph (\n\n),
    sentence (\n), words (space), and finally characters.

    Responsibilities:
    1. Split large blocks recursively.
    2. Enforce character count overlap boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initializes the RecursiveCharacterSplitter.

        Parameters:
            chunk_size (int):
                Maximum character length of each chunk.
            chunk_overlap (int):
                Character overlap length between consecutive chunks.
            separators (Optional[List[str]]):
                Separators list to split on.
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.separators: List[str] = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Internal recursive splitter implementation.

        Workflow:
        1. Gating check if text length is less than chunk_size.
        2. Pick active separator and split target string.
        3. Merge splits sequentially until chunk_size is reached.
        4. Apply overlap buffer step backwards.
        5. Return chunks list.

        Parameters:
            text (str):
                The input string.
            separators (List[str]):
                Sub-list of delimiters remaining.

        Returns:
            List[str]:
                List of string chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [text]

        # Select current separator
        separator: str = separators[0]
        next_separators: List[str] = separators[1:]

        # Split text by separator
        if separator == "":
            splits: List[str] = list(text)
        else:
            splits = text.split(separator)

        chunks: List[str] = []
        current_doc: List[str] = []
        current_len: int = 0

        for split in splits:
            split_len: int = len(split)
            
            # If a single split is larger than chunk_size, split it recursively
            if split_len > self.chunk_size:
                if current_doc:
                    chunks.append(separator.join(current_doc))
                    current_doc = []
                    current_len = 0
                
                # Recursively split the long block
                sub_splits: List[str] = self._split_text(split, next_separators)
                chunks.extend(sub_splits)
                continue

            # Check if adding this split exceeds chunk_size
            separator_len: int = len(separator) if current_doc else 0
            if current_len + split_len + separator_len > self.chunk_size:
                if current_doc:
                    chunks.append(separator.join(current_doc))
                
                # Start new doc, keeping overlap
                # Walk backward to satisfy overlap
                overlap_doc: List[str] = []
                overlap_len: int = 0
                for prev_split in reversed(current_doc):
                    prev_separator_len: int = len(separator) if overlap_doc else 0
                    if overlap_len + len(prev_split) + prev_separator_len <= self.chunk_overlap:
                        overlap_doc.insert(0, prev_split)
                        overlap_len += len(prev_split) + prev_separator_len
                    else:
                        break
                
                current_doc = overlap_doc
                current_len = overlap_len

            # Add split to current chunk
            current_doc.append(split)
            current_len += split_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            chunks.append(separator.join(current_doc))

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Splits a single string into chunks.

        Workflow:
        1. Delegate call to recursive splitter with initial separators.

        Parameters:
            text (str):
                The input string.

        Returns:
            List[str]:
                Splitted text chunks.
        """
        return self._split_text(text, self.separators)


class SemanticChunker:
    """
    Chunks documents semantically using sentence embeddings.
    Computes differences between adjacent sentences and cuts where similarity falls below threshold.

    Responsibilities:
    1. Parse sentences based on regular expressions.
    2. Compute cosine distances between adjacent embeddings.
    3. Split where distance exceeds percentile threshold.
    """

    def __init__(
        self,
        embedding_function: Optional[Callable[[List[str]], np.ndarray]] = None,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        threshold_percentile: float = 85.0
    ):
        """
        Initializes the SemanticChunker.

        Parameters:
            embedding_function (Optional[Callable[[List[str]], np.ndarray]]):
                Callable taking List[str] and returning numpy embedding matrix.
            max_chunk_size (int):
                Max character length if semantic chunk is too large.
            chunk_overlap (int):
                Overlap to use for fallback recursive split.
            threshold_percentile (float):
                Percentile of distance values to split on.
        """
        self.embedding_function: Optional[Callable[[List[str]], np.ndarray]] = embedding_function
        self.max_chunk_size: int = max_chunk_size
        self.threshold_percentile: float = threshold_percentile
        self.recursive_splitter: RecursiveCharacterSplitter = RecursiveCharacterSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using regex pattern matching.

        Workflow:
        1. Match sentence terminal symbols followed by spacing.
        2. Keep decimals and abbreviation boundaries.

        Parameters:
            text (str):
                Input string.

        Returns:
            List[str]:
                List of sentences.
        """
        sentence_end: re.Pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
        sentences: List[str] = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc: Document, total_pages: int) -> List[Document]:
        """
        Splits a Document into semantic chunks and enriches their metadata.

        Workflow:
        1. Return recursively split chunks if no embedder callback exists.
        2. Split text to sentences.
        3. Retrieve vector embeddings matrix.
        4. Compute adjacent cosine similarity distances.
        5. Formulate distance split thresholds.
        6. Group sentences together and map chunk metadata.

        Parameters:
            doc (Document):
                Enriched page-level Document.
            total_pages (int):
                Total pages of the source document.

        Returns:
            List[Document]:
                List of chunk-level Document objects.
        """
        content: str = doc.page_content
        metadata_mgr: MetadataManager = MetadataManager()
        
        # Fallback if no embedding function is provided
        if not self.embedding_function:
            logger.debug(f"No embedding function provided. Falling back to Recursive Splitter for page {doc.metadata.get('page')}")
            text_chunks: List[str] = self.recursive_splitter.split_text(content)
            chunks: List[Document] = []
            for idx, text in enumerate(text_chunks):
                meta: Dict[str, Any] = metadata_mgr.enrich_chunk_metadata(doc.metadata, idx, total_pages)
                chunks.append(Document(page_content=text, metadata=meta))
            return chunks

        # Extract sentences
        sentences: List[str] = self._split_into_sentences(content)
        if not sentences:
            return []

        # If text is too short, treat as single chunk
        if len(content) <= self.max_chunk_size:
            meta = metadata_mgr.enrich_chunk_metadata(doc.metadata, 0, total_pages)
            return [Document(page_content=content, metadata=meta)]

        try:
            # Compute Embeddings
            embeddings: np.ndarray = self.embedding_function(sentences)
            
            # Compute cosine distances between consecutive sentences
            distances: List[float] = []
            for i in range(len(embeddings) - 1):
                vec1: np.ndarray = embeddings[i]
                vec2: np.ndarray = embeddings[i + 1]
                
                norm1: float = np.linalg.norm(vec1)
                norm2: float = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    distance = 1.0
                else:
                    cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    distance = 1.0 - cosine_similarity
                
                distances.append(distance)

            # Determine threshold
            if distances:
                threshold: float = np.percentile(distances, self.threshold_percentile)
            else:
                threshold = 0.5

            # Group sentences into chunks based on threshold
            grouped_chunks: List[str] = []
            current_chunk_sentences: List[str] = [sentences[0]]
            
            for i, distance in enumerate(distances):
                next_sentence: str = sentences[i + 1]
                current_length: int = len(" ".join(current_chunk_sentences))
                
                # Check if distance exceeds threshold or adding next sentence exceeds max_chunk_size
                if distance > threshold or current_length + len(next_sentence) + 1 > self.max_chunk_size:
                    grouped_chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [next_sentence]
                else:
                    current_chunk_sentences.append(next_sentence)
            
            if current_chunk_sentences:
                grouped_chunks.append(" ".join(current_chunk_sentences))

            # Convert groups to Document objects and enrich metadata
            chunks = []
            for idx, text in enumerate(grouped_chunks):
                # If a semantic group somehow exceeds max_chunk_size, split recursively
                if len(text) > self.max_chunk_size:
                    sub_chunks: List[str] = self.recursive_splitter.split_text(text)
                    for sub_idx, sub_text in enumerate(sub_chunks):
                        meta = metadata_mgr.enrich_chunk_metadata(
                            doc.metadata, len(chunks), total_pages
                        )
                        chunks.append(Document(page_content=sub_text, metadata=meta))
                else:
                    meta = metadata_mgr.enrich_chunk_metadata(doc.metadata, len(chunks), total_pages)
                    chunks.append(Document(page_content=text, metadata=meta))

            return chunks

        except Exception as e:
            logger.exception(f"Semantic chunking failed on page {doc.metadata.get('page')}: {str(e)}")
            # Fail gracefully by falling back to recursive splitter
            text_chunks = self.recursive_splitter.split_text(content)
            chunks = []
            for idx, text in enumerate(text_chunks):
                meta = metadata_mgr.enrich_chunk_metadata(doc.metadata, idx, total_pages)
                chunks.append(Document(page_content=text, metadata=meta))
            return chunks
