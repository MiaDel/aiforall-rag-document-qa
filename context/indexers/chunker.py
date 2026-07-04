from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from observability.logger import logger


class TextChunker:
    """
    Splits cleaned text into semantic chunks
    for embedding and retrieval.
    """

    def chunk(
        self,
        text: str,
        metadata: dict,
    ) -> list[Document]:
        """
        Split cleaned text into chunks.
        """
        logger.info("Starting text chunking pipeline.")
        
        self._validate_input(text, metadata)

        splitter = self._create_splitter()

        chunks = self._split_text(
            splitter,
            text,
            metadata,
        )

        logger.info(
            f"Successfully created {len(chunks)} chunks from document."
        )

        return chunks
    
    def _validate_input(
        self,
        text: str,
        metadata: dict,
    ) -> None:
        """
        Validate input before chunking.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Input text must be a string."
            )

        if not text.strip():
            raise ValueError(
                "Input text is empty."
            )

        if not isinstance(metadata, dict):
            raise TypeError(
                "Metadata must be a dictionary."
            )

    def _create_splitter(
        self,
    ) -> RecursiveCharacterTextSplitter:
        """
        Create the text splitter used for chunking.
        """

        logger.info("Initializing RecursiveCharacterTextSplitter.")

        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def _split_text(
        self,
        splitter: RecursiveCharacterTextSplitter,
        text: str,
        metadata: dict,
    ) -> list[Document]:
        """
        Split text into LangChain Document chunks.
        """

        logger.info("Splitting text into chunks.")

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        return chunks