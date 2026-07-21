"""
Centralized resource manager for caching heavy models, vector database connections,
and compiled workflows. Uses Streamlit's cache_resource to prevent reload delays.
"""

import time
import logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

# Start up timing global log
STARTUP_TIME = time.time()

CHROMA_DIR = Path("data/chroma_db")
SAMPLE_DOCS_DIR = Path("data/sample_docs")


def _chroma_is_empty() -> bool:
    return not CHROMA_DIR.exists() or not any(CHROMA_DIR.glob("*"))


def _bootstrap_vectorstore(chroma_manager, embedder):
    """
    Builds the vectorstore from data/sample_docs/ on first run,
    so a fresh clone/fork has a working index without needing
    the (gitignored) chroma_db to be shipped in the repo.
    """
    from ingestion.pdf_loader import load_pdf
    from ingestion.cleaner import clean_text
    from ingestion.metadata_builder import build_metadata
    from chunking.semantic_chunker import semantic_chunk
    from chunking.metadata_manager import get_doc_hash

    pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No sample PDFs found in {SAMPLE_DOCS_DIR} — vectorstore left empty.")
        return

    logger.info(f"Bootstrapping vectorstore from {len(pdfs)} sample doc(s)...")
    start = time.time()
    for pdf_path in pdfs:
        raw_text = load_pdf(pdf_path)
        cleaned = clean_text(raw_text)
        metadata = build_metadata(cleaned, source=pdf_path.name)
        doc_hash = get_doc_hash(cleaned)
        chunks = semantic_chunk(cleaned, embedder)
        chroma_manager.add_documents(chunks, metadata, doc_id=doc_hash)
    elapsed = time.time() - start
    logger.info(f"[Timing] Vectorstore bootstrap time: {elapsed:.4f}s")
    print(f"[Timing] Vectorstore bootstrap time: {elapsed:.4f}s")


@st.cache_resource
def get_embedding_model():
    """
    Caches and returns the BGEEmbedder singleton instance.
    Logs embedding model load time.
    """
    start = time.time()
    logger.info("Initializing BGEEmbedder via Resource Manager...")
    from embeddings.embedder import BGEEmbedder
    embedder = BGEEmbedder()
    elapsed = time.time() - start
    logger.info(f"[Timing] Embedding load time: {elapsed:.4f}s")
    print(f"[Timing] Embedding load time: {elapsed:.4f}s")
    return embedder


@st.cache_resource
def get_reranker():
    """
    Caches and returns the CrossEncoderReranker instance.
    Logs CrossEncoder model load time.
    """
    start = time.time()
    logger.info("Initializing CrossEncoderReranker via Resource Manager...")
    from retrievers.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    elapsed = time.time() - start
    logger.info(f"[Timing] CrossEncoder load time: {elapsed:.4f}s")
    print(f"[Timing] CrossEncoder load time: {elapsed:.4f}s")
    return reranker


@st.cache_resource
def get_chroma_manager():
    """
    Caches and returns the persistent ChromaManager client instance.
    Prevents repeated vector database reconnections.
    On first run (empty/missing chroma_db), auto-builds the vectorstore
    from data/sample_docs/ so forks work out of the box.
    """
    logger.info("Initializing ChromaManager via Resource Manager...")
    from vectorstore.chroma_manager import ChromaManager
    chroma_manager = ChromaManager()

    if _chroma_is_empty():
        with st.spinner("First-time setup: building vectorstore from sample documents..."):
            embedder = get_embedding_model()
            _bootstrap_vectorstore(chroma_manager, embedder)

    return chroma_manager


@st.cache_resource
def get_graph():
    """
    Caches and returns the compiled LangGraph LoanRAGGraph instance.
    Prevents repeated graph compilation on Streamlit reruns.
    """
    logger.info("Initializing LoanRAGGraph via Resource Manager...")
    from graph.loan_rag_graph import LoanRAGGraph
    return LoanRAGGraph()


@st.cache_resource
def get_llm_router():
    """
    Caches and returns the LLMRouter priority router.
    """
    logger.info("Initializing LLMRouter via Resource Manager...")
    from llms.llm_router import LLMRouter
    return LLMRouter()


# Log total application startup time on import
elapsed_startup = time.time() - STARTUP_TIME
logger.info(f"[Timing] Application startup time: {elapsed_startup:.4f}s")
print(f"[Timing] Application startup time: {elapsed_startup:.4f}s")