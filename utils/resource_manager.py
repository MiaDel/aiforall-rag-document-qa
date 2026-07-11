"""
Centralized resource manager for caching heavy models, vector database connections,
and compiled workflows. Uses Streamlit's cache_resource to prevent reload delays.
"""

import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# Start up timing global log
STARTUP_TIME = time.time()


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
    """
    logger.info("Initializing ChromaManager via Resource Manager...")
    from vectorstore.chroma_manager import ChromaManager
    return ChromaManager()


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
