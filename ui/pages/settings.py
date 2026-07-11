"""
Settings Page component.
Displays loaded environments, settings, and thresholds.
"""

import streamlit as st
from config.settings import settings


def run_settings_page():
    """
    Renders the settings page.
    """
    st.title("⚙️ System Settings")
    st.write("Below are the active configuration variables loaded from `.env` or system defaults:")
    
    st.markdown("### 1. Vector Store Configurations")
    st.json({
        "CHROMA_PERSIST_DIRECTORY": settings.CHROMA_PERSIST_DIRECTORY,
        "COLLECTION_NAME": settings.COLLECTION_NAME,
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP
    })

    st.markdown("### 2. Retrieval Parameters")
    st.json({
        "TOP_K": settings.TOP_K,
        "SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD
    })

    st.markdown("### 3. Model Configuration Settings")
    st.json({
        "EMBEDDING_MODEL_NAME": settings.EMBEDDING_MODEL_NAME,
        "RERANKER_MODEL_NAME": settings.RERANKER_MODEL_NAME,
        "OLLAMA_HOST": settings.OLLAMA_HOST,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL
    })

    st.markdown("### 4. API Key Integrity Checks")
    st.json({
        "GEMINI_API_KEY_CONFIGURED": bool(settings.GEMINI_API_KEY),
        "GROQ_API_KEY_CONFIGURED": bool(settings.GROQ_API_KEY),
        "OPENAI_API_KEY_CONFIGURED": bool(settings.OPENAI_API_KEY)
    })
