"""
Top metric cards component for the Loan RAG Chatbot.
Visualizes database and model status values.
"""

import streamlit as st
from config.settings import settings


def render_metrics():
    """
    Renders the four top metrics card dashboard widgets.
    """
    # 1. Fetch dynamic statistics
    try:
        from utils.resource_manager import get_chroma_manager
        chroma_mgr = get_chroma_manager()
        stats = chroma_mgr.get_stats()
        docs_count = stats["unique_documents_count"]
        chunks_count = stats["total_chunks"]
        vectors_count = stats["total_chunks"]
    except Exception:
        docs_count = st.session_state.get("indexed_docs_count", 0)
        chunks_count = st.session_state.get("chunk_count", 0)
        vectors_count = chunks_count

    col1, col2, col3, col4 = st.columns(4)

    # Card 1: Documents
    with col1:
        st.markdown(
            f'<div class="metric-container">'
            f'  <div class="metric-icon icon-blue"><span style="font-size:20px;">📄</span></div>'
            f'  <div>'
            f'    <div class="metric-label">Documents</div>'
            f'    <div class="metric-value">{docs_count}</div>'
            f'    <div class="metric-sub sub-blue">PDF Files</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 2: Chunks
    with col2:
        st.markdown(
            f'<div class="metric-container">'
            f'  <div class="metric-icon icon-green"><span style="font-size:20px;">🥞</span></div>'
            f'  <div>'
            f'    <div class="metric-label">Chunks</div>'
            f'    <div class="metric-value">{chunks_count}</div>'
            f'    <div class="metric-sub sub-green">Text Chunks</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 3: Vectors
    with col3:
        st.markdown(
            f'<div class="metric-container">'
            f'  <div class="metric-icon icon-purple"><span style="font-size:20px;">🧬</span></div>'
            f'  <div>'
            f'    <div class="metric-label">Vectors</div>'
            f'    <div class="metric-value">{vectors_count}</div>'
            f'    <div class="metric-sub sub-purple">Stored Embeddings</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 4: Model
    with col4:
        st.markdown(
            f'<div class="metric-container">'
            f'  <div class="metric-icon icon-blue"><span style="font-size:20px;">🧠</span></div>'
            f'  <div>'
            f'    <div class="metric-label">Model</div>'
            f'    <div class="metric-value" style="font-size:16px; font-weight:700;">Llama3 Local</div>'
            f'    <div class="metric-sub sub-blue">LLM Provider</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
