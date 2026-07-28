"""
=========================================================
File Name : metrics.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Top metric cards component for the Loan RAG Chatbot.
Visualizes database and model status values.

Technologies:
- Streamlit
=========================================================
"""

import streamlit as st
from config.settings import settings


def render_metrics() -> None:
    """
    Renders the three top metrics card dashboard widgets.
    Shows status indicators for the assistant, active model, and total documents.
    """
    # 1. Fetch dynamic document count statistics from Chroma DB
    try:
        from utils.resource_manager import get_chroma_manager
        chroma_mgr = get_chroma_manager()
        stats = chroma_mgr.get_stats()
        docs_count = stats.get("unique_documents_count", 0)
    except Exception:
        docs_count = st.session_state.get("indexed_docs_count", 0)

    # 2. Query LLM host connectivity status
    from llms.llm_router import check_llama_connection
    is_llama_online: bool = check_llama_connection()
    status_label: str = "Ready" if is_llama_online else "Offline"
    status_color: str = "rgba(16, 185, 129, 0.15)" if is_llama_online else "rgba(239, 68, 68, 0.15)"
    status_text_color: str = "#10B981" if is_llama_online else "#EF4444"
    status_sub: str = "Systems Operational" if is_llama_online else "Ollama Offline"

    col1, col2, col3 = st.columns(3)

    # Card 1: AI Assistant Ready
    with col1:
        st.markdown(
            f'<div class="saas-metric-card">'
            f'  <div class="saas-metric-icon" style="background: {status_color}; color: {status_text_color};">🤖</div>'
            f'  <div>'
            f'    <div class="metric-label" style="font-size: 12px; color: var(--text-muted);">AI Assistant</div>'
            f'    <div class="metric-value" style="font-size: 18px; font-weight: 700; color: var(--text-main);">{status_label}</div>'
            f'    <div class="metric-sub" style="font-size: 11px; color: {status_text_color};">🟢 {status_sub}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 2: Active Model
    with col2:
        st.markdown(
            f'<div class="saas-metric-card">'
            f'  <div class="saas-metric-icon" style="background: rgba(99, 102, 241, 0.1); color: var(--primary);">🧠</div>'
            f'  <div>'
            f'    <div class="metric-label" style="font-size: 12px; color: var(--text-muted);">Active Model</div>'
            f'    <div class="metric-value" style="font-size: 18px; font-weight: 700; color: var(--text-main);">Llama3 Local</div>'
            f'    <div class="metric-sub" style="font-size: 11px; color: var(--text-muted);">Ollama LLM</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Card 3: Uploaded Documents
    with col3:
        st.markdown(
            f'<div class="saas-metric-card">'
            f'  <div class="saas-metric-icon" style="background: rgba(139, 92, 246, 0.1); color: var(--secondary);">📄</div>'
            f'  <div>'
            f'    <div class="metric-label" style="font-size: 12px; color: var(--text-muted);">Uploaded Documents</div>'
            f'    <div class="metric-value" style="font-size: 18px; font-weight: 700; color: var(--text-main);">{docs_count}</div>'
            f'    <div class="metric-sub" style="font-size: 11px; color: var(--text-muted);">Indexed Files</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
