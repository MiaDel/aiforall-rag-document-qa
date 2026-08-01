"""
=========================================================
File Name : metrics.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman, Mia Delhalle
Description:
Top metric cards component for the Loan RAG Chatbot.
Visualizes database and model status values.

Technologies:
- Streamlit
=========================================================
"""

import os
import streamlit as st
from config.settings import settings
from pathlib import Path


# def get_current_llm_provider() -> str:
#     """Retrieves the active LLM provider label dynamically."""
#     provider = None

#     # 1. Check Streamlit Secrets
#     try:
#         if "LLM_PROVIDER" in st.secrets:
#             provider = st.secrets["LLM_PROVIDER"]
#     except Exception:
#         pass

#     # 2. Check Environment Variables
#     if not provider:
#         provider = os.getenv("LLM_PROVIDER")

#     # 3. Check Settings
#     if not provider and hasattr(settings, "LLM_PROVIDER"):
#         provider = settings.LLM_PROVIDER

#     if not provider:
#         return "Unknown"

#     provider_map = {
#         "groq": "Groq (Llama 3.3)",
#         "openai": "OpenAI GPT-4o",
#         "gemini": "Google Gemini",
#         "ollama": "Llama3 Local",
#         "openrouter": "OpenRouter",
#     }
#     return provider_map.get(provider.lower(), provider.title())


# def get_current_llm_provider() -> str:
#     """Safely retrieves the LLM provider name without throwing missing secrets warnings."""
#     provider = None

#     # 1. Check Streamlit secrets ONLY if secrets file exists
#     try:
#         # Avoid accessing st.secrets directly if no secrets file is loaded
#         if hasattr(st, "secrets") and len(st.secrets) > 0 and "LLM_PROVIDER" in st.secrets:
#             provider = st.secrets["LLM_PROVIDER"]
#     except Exception:
#         pass

#     # 2. Check Environment Variables (Primary method on Docker / AWS ECS)
#     if not provider:
#         provider = os.getenv("LLM_PROVIDER")

#     # 3. Check Settings fallback
#     if not provider and hasattr(settings, "LLM_PROVIDER"):
#         provider = settings.LLM_PROVIDER

#     if not provider:
#         return "Unknown"

#     provider_map = {
#         "groq": "Groq (Llama 3.3 70B)",
#         "openai": "OpenAI (GPT-4o)",
#         "gemini": "Google Gemini",
#         "ollama": "Llama3 Local",
#         "openrouter": "OpenRouter",
#     }
#     return provider_map.get(provider.lower(), provider.title())


def get_current_llm_provider() -> str:
    """Safely retrieves the LLM provider name without triggering missing secrets warnings."""
    provider = None

    # 1. Check if a local/container secrets file ACTUALLY exists before accessing st.secrets
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            if "LLM_PROVIDER" in st.secrets:
                provider = st.secrets["LLM_PROVIDER"]
        except Exception:
            pass

    # 2. Check Environment Variables (Primary method on AWS ECS Docker container)
    if not provider:
        provider = os.getenv("LLM_PROVIDER")

    # 3. Check Settings fallback
    if not provider and hasattr(settings, "LLM_PROVIDER"):
        provider = settings.LLM_PROVIDER

    if not provider:
        return "Unknown"

    provider_map = {
        "groq": "Groq (Llama 3.3 70B)",
        "openai": "OpenAI (GPT-4o)",
        "gemini": "Google Gemini",
        "ollama": "Llama3 Local",
        "openrouter": "OpenRouter",
    }
    return provider_map.get(provider.lower(), provider.title())

def render_metrics() -> None:
    """Renders the three top metrics card dashboard widgets.
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

    # 2. Determine active provider & dynamic status
    provider_raw = os.getenv("LLM_PROVIDER", getattr(settings, "LLM_PROVIDER", "ollama")).lower()
    provider_label = get_current_llm_provider()

    # Check connection depending on whether provider is local (ollama) or cloud API
    if provider_raw == "ollama":
        from llms.llm_router import check_llama_connection
        is_online = check_llama_connection()
        status_sub = "Systems Operational" if is_online else "Ollama Offline"
        status_label = "Ready" if is_online else "Offline"
    else:
        # Cloud APIs (Groq, OpenAI, Gemini) are assumed online if configured
        is_online = True
        status_sub = "Cloud API Active"
        status_label = "Ready"

    status_color = "rgba(16, 185, 129, 0.15)" if is_online else "rgba(239, 68, 68, 0.15)"
    status_text_color = "#10B981" if is_online else "#EF4444"

    col1, col2, col3 = st.columns(3)

    # Card 1: AI Assistant Status
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
            unsafe_allow_html=True,
        )

    # Card 2: Active Model
    with col2:
        st.markdown(
            f'<div class="saas-metric-card">'
            f'  <div class="saas-metric-icon" style="background: rgba(99, 102, 241, 0.1); color: var(--primary);">🧠</div>'
            f'  <div>'
            f'    <div class="metric-label" style="font-size: 12px; color: var(--text-muted);">Active Model</div>'
            f'    <div class="metric-value" style="font-size: 18px; font-weight: 700; color: var(--text-main);">{provider_label}</div>'
            f'    <div class="metric-sub" style="font-size: 11px; color: var(--text-muted);">{provider_raw.title()} Provider</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
        )