"""
=========================================================
File Name : source_panel.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Right-hand panel component for visualising Recent Documents,
Recent Citations, and Top Referenced Files with page indicators
and expandable chunk previews.

Technologies:
- Streamlit
=========================================================
"""

import logging
from typing import List, Dict, Any, Set
import streamlit as st
from config.settings import settings

logger = logging.getLogger(__name__)


def render_source_panel() -> None:
    """Renders the redesigned right panel.

    Workflow:
    1. Retrieve unique document list from database/cache.
    2. Extract citations from the last assistant message.
    3. Render Recent Documents, Recent Citations, and Top Referenced Files sections.
    """
    # 1. Fetch unique documents from database/cache for lists
    try:
        from utils.resource_manager import get_chroma_manager
        chroma_mgr = get_chroma_manager()
        results = chroma_mgr.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        unique_files: List[str] = sorted(list(set([
            meta.get("file_name") for meta in metadatas if meta and meta.get("file_name")
        ])))
    except Exception:
        unique_files = []

    # Helper function for clickable documents navigation
    def navigate_to_doc(file_name: str):
        st.session_state["current_page"] = "Document Viewer"
        st.session_state["viewer_selected_file"] = file_name
        st.rerun()

    # --- Section A: Recent Citations ---
    st.markdown('<p class="right-panel-title">RECENT CITATIONS</p>', unsafe_allow_html=True)
    
    history = st.session_state.get("chat_history", [])
    assistant_msgs = [m for m in history if m["role"] == "assistant" and m.get("metadata")]
    
    if not assistant_msgs:
        st.markdown(
            '<div class="empty-state-card">'
            '  📁 No citations yet. Ask a question to see references.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        last_meta = assistant_msgs[-1]["metadata"]
        sources = last_meta.get("sources", [])
        confidence = last_meta.get("confidence", 0.7)
        evidence = last_meta.get("evidence", [])

        if not sources:
            st.markdown(
                '<div class="empty-state-card">'
                '  📄 Answer generated from general context.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            # Render references with navigation linkages
            for idx, source in enumerate(sources):
                file_name = source.get("file", "Unknown Document")
                page_num = source.get("page", 1)
                score_percentage = int(max(40, confidence * 100 - (idx * 6)))
                
                chunk_text = "No snippet content matched."
                if idx < len(evidence):
                    chunk_text = evidence[idx].strip()
                
                # Expandable citation card
                with st.expander(f"📖 {file_name} (Page {page_num})"):
                    st.markdown(f"**Relevance:** `{score_percentage}%`")
                    # Document selection link
                    if st.button("🔍 View full page", key=f"view_page_btn_{idx}_{file_name}", use_container_width=True):
                        navigate_to_doc(file_name)
                    st.markdown(
                        f'<div class="evidence-preview-block">'
                        f'  {chunk_text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Section B: Recent Documents ---
    st.markdown('<p class="right-panel-title">RECENT DOCUMENTS</p>', unsafe_allow_html=True)
    if not unique_files:
        st.markdown(
            '<div class="empty-state-card">'
            '  No documents uploaded yet.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Show last 5 uploaded files
        for idx, file_name in enumerate(unique_files[-5:]):
            col_icon, col_link = st.columns([1, 9])
            with col_icon:
                st.markdown("📄")
            with col_link:
                if st.button(file_name, key=f"recent_doc_link_{idx}", use_container_width=True):
                    navigate_to_doc(file_name)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Section C: Top Referenced Files ---
    st.markdown('<p class="right-panel-title">TOP REFERENCED FILES</p>', unsafe_allow_html=True)
    
    # Calculate referenced counts across session chat history
    referenced_counts: Dict[str, int] = {}
    for m in history:
        if m["role"] == "assistant" and m.get("metadata"):
            sources_list = m["metadata"].get("sources", [])
            for src in sources_list:
                f_name = src.get("file")
                if f_name:
                    referenced_counts[f_name] = referenced_counts.get(f_name, 0) + 1

    if not referenced_counts:
        st.markdown(
            '<div class="empty-state-card">'
            '  No referenced documents yet.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        sorted_referenced = sorted(referenced_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (file_name, count) in enumerate(sorted_referenced[:3]):
            col_icon, col_link = st.columns([1, 9])
            with col_icon:
                st.markdown("🔥")
            with col_link:
                btn_label = f"{file_name} ({count} citations)"
                if st.button(btn_label, key=f"top_ref_link_{idx}", use_container_width=True):
                    navigate_to_doc(file_name)
