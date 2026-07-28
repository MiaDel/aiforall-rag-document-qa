"""
=========================================================
File Name : sidebar.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Left sidebar component for the Loan RAG Chatbot UI.
Contains minimal vertical navigation tabs using custom SaaS styles.

Technologies:
- Streamlit
=========================================================
"""

import logging
from typing import Dict
import streamlit as st

logger = logging.getLogger(__name__)


def render_sidebar() -> None:
    """Renders the minimal SaaS Left Sidebar panel.

    Workflow:
    1. Render application logo and title.
    2. Render minimal vertical buttons for multi-page routing.
    """
    with st.sidebar:
        # App Header Logo & Title
        st.markdown(
            '<div class="sidebar-logo-container">'
            '  <span class="logo-emoji">💼</span>'
            '  <span class="logo-text">Loan RAG</span>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="sidebar-section-title">WORKSPACE</p>', unsafe_allow_html=True)
        
        # Define clean navigation pages
        pages: Dict[str, str] = {
            "Chat Assistant": "🏠 Chat",
            "Upload Documents": "📄 Upload Documents",
            "Document Viewer": "📚 Document Viewer",
            "Settings": "⚙️ Settings",
            "About": "❓ About"
        }

        # Keep active navigation page highlighting
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Chat Assistant"

        # Render menu buttons
        for page_id, label in pages.items():
            is_active: bool = st.session_state["current_page"] == page_id
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(label, key=f"nav_btn_{page_id}", use_container_width=True, type=btn_type):
                st.session_state["current_page"] = page_id
                st.rerun()
                
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
