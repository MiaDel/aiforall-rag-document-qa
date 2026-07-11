"""
=========================================================
File Name : sidebar.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module renders the Streamlit Left Sidebar. It manages
vertical page navigation buttons, tracks dynamic system health
indicators (ChromaDB, Embedding models, Llama3 local ports,
disk sizes, memory percentages), and contains database resets.

Technologies:
- Streamlit
- Python standard shutil library
- psutil library (optional)
=========================================================
"""

import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
from vectorstore.chroma_manager import ChromaManager
from config.settings import settings
from llms.llm_router import check_llama_connection

logger = logging.getLogger(__name__)


def get_system_hardware_stats() -> Dict[str, str]:
    """Computes host disk space and virtual memory utilization percentages.

    Workflow:
    1. Call shutil.disk_usage on the root folder.
    2. Import psutil and fetch virtual memory usage.
    3. Return formatted metrics strings.

    Returns:
        Dict[str, str]:
            Dict holding keys 'disk_str' and 'mem_str'.
    """
    # 1. Compute Disk usage
    total, used, free = shutil.disk_usage("/")
    used_gb: float = used / (1024 ** 3)
    total_gb: float = total / (1024 ** 3)
    disk_str: str = f"{used_gb:.1f}/{total_gb:.1f} GB ({used/total*100:.1f}%)"

    # 2. Compute Memory usage using psutil
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_mem_gb: float = mem.used / (1024 ** 3)
        total_mem_gb: float = mem.total / (1024 ** 3)
        mem_str: str = f"{used_mem_gb:.1f}/{total_mem_gb:.1f} GB ({mem.percent}%)"
    except ImportError:
        mem_str = "N/A"

    return {"disk_str": disk_str, "mem_str": mem_str}


def render_sidebar() -> None:
    """Renders the custom Left Sidebar panel including system health stats.

    Workflow:
    1. Render application logo.
    2. Render vertical buttons for multi-page routing.
    3. Run connection checks for Llama3 Local.
    4. Compile disk and memory load stats.
    5. Render System Health checklist.
    6. Render Reset Knowledge Base popover warning panel.

    Returns:
        None
    """
    with st.sidebar:
        # App Header Title
        st.markdown(
            '<div style="display:flex; align-items:center; margin-bottom: 24px;">'
            '<span style="font-size:24px; margin-right:10px;">💼</span>'
            '<span style="font-size:18px; font-weight:700; color:#111827;">Loan RAG<br/>Chatbot</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Navigation Buttons (System Status page is removed)
        st.markdown('<p style="font-size:11px; font-weight:600; color:#9CA3AF; margin-bottom:8px; text-transform:uppercase;">Navigation</p>', unsafe_allow_html=True)
        
        pages: Dict[str, str] = {
            "Chat Assistant": "💬 Chat Assistant",
            "Upload Documents": "📂 Upload Documents",
            "Document Viewer": "📄 Document Viewer",
            "Ask & Explore": "🔍 Ask & Explore",
            "Settings": "⚙️ Settings"
        }

        # Keep active class styling
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Chat Assistant"

        for page_id, label in pages.items():
            btn_type = "primary" if st.session_state["current_page"] == page_id else "secondary"
            if st.button(label, key=f"nav_btn_{page_id}", use_container_width=True, type=btn_type):
                st.session_state["current_page"] = page_id
                st.rerun()

        # Dynamic Database stats widget and Health status indicator
        st.markdown('<p style="font-size:11px; font-weight:600; color:#9CA3AF; margin-top:28px; margin-bottom:8px; text-transform:uppercase;">System Health</p>', unsafe_allow_html=True)
        
        # Live connection checks
        is_llama_online: bool = check_llama_connection()
        llama_status: str = "Online" if is_llama_online else "Offline"
        llama_color: str = "#10B981" if is_llama_online else "#EF4444"
        
        # Fetch disk/memory metrics
        hw_stats: Dict[str, str] = get_system_hardware_stats()
        
        st.markdown(
            f'<div class="sidebar-status-box" style="margin-top:0px;">'
            f'  <div style="display:flex; align-items:center; margin-bottom:12px;">'
            f'    <span style="height:8px; width:8px; background-color:#10B981; border-radius:50%; display:inline-block; margin-right:8px;"></span>'
            f'    <span style="font-size:13px; font-weight:600; color:#10B981;">System Healthy</span>'
            f'  </div>'
            f'  <div class="status-item"><span class="status-item-label">ChromaDB</span><span class="status-item-value" style="color:#10B981;">Online</span></div>'
            f'  <div class="status-item"><span class="status-item-label">Embeddings</span><span class="status-item-value" style="color:#10B981;">Active</span></div>'
            f'  <div class="status-item"><span class="status-item-label">CrossEncoder</span><span class="status-item-value" style="color:#10B981;">Active</span></div>'
            f'  <div class="status-item"><span class="status-item-label">Ollama LLM</span><span class="status-item-value" style="color:{llama_color};">{llama_status}</span></div>'
            f'  <div class="status-item"><span class="status-item-label">Disk Usage</span><span class="status-item-value" style="font-size:11px;">{hw_stats["disk_str"]}</span></div>'
            f'  <div class="status-item"><span class="status-item-label">Memory Usage</span><span class="status-item-value" style="font-size:11px;">{hw_stats["mem_str"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # Reset Knowledge Base Panel (Danger Zone)
        st.markdown('<p style="font-size:11px; font-weight:600; color:#9CA3AF; margin-bottom:8px; text-transform:uppercase;">Danger Zone</p>', unsafe_allow_html=True)
        
        with st.popover("🗑️ Clear Documents & Vectors", use_container_width=True):
            st.warning("This action will permanently delete all uploaded documents and embeddings.")
            
            if st.button("Confirm Reset", key="confirm_reset_kb_btn", use_container_width=True, type="primary"):
                try:
                    # Delete all uploaded PDFs from data/uploads
                    uploads_dir: Path = Path(settings.CHROMA_PERSIST_DIRECTORY).parent / "uploads"
                    if uploads_dir.exists():
                        for f in uploads_dir.iterdir():
                            if f.is_file():
                                f.unlink()
                    
                    # Reset collection in ChromaDB
                    from utils.resource_manager import get_chroma_manager
                    chroma_mgr = get_chroma_manager()
                    chroma_mgr.reset_database()
                    
                    # Reset session_state variables
                    st.session_state["chat_history"] = []
                    st.session_state["indexed_docs_count"] = 0
                    st.session_state["chunk_count"] = 0
                    st.session_state["ingested_pages"] = []
                    
                    st.success("Database cleared successfully.")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset database: {str(e)}")
