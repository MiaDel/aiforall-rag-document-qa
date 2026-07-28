"""
=========================================================
File Name : settings.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Settings and Diagnostics dashboard view. Displays system parameters,
health status cards (Memory, Disk, DB connections), and database resets.

Technologies:
- Streamlit
- Python standard shutil library
=========================================================
"""

import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any
import streamlit as st
from config.settings import settings
from vectorstore.chroma_manager import ChromaManager
from llms.llm_router import check_llama_connection

logger = logging.getLogger(__name__)


def get_hardware_status() -> Dict[str, str]:
    """Computes virtual memory and disk capacity stats.

    Returns:
        Dict[str, str]:
            Hardware performance percentages.
    """
    total, used, free = shutil.disk_usage("/")
    used_gb: float = used / (1024 ** 3)
    total_gb: float = total / (1024 ** 3)
    disk_str: str = f"{used_gb:.1f}/{total_gb:.1f} GB ({used/total*100:.1f}%)"

    try:
        import psutil
        mem = psutil.virtual_memory()
        used_mem_gb: float = mem.used / (1024 ** 3)
        total_mem_gb: float = mem.total / (1024 ** 3)
        mem_str: str = f"{used_mem_gb:.1f}/{total_mem_gb:.1f} GB ({mem.percent}%)"
    except ImportError:
        mem_str = "N/A"

    return {"disk_str": disk_str, "mem_str": mem_str}


def run_settings_page() -> None:
    """Renders the settings and system health diagnostic view."""
    st.markdown('<h2 style="margin-bottom:8px;">⚙️ Settings & System Health</h2>', unsafe_allow_html=True)
    st.write("Configure model thresholds, monitor system logs, and manage database persistence.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Health", "⚙️ Configurations", "🔍 Ask & Explore", "⚠️ Database Reset"])

    # --- Tab 1: System Health Card ---
    with tab1:
        st.markdown("### Compact System Health Diagnostics")
        
        # Connection check logs
        is_llama_online: bool = check_llama_connection()
        llama_badge: str = "🟢 Online" if is_llama_online else "🔴 Offline"
        llama_color: str = "#10B981" if is_llama_online else "#EF4444"
        
        hw: Dict[str, str] = get_hardware_status()
        
        st.markdown(
            f'<div class="saas-container">'
            f'  <h4 style="margin-top:0px; color:var(--text-main);">🖥️ System Health Metrics</h4>'
            f'  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-top:20px;">'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">ChromaDB</span><br/>'
            f'      <span style="font-weight:600; color:#10B981;">🟢 Connected</span>'
            f'    </div>'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">Embeddings Model</span><br/>'
            f'      <span style="font-weight:600; color:#10B981;">🟢 Active (BGE-Large)</span>'
            f'    </div>'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">Ollama Service</span><br/>'
            f'      <span style="font-weight:600; color:{llama_color};">{llama_badge}</span>'
            f'    </div>'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">LLM Provider</span><br/>'
            f'      <span style="font-weight:600; color:#10B981;">🟢 Llama3 Local</span>'
            f'    </div>'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">Disk Capacity</span><br/>'
            f'      <span style="font-weight:600;">{hw["disk_str"]}</span>'
            f'    </div>'
            f'    <div style="padding:12px; border:1px solid var(--border-color); border-radius:12px;">'
            f'      <span style="font-size:12px; color:var(--text-muted);">Memory Load</span><br/>'
            f'      <span style="font-weight:600;">{hw["mem_str"]}</span>'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # --- Tab 2: Configurations ---
    with tab2:
        st.markdown("### Active Configuration Parameters")
        
        st.markdown("#### Vector Store Parameters")
        st.json({
            "CHROMA_PERSIST_DIRECTORY": settings.CHROMA_PERSIST_DIRECTORY,
            "COLLECTION_NAME": settings.COLLECTION_NAME,
            "CHUNK_SIZE": settings.CHUNK_SIZE,
            "CHUNK_OVERLAP": settings.CHUNK_OVERLAP
        })

        st.markdown("#### Retrieval & Models")
        st.json({
            "TOP_K": settings.TOP_K,
            "SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL_NAME,
            "RERANKER_MODEL": settings.RERANKER_MODEL_NAME
        })

    # --- Tab 3: Ask & Explore ---
    with tab3:
        from ui.pages.ask_explore import run_ask_explore_page
        run_ask_explore_page()

    # --- Tab 4: Database Reset (Danger Zone) ---
    with tab4:
        st.markdown("### Danger Zone - Clear Database")
        st.write("Wipes out all document vectors, page references, and uploads.")
        
        with st.popover("🗑️ Clear Documents & Vectors", use_container_width=True):
            st.error("⚠️ This action will permanently delete all uploaded documents and embeddings.")
            
            if st.button("Confirm Deletion", key="confirm_reset_kb_tab_btn", use_container_width=True, type="primary"):
                try:
                    # 1. Delete all uploaded documents from uploads folder
                    uploads_dir: Path = Path(settings.CHROMA_PERSIST_DIRECTORY).parent / "uploads"
                    if uploads_dir.exists():
                        for f in uploads_dir.iterdir():
                            if f.is_file():
                                f.unlink()
                    
                    # 2. Reset collection in ChromaDB
                    from utils.resource_manager import get_chroma_manager
                    chroma_mgr = get_chroma_manager()
                    chroma_mgr.reset_database()
                    
                    # 3. Clear session_state variables
                    st.session_state["chat_history"] = []
                    st.session_state["indexed_docs_count"] = 0
                    st.session_state["chunk_count"] = 0
                    st.session_state["ingested_pages"] = []
                    
                    st.toast("Database cleared successfully.")
                    st.success("Database cleared successfully.")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset database: {str(e)}")
