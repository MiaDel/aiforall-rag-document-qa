"""
=========================================================
File Name : app.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Streamlit application main entry point. Coordinates multi-page
view dispatchers, configures global loggers, checks runtime
Python version compatibility, and renders the layout.

Technologies:
- Streamlit
- Python System Environment Libraries
=========================================================
"""

import sys
import logging
import streamlit as st
from pathlib import Path
from config.settings import settings
from ui.components.sidebar import render_sidebar

# Page Specific Views
from ui.pages.chat_assistant import run_chat_assistant_page
from ui.pages.upload import run_upload_page
from ui.pages.document_viewer import run_document_viewer_page
from ui.pages.ask_explore import run_ask_explore_page
from ui.pages.settings import run_settings_page

# Centralized logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# Page Configuration
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Startup validation check: warning if Python > 3.11
if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor > 11):
    st.warning("⚠️ Python 3.11 is recommended for full compatibility.")


def load_stylesheet() -> None:
    """Loads the premium dashboard theme CSS stylesheet and injects it.

    Workflow:
    1. Resolve path to ui/styles/main.css.
    2. Read file contents and wrap inside HTML style tags.
    3. Render in Streamlit using st.markdown.

    Returns:
        None
    """
    css_file: Path = Path(__file__).resolve().parent / "ui" / "styles" / "main.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"Theme stylesheet not found at {css_file}")


# Initialize session state variables
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Chat Assistant"
if "indexed_docs_count" not in st.session_state:
    st.session_state["indexed_docs_count"] = 0
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0
if "ingested_pages" not in st.session_state:
    st.session_state["ingested_pages"] = []


def main() -> None:
    """Main routing function for Streamlit page dispatching.

    Workflow:
    1. Load custom styles.
    2. Render Left Sidebar navigation layout.
    3. Match active navigation key to page callback mapping.

    Returns:
        None
    """
    # Load CSS Styles
    load_stylesheet()

    # Render Left Sidebar Navigation and Status Box
    render_sidebar()

    # Page Dispatcher dictionary map
    pages = {
        "Chat Assistant": run_chat_assistant_page,
        "Upload Documents": run_upload_page,
        "Document Viewer": run_document_viewer_page,
        "Ask & Explore": run_ask_explore_page,
        "Settings": run_settings_page
    }

    selected_page: str = st.session_state.get("current_page", "Chat Assistant")
    
    if selected_page in pages:
        pages[selected_page]()
    else:
        st.error(f"Page '{selected_page}' not found.")


if __name__ == "__main__":
    main()
