"""
=========================================================
File Name : app.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman, Mia Delhalle
Description:
Streamlit application main entry point. Coordinates multi-page
view dispatchers, checks runtime Python version, renders
theme switchers, top bars, and left navigation sidebars.

Technologies:
- Streamlit
- Python System Environment Libraries
=========================================================
"""

import sys
import os
import logging
import streamlit as st
from pathlib import Path
from config.settings import settings
from ui.components.sidebar import render_sidebar

# Page Specific Views
from ui.pages.chat_assistant import run_chat_assistant_page
from ui.pages.upload import run_upload_page
from ui.pages.document_viewer import run_document_viewer_page
from ui.pages.settings import run_settings_page
from utils.resource_manager import get_chroma_manager

# Page Configuration
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Triggers auto-build on first run; cached after that
chroma_manager = get_chroma_manager()

# Centralized logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")


# Startup validation check: warning if Python > 3.11
# if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor > 11):
#     st.warning("⚠️ Python 3.11 is recommended for full compatibility.")


def load_stylesheet() -> None:
    """Loads the premium dashboard theme CSS stylesheet and injects it.
    Also handles parent frame local storage synchronization for Light/Dark mode.
    """
    theme = st.session_state.get("theme", "dark")
    
    css_file: Path = Path(__file__).resolve().parent / "ui" / "styles" / "main.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            css_content = f.read()
        
        # 1. Inject CSS only using st.markdown
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        
        # 2. Inject JS silently using components.html(..., height=0)
        import streamlit.components.v1 as components
        js_code = f"""
        <script>
        try {{
            const theme = "{theme}";
            localStorage.setItem("theme", theme);
            
            // Set theme attributes on current document
            document.body.setAttribute("data-theme", theme);
            document.body.classList.add(theme === "dark" ? "dark-theme" : "light-theme");
            document.body.classList.remove(theme === "dark" ? "light-theme" : "dark-theme");
            
            // Set theme attributes on parent iframe (Streamlit app main container)
            if (window.parent && window.parent.document) {{
                const parentBody = window.parent.document.body;
                if (parentBody) {{
                    parentBody.setAttribute("data-theme", theme);
                    parentBody.classList.add(theme === "dark" ? "dark-theme" : "light-theme");
                    parentBody.classList.remove(theme === "dark" ? "light-theme" : "dark-theme");
                }}
            }}
            
            // Set theme attributes on grandparent (outermost window)
            if (window.parent.parent && window.parent.parent.document) {{
                const gpBody = window.parent.parent.document.body;
                if (gpBody) {{
                    gpBody.setAttribute("data-theme", theme);
                    gpBody.classList.add(theme === "dark" ? "dark-theme" : "light-theme");
                    gpBody.classList.remove(theme === "dark" ? "light-theme" : "dark-theme");
                }}
            }}
        }} catch(e) {{
            console.error("Theme toggle error:", e);
        }}
        </script>
        """
        components.html(js_code, height=0)
    else:
        logger.warning(f"Theme stylesheet not found at {css_file}")



# def get_current_llm_provider() -> str:
#     """Safely retrieves the LLM provider name from st.secrets, env vars, or settings."""
#     provider = None
    
#     # 1. Try reading from Streamlit Secrets (st.secrets)
#     try:
#         if "LLM_PROVIDER" in st.secrets:
#             provider = st.secrets["LLM_PROVIDER"]
#     except Exception:
#         pass
        
#     # 2. Try reading from environment variables (.env / os.environ)
#     if not provider:
#         provider = os.getenv("LLM_PROVIDER")
        
#     # 3. Try reading from your settings module fallback
#     if not provider and hasattr(settings, "LLM_PROVIDER"):
#         provider = settings.LLM_PROVIDER

#     if not provider:
#         return "Unknown Provider"

#     # Map raw key strings to user-friendly UI labels
#     provider_labels = {
#         "groq": "Groq (Llama 3.3 70B)",
#         "openai": "OpenAI (GPT-4o)",
#         "gemini": "Google Gemini",
#         "ollama": "Llama3 Local",
#         "openrouter": "OpenRouter",
#     }
    
#     return provider_labels.get(provider.lower(), provider.title())

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


def render_top_bar() -> None:
    """Renders the professional SaaS Top Bar."""
    col_logo, col_space, col_model, col_theme = st.columns([4, 1, 3.5, 1.5])
    
    current_provider_label = get_current_llm_provider()
    
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
        
    theme_icon = "☀️ Light" if st.session_state["theme"] == "dark" else "🌙 Dark"
    
    with col_logo:
        st.markdown(
            '<div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">'
            '  <span style="font-size: 24px;">📚</span>'
            '  <span style="font-size: 20px; font-weight: 700; font-family:\'Outfit\', sans-serif;">Loan RAG Chatbot</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
    with col_model:
        st.markdown(
            f'<div style="text-align: right; margin-top: 12px;">'
            f'  <span class="model-badge">🟢 {current_provider_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    with col_theme:
        st.markdown('<div style="margin-top: 4px;"></div>', unsafe_allow_html=True)
        if st.button(theme_icon, key="theme_toggle_btn", use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
            
    

def run_about_page() -> None:
    """Renders the About multi-document chatbot info page."""
    current_provider_label = get_current_llm_provider()
    
    st.markdown('<h2 style="margin-bottom:8px;">❓ About Loan RAG Chatbot</h2>', unsafe_allow_html=True)
    st.write("Browse architecture descriptions and technical details.")

    st.markdown(
        '<div class="saas-container">'
        '  <h3 style="margin-top:0px; color:var(--text-main);">💡 Intelligent Multi-Document Document Q&A</h3>'
        '  <p style="color:var(--text-muted);">The Loan RAG Chatbot is a state-of-the-art SaaS Q&A dashboard designed to help risk analysts and legal professionals query complex loan contracts, disclosures, and policies.</p>'
        '  <hr style="border-color: var(--border-color); margin: 20px 0;">'
        '  <h4 style="color:var(--text-main);">🔍 Pipeline Architecture</h4>'
        '  <ul style="color:var(--text-muted); line-height: 1.6;">'
        '    <li><b>PyMuPDF Parsing:</b> High-speed text extraction from contracts.</li>'
        '    <li><b>Semantic Chunking:</b> Segments documents at semantic shifts using embedding distance shifts.</li>'
        '    <li><b>Hybrid Search:</b> Connects dense vector search (ChromaDB) with lexical keyword matching (BM25).</li>'
        '    <li><b>Cross-Encoder:</b> MS-Marco MiniLM model reranks retrieved passages.</li>'
        f'    <li><b>{current_provider_label}:</b> Synthesizes context to generate accurate, citation-anchored answers.</li>'
        '  </ul>'
        '</div>',
        unsafe_allow_html=True
    )

# Initialize session state variables
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Chat Assistant"
if "indexed_docs_count" not in st.session_state:
    st.session_state["indexed_docs_count"] = 0
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0
if "ingested_pages" not in st.session_state:
    st.session_state["ingested_pages"] = []


def main() -> None:
    """Main routing function for Streamlit page dispatching."""
    # Load CSS Styles (including theme injector scripts)
    load_stylesheet()

    # Render SaaS Top Bar
    render_top_bar()
    st.markdown('<div class="sidebar-divider" style="margin-top:0px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    # Render Left Sidebar Navigation and Status Box
    render_sidebar()

    # Page Dispatcher dictionary map
    pages = {
        "Chat Assistant": run_chat_assistant_page,
        "Upload Documents": run_upload_page,
        "Document Viewer": run_document_viewer_page,
        "Settings": run_settings_page,
        "About": run_about_page
    }

    selected_page: str = st.session_state.get("current_page", "Chat Assistant")
    
    if selected_page in pages:
        pages[selected_page]()
    else:
        st.error(f"Page '{selected_page}' not found.")


if __name__ == "__main__":
    main()
