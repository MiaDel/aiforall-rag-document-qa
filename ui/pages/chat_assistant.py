"""
=========================================================
File Name : chat_assistant.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module manages the Chat Assistant interface page in
Streamlit. It displays greeting headers, statistics metric cards,
chat log bubbles, citations panels, and coordinates uploader
indexing pipelines directly from the chat prompt.

Technologies:
- Streamlit
- LangGraph
- Python Standard Libraries (zipfile, xml)
=========================================================
"""

import time
import zipfile
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from ui.components.metrics import render_metrics
from ui.components.source_panel import render_source_panel
from utils.resource_manager import get_graph, get_embedding_model, get_chroma_manager
from ingestion.pdf_loader import PDFLoader, Document
from ingestion.cleaner import TextCleaner
from ingestion.metadata_builder import MetadataBuilder
from chunking.semantic_chunker import SemanticChunker
from vectorstore.chroma_manager import ChromaManager
from config.settings import settings

logger = logging.getLogger(__name__)


def parse_docx_bytes(path: Path) -> str:
    """Extracts text from a DOCX file using XML parsing.

    Workflow:
    1. Open DOCX container as a Zip archive.
    2. Read word/document.xml contents.
    3. Traverse the XML tree and capture all text nodes inside paragraphs.
    4. Join strings with double spacing.

    Args:
        path (Path):
            Absolute file path to the docx document.

    Returns:
        str:
            Extracted clean document text.

    Raises:
        ValueError: If file parsing fails or archive is corrupted.
    """
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content: bytes = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces: Dict[str, str] = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs: List[str] = []
            for para in root.findall('.//w:p', namespaces):
                texts: List[str] = [node.text for node in para.findall('.//w:t', namespaces) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Error parsing DOCX file {path.name}: {str(e)}")
        raise ValueError(f"Failed to parse DOCX file '{path.name}'. File may be corrupted.") from e


def run_chat_assistant_page() -> None:
    """Renders the primary chat assistant dashboard layout.

    Workflow:
    1. Render greeting headers.
    2. Render top metrics cards.
    3. Render conversations bubble log.
    4. Detect user input and launch LangGraph.
    5. Run status progress updates (Searching documents..., Retrieving chunks..., Reranking..., Generating answer..., Answer ready.).
    6. Render right Top Sources panel.

    Returns:
        None
    """
    # Header greeting
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<h2 style="margin-bottom:2px; font-weight:700; color:#111827;">📚 Multi-Document RAG Chatbot</h2>'
            '<p style="color:#6B7280; font-size:14px; margin-top:0px;">Ask questions across multiple PDFs with citations and intelligent retrieval.</p>',
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(
            '<div style="text-align:right; margin-top: 10px;">'
            '  <span class="status-badge">All Systems Ready</span>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Render Top Metric Cards
    render_metrics()

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # Main Dashboard Layout (2 Columns: Chat Assistant & Right Citation Panel)
    col_chat, col_sources = st.columns([3, 1])

    # Left Column: Chat Assistant
    with col_chat:
        st.markdown(
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
            '  <span style="font-weight:700; font-size:16px; color:#111827;">Chat with your documents</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Initialize chat history if not present
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Default Assistant Greeting if chat is empty
        if not st.session_state["chat_history"]:
            greeting_text: str = (
                "Hi! 👋\n\n"
                "I'm your Multi-Document RAG Assistant. Ask me anything about your "
                "loan documents, policies, eligibility, or procedures.\n\n"
                "How can I help you today?"
            )
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": greeting_text,
                "timestamp": datetime.now().strftime("%H:%M")
            })

        # Render conversational message log
        for message in st.session_state["chat_history"]:
            role: str = message["role"]
            content: str = message["content"]
            timestamp: str = message.get("timestamp", datetime.now().strftime("%H:%M"))
            
            if role == "assistant":
                st.markdown(
                    f'<div style="display:flex; align-items:start; margin-bottom:12px;">'
                    f'  <div style="background-color:#4F46E5; width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-size:14px; margin-right:12px; font-weight:700;">🤖</div>'
                    f'  <div class="chat-bubble-assistant">'
                    f'    <div>{content.replace(chr(10), "<br/>")}</div>'
                    f'    <div class="timestamp">{timestamp}</div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="display:flex; align-items:start; margin-bottom:12px; justify-content:end;">'
                    f'  <div class="chat-bubble-user">'
                    f'    <div>{content.replace(chr(10), "<br/>")}</div>'
                    f'    <div class="timestamp" style="text-align:right;">{timestamp} ✔✔</div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Chat Input Box
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        if prompt := st.chat_input("Type your question here..."):
            st.session_state["chat_history"].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

        # Check if the last message was from the user (trigger generation)
        if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
            user_prompt: str = st.session_state["chat_history"][-1]["content"]
            
            # Render status steps dynamically instead of writing terminal logs
            with st.status("Processing query...", expanded=True) as status:
                st.write("Searching documents...")
                time.sleep(0.4)
                st.write("Retrieving chunks...")
                time.sleep(0.4)
                st.write("Reranking...")
                time.sleep(0.4)
                st.write("Generating answer...")
                
                try:
                    graph = get_graph()
                    state: Dict[str, Any] = graph.run(user_prompt)
                    
                    answer: str = state.get("answer", "I could not find the answer in the uploaded documents.")
                    sources: List[Dict[str, Any]] = state.get("sources", [])
                    confidence: float = state.get("confidence", 0.0)
                    provider: str = state.get("provider_used", "Unknown")
                    evidence: List[str] = state.get("evidence", [])
                    
                    st.write("Answer ready.")
                    status.update(label="Answer ready.", state="complete", expanded=False)
                    
                    # Store response in session state
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "metadata": {
                            "sources": sources,
                            "confidence": confidence,
                            "provider": provider,
                            "evidence": evidence
                        }
                    })
                    st.rerun()
                except Exception as e:
                    status.update(label="Failed to generate answer.", state="error", expanded=True)
                    st.error(f"Error compiling response: {str(e)}")
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"An error occurred while compiling response: {str(e)}",
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.rerun()

        # Action bar under input: Attach File Popover & Filters
        col_act1, col_act2, col_act3 = st.columns([2, 1, 2])
        
        with col_act1:
            with st.popover("📎 Attach File", use_container_width=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                    key="popover_file_uploader"
                )
                
                if uploaded_files:
                    if st.button("🚀 Index Documents", key="popover_index_btn", use_container_width=True):
                        uploads_dir: Path = Path(settings.CHROMA_PERSIST_DIRECTORY).parent / "uploads"
                        uploads_dir.mkdir(parents=True, exist_ok=True)
                        
                        cleaner = TextCleaner()
                        builder = MetadataBuilder()
                        embedder = get_embedding_model()
                        chroma_mgr = get_chroma_manager()
                        chunker = SemanticChunker(
                            embedding_function=embedder.embed_documents,
                            max_chunk_size=settings.CHUNK_SIZE,
                            chunk_overlap=settings.CHUNK_OVERLAP
                        )
                        
                        total_files: int = len(uploaded_files)
                        progress_bar = st.progress(0)
                        status_msg = st.empty()
                        
                        successful_uploads: List[str] = []
                        all_ingested: List[Document] = []
                        total_chunks: int = 0
                        
                        for idx, f in enumerate(uploaded_files):
                            try:
                                status_msg.text(f"Saving {f.name}...")
                                save_path: Path = uploads_dir / f.name
                                with open(save_path, "wb") as f_out:
                                    f_out.write(f.getbuffer())
                                
                                doc_id: str = ChromaManager.calculate_file_hash(save_path)
                                raw_docs: List[Document] = []
                                
                                if f.name.endswith(".pdf"):
                                    status_msg.text(f"Extracting {f.name}...")
                                    loader = PDFLoader(str(save_path))
                                    raw_docs = loader.load()
                                elif f.name.endswith(".txt"):
                                    status_msg.text(f"Extracting {f.name}...")
                                    with open(save_path, "r", encoding="utf-8", errors="ignore") as txt_in:
                                        text_content = txt_in.read()
                                    raw_docs = [Document(page_content=text_content, metadata={
                                        "source": save_path.as_posix(),
                                        "file_name": f.name,
                                        "page": 1,
                                        "section": "Introduction"
                                    })]
                                elif f.name.endswith(".docx"):
                                    status_msg.text(f"Extracting {f.name}...")
                                    docx_text: str = parse_docx_bytes(save_path)
                                    raw_docs = [Document(page_content=docx_text, metadata={
                                        "source": save_path.as_posix(),
                                        "file_name": f.name,
                                        "page": 1,
                                        "section": "Introduction"
                                    })]
                                
                                status_msg.text(f"Cleaning {f.name}...")
                                cleaned_docs: List[Document] = cleaner.clean_documents(raw_docs)
                                enriched_docs: List[Document] = builder.enrich_documents(cleaned_docs)
                                
                                for doc in enriched_docs:
                                    doc.metadata["doc_id"] = doc_id
                                
                                status_msg.text(f"Chunking {f.name}...")
                                page_count: int = len(enriched_docs)
                                chunks: List[Document] = []
                                for page_doc in enriched_docs:
                                    page_chunks = chunker.chunk_document(page_doc, total_pages=page_count)
                                    chunks.extend(page_chunks)
                                
                                status_msg.text(f"Indexing {f.name}...")
                                chroma_mgr.add_documents(chunks)
                                
                                all_ingested.extend(enriched_docs)
                                successful_uploads.append(f.name)
                                total_chunks += len(chunks)
                                
                                progress_bar.progress(int(((idx + 1) / total_files) * 100))
                            except Exception as ex:
                                st.error(f"Failed indexing {f.name}: {str(ex)}")
                        
                        progress_bar.empty()
                        status_msg.empty()
                        
                        if successful_uploads:
                            st.success(f"Indexed {len(successful_uploads)} documents successfully!")
                            if "ingested_pages" not in st.session_state:
                                st.session_state["ingested_pages"] = []
                            st.session_state["ingested_pages"].extend(all_ingested)
                            st.session_state["indexed_docs_count"] += len(successful_uploads)
                            st.session_state["chunk_count"] += total_chunks
                            time.sleep(1.0)
                            st.rerun()

        with col_act2:
            st.button("⚙️ Filters", key="filters_btn", use_container_width=True)

    # Right Column: Source Citation Panel
    with col_sources:
        render_source_panel()
