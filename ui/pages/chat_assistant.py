"""
=========================================================
File Name : chat_assistant.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module manages the Chat Assistant interface page in
Streamlit. It displays clean SaaS top bars, metrics status,
chat history, citations panels, and coordinates the uploader.

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

    Args:
        path (Path):
            Absolute file path to the docx document.

    Returns:
        str:
            Extracted clean document text.
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
        raise ValueError(f"Failed to parse DOCX file '{path.name}'.") from e


def run_chat_assistant_page() -> None:
    """Renders the primary chat assistant dashboard layout."""
    # 1. Page Header/Top Bar (Already loaded globally in app.py)
    render_metrics()

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # 2. Main Dashboard Layout (2 Columns: Chat Assistant & Right Panel)
    col_chat, col_sources = st.columns([3, 1])

    # Left Column: Chat Assistant
    with col_chat:
        st.markdown('<p class="right-panel-title">CONVERSATION</p>', unsafe_allow_html=True)

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

        # Render conversational message log using styled chat message containers
        for idx, message in enumerate(st.session_state["chat_history"]):
            role: str = message["role"]
            content: str = message["content"]
            
            if role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
                    
                    # Highlight inline citations if present in latest answers
                    meta = message.get("metadata")
                    if meta:
                        sources = meta.get("sources", [])
                        if sources:
                            cit_badges = [f"`Page {s.get('page')} ({s.get('file')})`" for s in sources]
                            st.markdown(f"**Citations:** {' | '.join(cit_badges)}")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)

        # Gating session state variables for uploader lockouts
        if "indexing_chat" not in st.session_state:
            st.session_state["indexing_chat"] = False

        # Chat Input Box (Disabled while LLM generates or files are indexing)
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        # Check if the last message was user (thinking state)
        is_processing = False
        if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
            is_processing = True

        prompt = st.chat_input("Ask a question about your documents...", disabled=st.session_state["indexing_chat"] or is_processing)
        
        if prompt:
            st.session_state["chat_history"].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

        # Trigger answer generation when user message is pending
        if is_processing:
            user_prompt: str = st.session_state["chat_history"][-1]["content"]
            
            with st.chat_message("assistant", avatar="🤖"):
                # Render Notion-like pulsing dots indicator while generating
                st.markdown('<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>', unsafe_allow_html=True)
                
                try:
                    graph = get_graph()
                    state: Dict[str, Any] = graph.run(user_prompt)
                    
                    answer: str = state.get("answer", "I could not find the answer in the uploaded documents.")
                    sources: List[Dict[str, Any]] = state.get("sources", [])
                    confidence: float = state.get("confidence", 0.0)
                    provider: str = state.get("provider_used", "Unknown")
                    evidence: List[str] = state.get("evidence", [])
                    
                    # Store generated response
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
                    logger.error(f"Error compiling response: {str(e)}")
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"An error occurred while compiling response: {str(e)}",
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.rerun()

        # Action bar: Attach File Popover
        col_act1, col_act2 = st.columns([3, 7])
        
        with col_act1:
            with st.popover("📎 Attach File", use_container_width=True):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                    key="popover_file_uploader",
                    disabled=st.session_state["indexing_chat"]
                )
                
                if uploaded_files:
                    # Index Button disabled during indexing
                    if st.button("🚀 Index Documents", key="popover_index_btn", use_container_width=True, disabled=st.session_state["indexing_chat"]):
                        st.session_state["indexing_chat"] = True
                        
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
                                # A. Uploading PDF...
                                status_msg.markdown(f"📤 **Uploading {f.name}...**")
                                progress_bar.progress(10)
                                time.sleep(0.3)
                                
                                save_path: Path = uploads_dir / f.name
                                with open(save_path, "wb") as f_out:
                                    f_out.write(f.getbuffer())
                                
                                doc_id: str = ChromaManager.calculate_file_hash(save_path)
                                
                                # Duplicate uploads detection check
                                if chroma_mgr.is_document_indexed(doc_id):
                                    status_msg.warning(f"⚠️ Document '{f.name}' is already indexed. Skipping.")
                                    st.toast(f"⚠️ Document '{f.name}' already indexed.", icon="⚠️")
                                    progress_bar.progress(100)
                                    time.sleep(0.6)
                                    continue
                                
                                # B. Reading Document...
                                status_msg.markdown(f"📖 **Reading {f.name}...**")
                                progress_bar.progress(35)
                                time.sleep(0.3)
                                
                                raw_docs: List[Document] = []
                                if f.name.endswith(".pdf"):
                                    loader = PDFLoader(str(save_path))
                                    raw_docs = loader.load()
                                elif f.name.endswith(".txt"):
                                    with open(save_path, "r", encoding="utf-8", errors="ignore") as txt_in:
                                        text_content = txt_in.read()
                                    raw_docs = [Document(page_content=text_content, metadata={
                                        "source": save_path.as_posix(),
                                        "file_name": f.name,
                                        "page": 1,
                                        "section": "Introduction"
                                    })]
                                elif f.name.endswith(".docx"):
                                    docx_text = parse_docx_bytes(save_path)
                                    raw_docs = [Document(page_content=docx_text, metadata={
                                        "source": save_path.as_posix(),
                                        "file_name": f.name,
                                        "page": 1,
                                        "section": "Introduction"
                                    })]
                                
                                # C. Creating Chunks...
                                status_msg.markdown(f"🥞 **Creating Chunks for {f.name}...**")
                                progress_bar.progress(60)
                                time.sleep(0.4)
                                
                                cleaned_docs: List[Document] = cleaner.clean_documents(raw_docs)
                                enriched_docs: List[Document] = builder.enrich_documents(cleaned_docs)
                                
                                for doc in enriched_docs:
                                    doc.metadata["doc_id"] = doc_id
                                
                                page_count: int = len(enriched_docs)
                                chunks: List[Document] = []
                                for page_doc in enriched_docs:
                                    page_chunks = chunker.chunk_document(page_doc, total_pages=page_count)
                                    chunks.extend(page_chunks)
                                
                                # D. Generating Embeddings...
                                status_msg.markdown(f"🧬 **Generating Embeddings for {f.name}...**")
                                progress_bar.progress(85)
                                time.sleep(0.4)
                                
                                # E. Saving to ChromaDB...
                                status_msg.markdown(f"💾 **Saving {f.name} to ChromaDB...**")
                                chroma_mgr.add_documents(chunks)
                                
                                all_ingested.extend(enriched_docs)
                                successful_uploads.append(f.name)
                                total_chunks += len(chunks)
                                
                                progress_bar.progress(100)
                                status_msg.markdown(f"✅ **{f.name} Indexed Successfully!**")
                                st.toast(f"✅ Indexed {f.name}!", icon="✅")
                                time.sleep(0.5)
                                
                            except Exception as ex:
                                st.error(f"Failed indexing {f.name}: {str(ex)}")
                        
                        progress_bar.empty()
                        status_msg.empty()
                        
                        if successful_uploads:
                            st.toast("✅ Indexed Documents successfully!")
                            if "ingested_pages" not in st.session_state:
                                st.session_state["ingested_pages"] = []
                            st.session_state["ingested_pages"].extend(all_ingested)
                            st.session_state["indexed_docs_count"] += len(successful_uploads)
                            st.session_state["chunk_count"] += total_chunks
                        
                        # Reset indexing flag and refresh page
                        st.session_state["indexing_chat"] = False
                        st.rerun()

    # Right Column: Source Citation Panel
    with col_sources:
        render_source_panel()
