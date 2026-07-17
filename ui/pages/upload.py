"""
=========================================================
File Name : upload.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module provides the Streamlit user interface and backend
integration for document uploads. It supports PDF, DOCX, and
TXT files, performs text extraction, cleaning, metadata
enrichment, and indexes them in ChromaDB.

Technologies:
- Streamlit
- PyMuPDF (fitz)
- Python Standard Libraries (zipfile, xml)
=========================================================
"""

import os
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from config.settings import settings
from ingestion.pdf_loader import PDFLoader, Document
from ingestion.cleaner import TextCleaner
from ingestion.metadata_builder import MetadataBuilder
from chunking.semantic_chunker import SemanticChunker
from vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


def parse_docx_text(path: Path) -> str:
    """Extracts plain text from a DOCX file using XML parsing.

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


def run_upload_page() -> None:
    """Renders and manages the document upload page dashboard."""
    st.markdown('<h2 style="margin-bottom:8px;">📂 Upload & Index Documents</h2>', unsafe_allow_html=True)
    st.write(
        "Upload contracts, disclosures, policies, or manuals (PDF, DOCX, TXT). "
        "The system will extract text, segment it semantically, generate vector embeddings, "
        "and store them in the persistent ChromaDB store."
    )

    # State variables to prevent parallel indexing clicks
    if "indexing_upload" not in st.session_state:
        st.session_state["indexing_upload"] = False

    # File Uploader UI supporting PDF, DOCX, TXT
    uploaded_files = st.file_uploader(
        "Select Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files (PDF, DOCX, TXT) at once.",
        disabled=st.session_state["indexing_upload"]
    )

    if uploaded_files:
        st.markdown(f"### Selected Documents ({len(uploaded_files)})")
        
        # Display selected documents size stats
        for uploaded_file in uploaded_files:
            file_size_mb: float = uploaded_file.size / (1024 * 1024)
            st.text(f"• {uploaded_file.name} ({file_size_mb:.2f} MB)")

        # Indexing Button trigger (disabled during active indexing run)
        if st.button("🚀 Index Uploaded Documents", use_container_width=True, disabled=st.session_state["indexing_upload"]):
            st.session_state["indexing_upload"] = True
            st.rerun()

    # Trigger indexing sequence if active
    if st.session_state["indexing_upload"] and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Setup uploads directory folder
        uploads_dir: Path = Path(settings.CHROMA_PERSIST_DIRECTORY).parent / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        total_files: int = len(uploaded_files)
        all_ingested_docs: List[Document] = []
        successful_uploads: List[str] = []
        total_chunks_created: int = 0
        
        # Initialize core RAG components
        cleaner = TextCleaner()
        builder = MetadataBuilder()
        
        # Retrieve cached model engines from Resource Manager
        from utils.resource_manager import get_embedding_model, get_chroma_manager
        embedder = get_embedding_model()
        chroma_mgr = get_chroma_manager()
        
        chunker = SemanticChunker(
            embedding_function=embedder.embed_documents,
            max_chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # 1. Uploading PDF...
                status_text.markdown(f"📤 **Uploading {uploaded_file.name}...**")
                progress_bar.progress(10)
                import time; time.sleep(0.3)
                
                save_path: Path = uploads_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Compute unique SHA256 document ID
                doc_id: str = ChromaManager.calculate_file_hash(save_path)
                
                # Duplicate upload check
                if chroma_mgr.is_document_indexed(doc_id):
                    status_text.warning(f"⚠️ Document '{uploaded_file.name}' is already indexed. Skipping.")
                    st.toast(f"⚠️ Document '{uploaded_file.name}' already exists in database.", icon="⚠️")
                    progress_bar.progress(100)
                    time.sleep(0.6)
                    continue
                
                # 2. Reading Document...
                status_text.markdown(f"📖 **Reading {uploaded_file.name}...**")
                progress_bar.progress(35)
                time.sleep(0.3)
                
                raw_docs: List[Document] = []
                # Parse files based on suffix extension
                if uploaded_file.name.endswith(".pdf"):
                    loader = PDFLoader(str(save_path))
                    raw_docs = loader.load()
                elif uploaded_file.name.endswith(".txt"):
                    with open(save_path, "r", encoding="utf-8", errors="ignore") as txt_in:
                        text_content: str = txt_in.read()
                    raw_docs = [Document(page_content=text_content, metadata={
                        "source": save_path.as_posix(),
                        "file_name": uploaded_file.name,
                        "page": 1,
                        "section": "Introduction"
                    })]
                elif uploaded_file.name.endswith(".docx"):
                    docx_text: str = parse_docx_text(save_path)
                    raw_docs = [Document(page_content=docx_text, metadata={
                        "source": save_path.as_posix(),
                        "file_name": uploaded_file.name,
                        "page": 1,
                        "section": "Introduction"
                    })]
                
                # 3. Creating Chunks...
                status_text.markdown(f"🥞 **Creating Chunks for {uploaded_file.name}...**")
                progress_bar.progress(60)
                time.sleep(0.4)
                
                cleaned_docs: List[Document] = cleaner.clean_documents(raw_docs)
                enriched_docs: List[Document] = builder.enrich_documents(cleaned_docs)
                
                for doc in enriched_docs:
                    doc.metadata["doc_id"] = doc_id
                
                # Segment pages into semantic chunks
                total_pages: int = len(enriched_docs)
                chunks: List[Document] = []
                for page_doc in enriched_docs:
                    page_chunks: List[Document] = chunker.chunk_document(page_doc, total_pages=total_pages)
                    chunks.extend(page_chunks)
                
                # 4. Generating Embeddings...
                status_text.markdown(f"🧬 **Generating Embeddings for {uploaded_file.name}...**")
                progress_bar.progress(85)
                time.sleep(0.4)
                
                # 5. Saving to ChromaDB...
                status_text.markdown(f"💾 **Saving {uploaded_file.name} to ChromaDB...**")
                success: bool = chroma_mgr.add_documents(chunks)
                
                if success:
                    all_ingested_docs.extend(enriched_docs)
                    successful_uploads.append(uploaded_file.name)
                    total_chunks_created += len(chunks)
                
                # Update progress bar
                progress_bar.progress(100)
                status_text.markdown(f"✅ **{uploaded_file.name} Indexed Successfully!**")
                st.toast(f"✅ Indexed {uploaded_file.name}!", icon="✅")
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed indexing file {uploaded_file.name}: {str(e)}")
                st.error(f"Error indexing {uploaded_file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if successful_uploads:
            st.success(
                f"Successfully ingested {len(successful_uploads)} documents! "
                f"Created {total_chunks_created} semantic chunks across {len(all_ingested_docs)} pages."
            )
            
            # Cache pages in session state for viewer purposes
            if "ingested_pages" not in st.session_state:
                st.session_state["ingested_pages"] = []
            
            st.session_state["ingested_pages"].extend(all_ingested_docs)
            st.session_state["indexed_docs_count"] += len(successful_uploads)
            st.session_state["chunk_count"] += total_chunks_created
            
            # Render metadata preview
            st.markdown("### 🔍 Sample Metadata Preview")
            preview_count: int = min(3, len(all_ingested_docs))
            for i in range(preview_count):
                doc: Document = all_ingested_docs[i]
                with st.expander(f"{doc.metadata['file_name']} - Page {doc.metadata['page']}"):
                    st.json(doc.metadata)
                    st.text_area("Cleaned Page Text (Preview)", value=doc.page_content[:400], height=120, key=f"preview_text_{i}")
        else:
            st.error("No documents were successfully processed.")
            
        # Reset indexing state variables
        st.session_state["indexing_upload"] = False
        st.rerun()
    else:
        if not uploaded_files:
            st.info("Please select one or more files (PDF, DOCX, TXT) to begin.")
