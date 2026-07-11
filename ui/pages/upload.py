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
            # Standard namespaces definition for OpenXML Paragraph elements
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


def run_upload_page() -> None:
    """Renders and manages the document upload page dashboard.

    Workflow:
    1. Render drag-and-drop file uploader for PDF, DOCX, and TXT files.
    2. Save uploaded items to target data/uploads workspace folder.
    3. Parse text based on document suffix.
    4. Pass texts to cleaner, metadata builder, semantic chunker, and ChromaDB.
    5. Save results and update session status counters.

    Returns:
        None
    """
    st.title("📂 Upload & Index Documents")
    st.write(
        "Upload contracts, disclosures, policies, or manuals (PDF, DOCX, TXT). "
        "The system will extract text, segment it semantically, generate vector embeddings, "
        "and store them in the persistent ChromaDB store."
    )

    # File Uploader UI supporting PDF, DOCX, TXT
    uploaded_files = st.file_uploader(
        "Select Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files (PDF, DOCX, TXT) at once."
    )

    if uploaded_files:
        st.markdown(f"### Selected Documents ({len(uploaded_files)})")
        
        # Display selected documents size stats
        for uploaded_file in uploaded_files:
            file_size_mb: float = uploaded_file.size / (1024 * 1024)
            st.text(f"• {uploaded_file.name} ({file_size_mb:.2f} MB)")

        # Indexing Button trigger
        if st.button("🚀 Index Uploaded Documents", use_container_width=True):
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
                    # Save to local file system
                    status_text.text(f"Saving {uploaded_file.name}...")
                    save_path: Path = uploads_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Compute unique SHA256 document ID
                    doc_id: str = ChromaManager.calculate_file_hash(save_path)
                    raw_docs: List[Document] = []
                    
                    # Parse files based on suffix extension
                    if uploaded_file.name.endswith(".pdf"):
                        status_text.text(f"Extracting text from PDF {uploaded_file.name}...")
                        loader = PDFLoader(str(save_path))
                        raw_docs = loader.load()
                    elif uploaded_file.name.endswith(".txt"):
                        status_text.text(f"Extracting text from TXT {uploaded_file.name}...")
                        with open(save_path, "r", encoding="utf-8", errors="ignore") as txt_in:
                            text_content: str = txt_in.read()
                        raw_docs = [Document(page_content=text_content, metadata={
                            "source": save_path.as_posix(),
                            "file_name": uploaded_file.name,
                            "page": 1,
                            "section": "Introduction"
                        })]
                    elif uploaded_file.name.endswith(".docx"):
                        status_text.text(f"Extracting text from DOCX {uploaded_file.name}...")
                        docx_text: str = parse_docx_text(save_path)
                        raw_docs = [Document(page_content=docx_text, metadata={
                            "source": save_path.as_posix(),
                            "file_name": uploaded_file.name,
                            "page": 1,
                            "section": "Introduction"
                        })]
                    
                    # Clean extracted text
                    status_text.text(f"Cleaning formatting for {uploaded_file.name}...")
                    cleaned_docs: List[Document] = cleaner.clean_documents(raw_docs)
                    
                    # Build page metadata
                    status_text.text(f"Extracting section titles for {uploaded_file.name}...")
                    enriched_docs: List[Document] = builder.enrich_documents(cleaned_docs)
                    
                    # Assign document ID to metadata
                    for doc in enriched_docs:
                        doc.metadata["doc_id"] = doc_id
                    
                    # Segment pages into semantic chunks
                    status_text.text(f"Applying semantic chunking to {uploaded_file.name}...")
                    total_pages: int = len(enriched_docs)
                    chunks: List[Document] = []
                    for page_doc in enriched_docs:
                        page_chunks: List[Document] = chunker.chunk_document(page_doc, total_pages=total_pages)
                        chunks.extend(page_chunks)
                    
                    # Index chunks in ChromaDB
                    status_text.text(f"Embedding and storing {len(chunks)} chunks in ChromaDB...")
                    success: bool = chroma_mgr.add_documents(chunks)
                    
                    if success:
                        all_ingested_docs.extend(enriched_docs)
                        successful_uploads.append(uploaded_file.name)
                        total_chunks_created += len(chunks)
                    
                    # Update progress bar
                    progress_bar.progress(int(((idx + 1) / total_files) * 100))
                    
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
                st.session_state["indexed_docs_count"] = len(successful_uploads)
                st.session_state["chunk_count"] = total_chunks_created
                
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
    else:
        st.info("Please select one or more files (PDF, DOCX, TXT) to begin.")
