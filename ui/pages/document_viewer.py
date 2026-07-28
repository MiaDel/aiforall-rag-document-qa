"""
=========================================================
File Name : document_viewer.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module provides the Document Viewer interface in Streamlit.
It allows users to inspect ingested documents page-by-page,
conduct text searches, highlight matches, and browse structural
sections and vector chunks.

Technologies:
- Streamlit
- HTML & CSS rendering
=========================================================
"""

import re
import streamlit as st
from typing import List, Dict, Any


def run_document_viewer_page() -> None:
    """Renders the document viewer page layout."""
    st.markdown('<h2 style="margin-bottom:8px;">📚 Document Viewer</h2>', unsafe_allow_html=True)
    st.write("Browse, search, and inspect the text contents and vector chunks of all indexed files.")
    
    if "ingested_pages" not in st.session_state or not st.session_state["ingested_pages"]:
        st.warning("No documents have been indexed yet. Please upload files in the 'Upload Documents' tab first.")
        return

    # Group by file name
    grouped: Dict[str, List[Any]] = {}
    for doc in st.session_state["ingested_pages"]:
        file_name: str = doc.metadata.get("file_name", "Unknown")
        if file_name not in grouped:
            grouped[file_name] = []
        grouped[file_name].append(doc)

    # Allow clickable citations to select the file automatically
    selected_file_default = list(grouped.keys())[0] if list(grouped.keys()) else None
    if "viewer_selected_file" in st.session_state and st.session_state["viewer_selected_file"] in grouped:
        selected_file_default = st.session_state["viewer_selected_file"]

    # Render Document Selectbox
    selected_file: str = st.selectbox(
        "Select Document to View", 
        list(grouped.keys()), 
        index=list(grouped.keys()).index(selected_file_default) if selected_file_default in list(grouped.keys()) else 0
    )
    
    # Store selected file in state for persistence
    st.session_state["viewer_selected_file"] = selected_file
    
    if selected_file:
        pages: List[Any] = grouped[selected_file]
        # Sort by page number
        pages.sort(key=lambda x: x.metadata.get("page", 1))
        
        st.markdown(f"### {selected_file} ({len(pages)} Pages)")
        
        # Tabs for Page-by-Page vs Chunk View
        viewer_tab1, viewer_tab2 = st.tabs(["📄 Page-by-Page View", "🥞 Indexed Semantic Chunks"])
        
        with viewer_tab1:
            # Search Bar inside Document
            search_query: str = st.text_input("🔍 Search text inside this document:", key=f"search_inside_{selected_file}")
            
            # Filter and display pages
            matched_pages_count: int = 0
            for doc in pages:
                page_num: int = doc.metadata.get("page", 1)
                section: str = doc.metadata.get("section", "General")
                page_content: str = doc.page_content
                
                # Gating filter for search query
                if search_query:
                    if search_query.lower() not in page_content.lower():
                        continue
                    matched_pages_count += 1
                    
                    # Highlight query term inside text using HTML style regex marks
                    escaped_query: str = re.escape(search_query)
                    highlighted_content: str = re.sub(
                        f"({escaped_query})",
                        r"<mark style='background-color:#FDE047; padding: 2px; border-radius:3px; color:#000000;'>\1</mark>",
                        page_content,
                        flags=re.IGNORECASE
                    )
                    
                    with st.expander(f"📖 Page {page_num} (Section: {section}) - Match Found"):
                        st.markdown(
                            f"<div class='saas-card' style='max-height:350px; overflow-y:auto; font-family:monospace; font-size:12px; white-space:pre-wrap;'>"
                            f"{highlighted_content}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    # Standard view without query highlight
                    with st.expander(f"📖 Page {page_num} (Section: {section})"):
                        st.text_area(
                            "Page Text",
                            value=page_content,
                            height=200,
                            disabled=True,
                            key=f"viewer_page_text_{selected_file}_{page_num}"
                        )
            
            if search_query:
                if matched_pages_count == 0:
                    st.info(f"No pages matched the query string '{search_query}'.")
                else:
                    st.success(f"Found matches in {matched_pages_count} pages.")
                    
        with viewer_tab2:
            st.markdown("#### Database Vector Chunks")
            st.write("Below are the exact chunk partitions generated by the Semantic Splitter:")
            
            try:
                from utils.resource_manager import get_chroma_manager
                chroma_mgr = get_chroma_manager()
                # Query Chroma for all chunks matching the file_name
                results = chroma_mgr.collection.get(where={"file_name": selected_file}, include=["documents", "metadatas"])
                chunks_docs = results.get("documents", [])
                chunks_metas = results.get("metadatas", [])
                
                if not chunks_docs:
                    st.info("No vector chunks found in ChromaDB for this file.")
                else:
                    # Render all chunks as expandable cards
                    for idx, (chunk_text, chunk_meta) in enumerate(zip(chunks_docs, chunks_metas)):
                        chunk_id = chunk_meta.get("chunk_id", f"c{idx}")
                        section_title = chunk_meta.get("section_title", "Unknown")
                        page_number = chunk_meta.get("page_number", 1)
                        
                        with st.expander(f"🥞 Chunk Index {idx + 1} - Page {page_number} ({section_title})"):
                            st.code(f"Chunk ID: {chunk_id}\nSection: {section_title}\nPage: {page_number}", language="yaml")
                            st.write(chunk_text)
            except Exception as e:
                st.error(f"Failed to fetch vector chunks: {str(e)}")
