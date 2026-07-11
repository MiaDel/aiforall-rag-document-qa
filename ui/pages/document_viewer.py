"""
=========================================================
File Name : document_viewer.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
This module provides the Document Viewer interface in Streamlit.
It allows users to inspect ingested documents page-by-page,
conduct text searches, highlight matches, and browse structural
sections.

Technologies:
- Streamlit
- HTML & CSS rendering
=========================================================
"""

import re
import streamlit as st
from typing import List, Dict, Any


def run_document_viewer_page() -> None:
    """Renders the document viewer page layout.

    Workflow:
    1. Check if documents exist in st.session_state.
    2. Group parsed pages by document source filename.
    3. Render document selectbox.
    4. Render text search bar ("Search inside document").
    5. Highlight matched text strings in expandable page previews.

    Returns:
        None
    """
    st.title("📄 Document Viewer")
    st.write("Browse, search, and inspect the text contents of all indexed files.")
    
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

    selected_file: str = st.selectbox("Select Document to View", list(grouped.keys()))
    
    if selected_file:
        pages: List[Any] = grouped[selected_file]
        # Sort by page number
        pages.sort(key=lambda x: x.metadata.get("page", 1))
        
        st.markdown(f"### {selected_file} ({len(pages)} Pages)")
        
        # 1. Search Bar inside Document
        search_query: str = st.text_input("🔍 Search text inside this document:")
        
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
                
                # Highlight query term inside text
                # Uses safe HTML mark tag replacement
                escaped_query: str = re.escape(search_query)
                highlighted_content: str = re.sub(
                    f"({escaped_query})",
                    r"<mark style='background-color:#FDE047; padding: 2px; border-radius:3px; color:#000000;'>\1</mark>",
                    page_content,
                    flags=re.IGNORECASE
                )
                
                with st.expander(f"📖 Page {page_num} (Section: {section}) - Match Found"):
                    st.markdown(
                        f"<div style='background-color:#FFFFFF; border:1px solid #E4E7EC; padding:16px; border-radius:8px; max-height:350px; overflow-y:auto; font-family:monospace; font-size:12px; white-space:pre-wrap;'>"
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
        
        # Render match logs
        if search_query:
            if matched_pages_count == 0:
                st.info(f"No pages matched the query string '{search_query}'.")
            else:
                st.success(f"Found matches in {matched_pages_count} pages.")
