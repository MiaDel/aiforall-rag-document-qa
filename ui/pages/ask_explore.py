"""
Ask & Explore Page component.
An advanced playground allowing users to query dense, sparse, or hybrid indices
and inspect raw matching scores directly.
"""

import streamlit as st
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.reranker import CrossEncoderReranker
from config.settings import settings


def run_ask_explore_page():
    """
    Renders the Ask & Explore playground.
    """
    st.title("🔍 Ask & Explore Playground")
    st.write(
        "Tune search configurations dynamically and inspect raw database "
        "relevance scores before generation."
    )

    # Sidebar parameters or local playground parameters
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        search_mode = st.selectbox(
            "Retrieval Mode",
            ["Hybrid (RRF)", "Dense Semantic (Chroma)", "Sparse Keyword (BM25)"]
        )
    with col_cfg2:
        top_k = st.slider("Top K Retrieve", min_value=1, max_value=20, value=settings.TOP_K)
    with col_cfg3:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings.SIMILARITY_THRESHOLD,
            step=0.05
        )

    query = st.text_input("Enter search query:")

    if query:
        with st.status("Executing retrieval...", expanded=True) as status:
            from utils.resource_manager import get_chroma_manager
            chroma_mgr = get_chroma_manager()
            
            if search_mode == "Dense Semantic (Chroma)":
                st.write("Initializing DenseRetriever...")
                retriever = DenseRetriever(chroma_manager=chroma_mgr)
                results = retriever.retrieve(query, top_k=top_k)
            elif search_mode == "Sparse Keyword (BM25)":
                st.write("Initializing BM25Retriever...")
                retriever = BM25Retriever(chroma_manager=chroma_mgr)
                results = retriever.retrieve(query, top_k=top_k)
            else:
                st.write("Initializing HybridRetriever...")
                retriever = HybridRetriever(chroma_manager=chroma_mgr)
                results = retriever.retrieve(query, top_k=top_k)
                
            status.update(label=f"Retrieved {len(results)} matches", state="complete")

        if not results:
            st.warning("No matches found for this query in database.")
        else:
            st.markdown(f"### Retrieval Results ({len(results)})")
            
            for idx, item in enumerate(results):
                doc = item["document"]
                score = item["score"]
                meta = doc.metadata
                
                # Check threshold gating
                is_above_threshold = score >= threshold
                badge_color = "green" if is_above_threshold else "red"
                status_text = "Above Threshold" if is_above_threshold else "Below Threshold"
                
                st.markdown(
                    f'<div style="background-color:#FFFFFF; border: 1px solid #EEF2F6; padding:16px; border-radius:12px; margin-bottom:12px;">'
                    f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
                    f'    <span style="font-weight:600; font-size:14px; color:#111827;">📕 {meta.get("file_name")} (Page {meta.get("page_number")})</span>'
                    f'    <span style="font-size:12px; font-weight:700; color:{badge_color}; background-color:#F9FAFB; padding:4px 8px; border-radius:6px;">'
                    f'      Score: {score:.4f} ({status_text})'
                    f'    </span>'
                    f'  </div>'
                    f'  <p style="font-size:13px; color:#374151; line-height:1.5; margin:0;">{doc.page_content}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
