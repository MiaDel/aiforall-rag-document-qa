"""
Right-hand source panel component for the Loan RAG Chatbot.
Visualizes references, page numbers, and relevance scores for the latest response with expandable chunk previews.
"""

import streamlit as st


def render_source_panel():
    """
    Renders the right panel containing document citations and chunk text previews.
    """
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
                '<span style="font-weight:700; font-size:16px; color:#111827;">Top Sources</span>'
                '<span style="font-size:12px; color:#4F46E5; font-weight:600; cursor:pointer;">View All</span>'
                '</div>', unsafe_allow_html=True)

    # 1. Fetch last assistant message citations
    history = st.session_state.get("chat_history", [])
    assistant_msgs = [m for m in history if m["role"] == "assistant" and m.get("metadata")]
    
    if not assistant_msgs:
        # Default empty state
        st.markdown(
            '<div style="text-align:center; padding: 40px 20px; color:#9CA3AF; font-size:13px; border:1px dashed #E4E7EC; border-radius:12px; background-color:#FFFFFF;">'
            '📁 No sources cited yet.<br/>Ask a question to view document citations.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Get citations from the last assistant message
        last_meta = assistant_msgs[-1]["metadata"]
        sources = last_meta.get("sources", [])
        confidence = last_meta.get("confidence", 0.7)
        evidence = last_meta.get("evidence", [])

        if not sources:
            st.markdown(
                '<div style="text-align:center; padding: 40px 20px; color:#9CA3AF; font-size:13px; border:1px dashed #E4E7EC; border-radius:12px; background-color:#FFFFFF;">'
                '📄 Answer generated from general context or no chunks matches.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            # Render an expander for each source
            for idx, source in enumerate(sources):
                file_name = source.get("file", "Unknown Contract")
                page_num = source.get("page", 1)
                
                # Compute mock relevance percentage based on RAG confidence
                score_percentage = int(max(40, confidence * 100 - (idx * 6)))
                
                # Fetch corresponding raw chunk text
                chunk_text = "No snippet content matched."
                if idx < len(evidence):
                    chunk_text = evidence[idx].strip()
                
                # Render clean expandable preview card
                with st.expander(f"📕 {file_name} (Page {page_num})"):
                    st.markdown(f"**Relevance:** `{score_percentage}%`")
                    st.markdown(
                        f'<div style="background-color:#F9FAFB; border-left:3px solid #4F46E5; padding:8px 12px; border-radius:4px; font-size:12px; color:#374151; max-height: 150px; overflow-y: auto;">'
                        f'  {chunk_text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # 2. Bottom Information box
    st.markdown(
        '<div style="background-color:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:12px 16px; display:flex; align-items:start; margin-top:16px;">'
        '  <span style="font-size:18px; margin-right:12px; color:#3B82F6;">ℹ️</span>'
        '  <span style="font-size:11px; color:#1E3A8A; font-weight:500; line-height:1.4;">'
        '    Sources show the most relevant document chunks used to answer your question.'
        '  </span>'
        '</div>',
        unsafe_allow_html=True
    )
