import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from context.pdf_parser import parse_pdf
from context.chunker import create_chunks
from context.retrievers.retriever import (
    index_chunks,
    retrieve
)

st.set_page_config(page_title="RAG Document QA")
st.title("📄 RAG Document QA Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = f"data/temp_uploads/{uploaded_file.name}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    with st.spinner("Indexing document..."):
        docs = parse_pdf(save_path)
        chunks = create_chunks(docs)
        index_chunks(chunks)

    st.success("Document Indexed Successfully!")

    question = st.text_input(
        "Ask a question about the document"
    )

    if st.button("Get Answer"):

        if question.strip():

            result = retrieve(question)

            docs = result["documents"][0]
            metas = result["metadatas"][0]

            st.subheader("Retrieved Chunks")

            for i, doc in enumerate(docs):

                page = metas[i].get("page", "N/A")

                with st.expander(
                    f"Chunk {i+1} (Page {page})"
                ):
                    st.write(doc)