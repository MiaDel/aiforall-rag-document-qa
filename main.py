<<<<<<< HEAD
# main.py
# Terminal entry point for the RAG pipeline.
# Run this file to ingest documents and ask questions via the terminal.
# Usage:
#   Ingest a document:   python main.py --ingest path/to/document.pdf
#   Ask a question:      python main.py --query "What is the prepayment penalty?"
#   Do both at once:     python main.py --ingest path/to/document.pdf --query "What is the prepayment penalty?"

import argparse
import os

from config import SUPPORTED_FILE_TYPES, TEMP_UPLOAD_PATH
from ingestion.router import route_document
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_and_store
from query.question_embedder import embed_question
from query.retriever import retrieve_chunks
from query.reranker import rerank_chunks
from query.llm import generate_answer
from query.citation_formatter import format_citations
from config import RERANKING_ENABLED


# ─────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────

def ingest(file_path: str):
    """
    Runs the full ingestion pipeline on a given file.
    Steps: validate → parse → clean → chunk → embed → store
    """

    print(f"\n--- Starting ingestion for: {file_path} ---\n")

    # Step 1: Check the file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    # Step 2: Check the file type is supported
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in SUPPORTED_FILE_TYPES:
        print(f"ERROR: Unsupported file type '{file_extension}'.")
        print(f"Supported types are: {SUPPORTED_FILE_TYPES}")
        return

    # Step 3: Parse text using the correct parser for the file type
    print("Step 1/4: Parsing document...")
    raw_text, metadata = route_document(file_path)
    if not raw_text:
        print("ERROR: Could not extract text from document. File may be corrupt or empty.")
        return

    # Step 4: Clean and normalize the extracted text
    print("Step 2/4: Cleaning and normalizing text...")
    clean = clean_text(raw_text)

    # Step 5: Split into chunks
    print("Step 3/4: Chunking text...")
    chunks = chunk_text(clean, metadata)
    print(f"         Created {len(chunks)} chunks.")

    # Step 6: Embed and store in ChromaDB
    print("Step 4/4: Embedding and storing in ChromaDB...")
    embed_and_store(chunks)

    print(f"\n--- Ingestion complete for: {file_path} ---\n")


# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

def query(question: str):
    """
    Runs the full query pipeline for a given question.
    Steps: embed question → retrieve chunks → (rerank) → generate answer → format citations
    """

    print(f"\n--- Processing question: '{question}' ---\n")

    # Step 1: Embed the question
    print("Step 1/4: Embedding question...")
    question_vector = embed_question(question)

    # Step 2: Retrieve relevant chunks from ChromaDB
    print("Step 2/4: Retrieving relevant chunks...")
    chunks = retrieve_chunks(question_vector)
    if not chunks:
        print("No relevant chunks found. Make sure documents have been ingested first.")
        return

    # Step 3: Rerank chunks if enabled in config
    if RERANKING_ENABLED:
        print("Step 3/4: Reranking chunks...")
        chunks = rerank_chunks(question, chunks)
    else:
        print("Step 3/4: Reranking skipped (set RERANKING_ENABLED = True in config.py to enable).")

    # Step 4: Send to LLM and get answer
    print("Step 4/4: Generating answer...")
    answer = generate_answer(question, chunks)

    # Step 5: Format and print citations
    citations = format_citations(chunks)

    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    print(citations)
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    # Set up terminal argument parsing
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A System — Terminal Interface"
    )
    parser.add_argument(
        "--ingest",
        type=str,
        help="Path to a document to ingest (PDF, Word, or image)",
        default=None
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask about the ingested documents",
        default=None
    )

    args = parser.parse_args()

    # Must provide at least one argument
    if not args.ingest and not args.query:
        print("ERROR: Please provide --ingest, --query, or both.")
        print("Examples:")
        print("  python main.py --ingest path/to/document.pdf")
        print('  python main.py --query "What is the prepayment penalty?"')
        print('  python main.py --ingest path/to/document.pdf --query "What is the prepayment penalty?"')
        return

    # Run ingestion if a file path was provided
    if args.ingest:
        ingest(args.ingest)

    # Run query if a question was provided
    if args.query:
        query(args.query)

=======
from ingestion.file_router import route_file

def main():
    print("RAG Pipeline Starting")
>>>>>>> 6b0c89a0fb9f01e94c17060d4bba11921568d2b8

if __name__ == "__main__":
    main()