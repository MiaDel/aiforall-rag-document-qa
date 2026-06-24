# config.py
# Central configuration file for the RAG pipeline.
# All shared settings live here. Never hardcode these values in other files.
# To change a setting, change it here and it updates everywhere automatically.

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

# Root directory of the project (wherever config.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where ChromaDB will store its files
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# Where sample/test documents are stored
SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, "data", "sample_docs")

# Temporary folder for files uploaded on the fly
TEMP_UPLOAD_PATH = os.path.join(BASE_DIR, "data", "temp_uploads")


# ─────────────────────────────────────────────
# DOCUMENT PARSING
# ─────────────────────────────────────────────

# Supported file types for the router
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff"]


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

# Maximum number of characters per chunk
CHUNK_SIZE = 500

# Number of characters to overlap between consecutive chunks
# Overlap prevents losing context at chunk boundaries
CHUNK_OVERLAP = 50


# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────

# Must be the same model for both ingestion and querying — never mix
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Number of chunks to embed at a time
# Keeps memory usage manageable for large documents
EMBEDDING_BATCH_SIZE = 100


# ─────────────────────────────────────────────
# VECTOR DATABASE (ChromaDB)
# ─────────────────────────────────────────────

# Name of the ChromaDB collection where chunks are stored
CHROMA_COLLECTION_NAME = "rag_documents"

# Number of chunks to retrieve for each query
TOP_K_RESULTS = 5


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

# Options: "llama3" (local via Ollama) or "gemini-pro" (requires API key)
LLM_PROVIDER = "llama3"

# Ollama settings (used when LLM_PROVIDER is "llama3")
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Gemini settings (used when LLM_PROVIDER is "gemini-pro")
# Never hardcode the API key here — it is read from the .env file
GEMINI_MODEL = "gemini-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

# Temperature controls how creative vs factual the LLM is
# 0.0 - 0.2 for factual Q&A (recommended)
# 0.7 for summarization if you want more fluent output
LLM_TEMPERATURE = 0.1

# Maximum tokens the LLM can return in a single response
LLM_MAX_TOKENS = 1024


# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────

# Cross-encoder model used for reranking retrieved chunks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Set to False initially until the core pipeline is working
# Then switch to True to enable reranking and compare results
RERANKING_ENABLED = False


# ─────────────────────────────────────────────
# RETRIEVAL PROMPT TEMPLATE
# ─────────────────────────────────────────────

# This is the prompt sent to the LLM with the retrieved context and user question
# {context} and {question} are filled in automatically at query time
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based strictly on the provided document context.
If the answer is not found in the context, say "I could not find this information in the provided documents."
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""