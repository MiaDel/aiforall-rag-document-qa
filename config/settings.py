"""
=========================================================
File Name : settings.py
Project   : Multi-Document RAG Chatbot
Author    : Kaif Rehman
Description:
Configuration module for loading and validating system environment variables.
Applies environment setup rules for HuggingFace caching, telemetry,
and offline execution switches.

Technologies:
- Python standard os/pathlib libraries
- python-dotenv
=========================================================
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Pinned HuggingFace Environment settings
os.environ["HF_HOME"] = "models"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Automatically enable offline execution once local models are present
if Path("./models/hub").exists() or Path("./models/models--BAAI--bge-large-en-v1.5").exists():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

# 2. Setup standard logging limits to reduce terminal clutter noise
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Automatically load the .env file if it exists
load_dotenv()

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

# Create directories if they do not exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    """Application configurations. Loads values from environment variables or applies defaults.

    Responsibilities:
    1. Parse system log levels.
    2. Maintain database connection parameters.
    3. Maintain chunk size settings.
    4. Maintain api key credentials.
    """
    # System settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", str(CHROMA_PERSIST_DIR.resolve()))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "loan_documents")

    # Ingestion & Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval parameters
    TOP_K: int = int(os.getenv("TOP_K", 5))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))

    # Embeddings & Reranking models
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM Provider settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Singleton configuration object
settings: Settings = Settings()
