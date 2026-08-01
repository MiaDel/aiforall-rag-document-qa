import os
import logging
from pathlib import Path
from dotenv import load_dotenv

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

# HuggingFace Environment settings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Explicitly ensure offline flags are cleared in runtime
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

# 2. Setup standard logging limits to reduce terminal clutter noise
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_secret_or_env(key: str, default: str = "") -> str:
    """Helper to retrieve configuration value.
    First checks Streamlit secrets, then checks environment variables/defaults.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


class Settings:
    """Application configurations. Loads values from environment variables or applies defaults."""
    # System settings
    LOG_LEVEL: str = get_secret_or_env("LOG_LEVEL", "INFO")
    CHROMA_PERSIST_DIRECTORY: str = get_secret_or_env("CHROMA_PERSIST_DIRECTORY", str(CHROMA_PERSIST_DIR.resolve()))
    COLLECTION_NAME: str = get_secret_or_env("COLLECTION_NAME", "loan_documents")

    # Ingestion & Chunking
    CHUNK_SIZE: int = int(get_secret_or_env("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(get_secret_or_env("CHUNK_OVERLAP", "200"))

    # Retrieval parameters
    TOP_K: int = int(get_secret_or_env("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(get_secret_or_env("SIMILARITY_THRESHOLD", "0.3"))

    # Embeddings & Reranking models
    EMBEDDING_MODEL_NAME: str = get_secret_or_env("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    RERANKER_MODEL_NAME: str = get_secret_or_env("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM Provider settings
    LLM_PROVIDER: str = get_secret_or_env("LLM_PROVIDER", "openai")
    OLLAMA_HOST: str = get_secret_or_env("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = get_secret_or_env("OLLAMA_MODEL", "llama3")
    GEMINI_API_KEY: str = get_secret_or_env("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = get_secret_or_env("GROQ_API_KEY", "")
    GROQ_MODEL: str = get_secret_or_env("GROQ_MODEL", "llama3-70b-8192")
    OPENAI_API_KEY: str = get_secret_or_env("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = get_secret_or_env("OPENAI_MODEL", "gpt-4o-mini")


# Singleton configuration object
settings: Settings = Settings()