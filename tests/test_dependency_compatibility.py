"""
Unit tests verifying third-party library import compatibility and stability.
"""

import sys
import pytest


def test_python_compatibility():
    """
    Documents and asserts current python environment.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert major == 3
    # Log information about active python version
    print(f"Active testing Python version: {major}.{minor}")


def test_core_library_imports():
    """
    Verifies that all direct, transitive, and orchestrator modules load correctly.
    """
    # 1. Parsing libraries
    import fitz  # PyMuPDF
    import pdfplumber
    assert fitz.__name__ == "fitz"
    assert pdfplumber.__name__ == "pdfplumber"

    # 2. Vector DB & Embeddings
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder
    assert chromadb.__name__ == "chromadb"
    assert SentenceTransformer.__name__ == "SentenceTransformer"
    assert CrossEncoder.__name__ == "CrossEncoder"

    # 3. LLM API Providers
    from google import genai
    from groq import Groq
    from openai import OpenAI
    import ollama
    assert genai.__name__ == "google.genai"
    assert Groq.__name__ == "Groq"
    assert OpenAI.__name__ == "OpenAI"
    assert ollama.__name__ == "ollama"

    # 4. LangChain Integration
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    assert Document.__name__ == "Document"
    assert RunnableLambda.__name__ == "RunnableLambda"

    # 5. LangGraph workflow
    from langgraph.graph import StateGraph, END
    assert StateGraph.__name__ == "StateGraph"
    assert END == "__end__"
