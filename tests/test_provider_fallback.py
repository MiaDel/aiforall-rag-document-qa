"""
Unit tests for provider connection health checks and fallback routing.
"""

import pytest
import requests
from unittest.mock import patch, MagicMock
from config.settings import settings
from ingestion.pdf_loader import Document
from llms.llm_router import (
    LLMRouter,
    check_llama_connection,
    check_gemini_connection,
    check_grok_connection
)


@pytest.fixture
def mock_chunks():
    """
    Returns mock chunks for query routing validation.
    """
    return [
        {
            "document": Document(
                page_content="Repayment happens over 30 years.",
                metadata={
                    "doc_id": "doc1",
                    "file_name": "terms.pdf",
                    "page_number": 1,
                    "chunk_id": "doc1_p1_c0",
                    "section_title": "Repayment Period",
                    "source": "/workspace/terms.pdf"
                }
            ),
            "score": 0.8
        }
    ]


@patch("requests.get")
def test_check_llama_connection(mock_get):
    """
    Verifies check_llama_connection succeeds when HTTP responses are status 200.
    """
    # Success mock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    assert check_llama_connection() is True

    # Failure mock
    mock_get.side_effect = requests.ConnectionError("Offline")
    assert check_llama_connection() is False


def test_check_gemini_connection():
    """
    Verifies check_gemini_connection checks API key length.
    """
    original_key = settings.GEMINI_API_KEY
    
    settings.GEMINI_API_KEY = "test_valid_key_1234567"
    assert check_gemini_connection() is True

    settings.GEMINI_API_KEY = ""
    assert check_gemini_connection() is False
    
    settings.GEMINI_API_KEY = original_key


def test_check_grok_connection():
    """
    Verifies check_grok_connection checks API key length.
    """
    original_key = settings.GROQ_API_KEY
    
    settings.GROQ_API_KEY = "test_valid_key_1234567"
    assert check_grok_connection() is True

    settings.GROQ_API_KEY = ""
    assert check_grok_connection() is False
    
    settings.GROQ_API_KEY = original_key


@patch("llms.llama_provider.LlamaProvider.generate")
@patch("llms.gemini_provider.GeminiProvider.generate")
@patch("llms.groq_provider.GroqProvider.generate")
def test_fallback_sequence_all_fail(mock_groq, mock_gemini, mock_llama, mock_chunks):
    """
    Verifies router returns a failure message if all priority providers return failure states.
    """
    mock_llama.return_value = {"success": False, "error": "Llama offline"}
    mock_gemini.return_value = {"success": False, "error": "Gemini key error"}
    mock_groq.return_value = {"success": False, "error": "Groq rate limit"}
    
    router = LLMRouter()
    res = router.generate_answer("Rate query", mock_chunks)
    
    assert "all available llm endpoints" in res["answer"].lower()
    assert res["provider_used"] == "None (All Failed)"
