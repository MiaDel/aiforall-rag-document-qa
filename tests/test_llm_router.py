"""
Unit tests for the LLM router and provider integrations (LlamaProvider, GeminiProvider, GroqProvider, LLMRouter).
Uses unittest.mock to simulate API behaviors and verify failover routing offline.
"""

import pytest
import requests
from unittest.mock import MagicMock, patch
from config.settings import settings
from ingestion.pdf_loader import Document
from llms.llama_provider import LlamaProvider
from llms.llm_router import LLMRouter


@pytest.fixture
def mock_chunks():
    """
    Mock retrieved chunks list for citation testing.
    """
    return [
        {
            "document": Document(
                page_content="Fragment 1 text.",
                metadata={
                    "doc_id": "doc1",
                    "file_name": "agreement_a.pdf",
                    "page_number": 2,
                    "section_title": "Section A",
                    "source": "/path/to/agreement_a.pdf"
                }
            ),
            "score": 0.8
        },
        {
            "document": Document(
                page_content="Fragment 2 text.",
                metadata={
                    "doc_id": "doc2",
                    "file_name": "agreement_b.pdf",
                    "page_number": 5,
                    "section_title": "Section B",
                    "source": "/path/to/agreement_b.pdf"
                }
            ),
            "score": 0.6
        }
    ]


@patch("requests.post")
def test_llama3_provider_generation(mock_post):
    """
    Verifies that LlamaProvider parses success payloads from Ollama HTTP endpoints.
    """
    # Mock successful requests response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "This is a mock answer from Llama3.",
        "prompt_eval_count": 30,
        "eval_count": 20
    }
    mock_post.return_value = mock_resp
    
    provider = LlamaProvider()
    res = provider.generate(prompt="Hello", system_prompt="Test")
    
    assert res["success"] is True
    assert res["answer"] == "This is a mock answer from Llama3."
    assert res["provider"] == "llama3"
    assert res["tokens_used"] == 50
    assert res["error"] is None
    mock_post.assert_called_once()


@patch("requests.post")
def test_llama3_provider_timeout(mock_post):
    """
    Verifies that LlamaProvider handles connection timeouts without throwing uncaught exceptions.
    """
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out.")
    
    provider = LlamaProvider()
    res = provider.generate(prompt="Hello", timeout=1.0)
    
    assert res["success"] is False
    assert "timeout" in res["error"].lower() or "timed out" in res["error"].lower()


def test_empty_context_refusal():
    """
    Verifies that providing empty retrieval contexts immediately triggers a refusal response.
    """
    router = LLMRouter()
    res = router.generate_answer(query="What is the rate?", retrieved_chunks=[])
    
    assert res["answer"] == "I could not find the answer in the uploaded documents."
    assert res["sources"] == []
    assert res["confidence"] == 0.0
    assert res["provider_used"] == "Context Guardrail"


@patch("llms.llama_provider.LlamaProvider.generate")
@patch("llms.gemini_provider.GeminiProvider.generate")
@patch("llms.groq_provider.GroqProvider.generate")
def test_provider_fallback_routing(mock_groq, mock_gemini, mock_llama, mock_chunks):
    """
    Verifies the fallback mechanism: Llama fails, Gemini fails, Groq succeeds.
    """
    # Llama throws connection failure
    mock_llama.return_value = {"success": False, "error": "Ollama offline."}
    # Gemini throws key error
    mock_gemini.return_value = {"success": False, "error": "Missing API Key."}
    # Groq returns successfully
    mock_groq.return_value = {
        "success": True,
        "answer": "Answer generated via Groq.",
        "provider": "groq",
        "tokens_used": 75,
        "error": None
    }
    
    router = LLMRouter()
    res = router.generate_answer(query="Test Query", retrieved_chunks=mock_chunks)
    
    # Assert result came from Groq
    assert res["answer"] == "Answer generated via Groq."
    assert res["provider_used"] == "groq"
    assert res["tokens_used"] == 75
    
    # Verify both failures were logged and Groq was called
    mock_llama.assert_called_once()
    mock_gemini.assert_called_once()
    mock_groq.assert_called_once()


def test_citation_generation(mock_chunks):
    """
    Verifies that citations and confidence values are parsed correctly from retrieved metadata.
    """
    router = LLMRouter()
    
    # Mock llama provider to succeed immediately so we can inspect output metadata mappings
    mock_success = {
        "success": True,
        "answer": "Test Answer",
        "provider": "llama3",
        "tokens_used": 15,
        "error": None
    }
    with patch("llms.llama_provider.LlamaProvider.generate", return_value=mock_success):
        res = router.generate_answer(query="Rate query", retrieved_chunks=mock_chunks)
        
        # Verify sources mapped
        assert len(res["sources"]) == 2
        # Verify confidence score is average of chunk scores (0.8 + 0.6) / 2 = 0.7
        assert res["confidence"] == 0.7
        # Verify metadata references mapped
        assert len(res["chunk_references"]) == 2
        assert res["chunk_references"][0]["doc_id"] == "doc1"
