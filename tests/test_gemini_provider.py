"""
Unit tests for the migrated GeminiProvider utilizing the google-genai SDK.
Mocks the GenAI Client to run tests offline.
"""

import pytest
from unittest.mock import MagicMock, patch
from google.genai.errors import APIError
from config.settings import settings
from llms.gemini_provider import GeminiProvider


@pytest.fixture
def mock_genai_client():
    """
    Mocks the google-genai Client and generate_content responses.
    """
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "Mocked answer from Google Gemini."
        
        # Setup mock usage metadata
        mock_usage = MagicMock()
        mock_usage.total_token_count = 150
        mock_response.usage_metadata = mock_usage
        
        mock_client.models.generate_content.return_value = mock_response
        yield mock_client


def test_gemini_generation_success(mock_genai_client):
    """
    Verifies that GeminiProvider returns the correct response schema on success.
    """
    # Temporarily set key for test validation
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = "test_valid_key_1234567"
    
    provider = GeminiProvider()
    res = provider.generate("What is loan interest?", system_prompt="Answer concisely.")
    
    assert res["success"] is True
    assert res["answer"] == "Mocked answer from Google Gemini."
    assert res["provider"] == "gemini"
    assert res["tokens_used"] == 150
    assert res["error"] is None
    
    # Restore key
    settings.GEMINI_API_KEY = original_key


def test_gemini_missing_api_key():
    """
    Verifies behavior when Gemini key is missing.
    """
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = ""
    
    provider = GeminiProvider()
    res = provider.generate("Hello")
    
    assert res["success"] is False
    assert "missing" in res["error"].lower() or "not configured" in res["error"].lower()
    
    settings.GEMINI_API_KEY = original_key


def test_gemini_api_error_retry(mock_genai_client):
    """
    Verifies that GeminiProvider retries when encountering APIError and eventually fails.
    """
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = "test_valid_key_1234567"
    
    mock_genai_client.models.generate_content.side_effect = APIError("Rate limit exceeded.", response_json={})
    
    # Speed up sleep in retries
    with patch("time.sleep", return_value=None):
        provider = GeminiProvider()
        res = provider.generate("Hello")
        
        assert res["success"] is False
        assert "Rate limit exceeded" in res["error"]
        # Confirm it retried 3 times
        assert mock_genai_client.models.generate_content.call_count == 3
        
    settings.GEMINI_API_KEY = original_key
