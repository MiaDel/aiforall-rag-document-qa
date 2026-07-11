"""
Unit tests for handling invalid API key errors from providers.
Verifies that authentication errors map to success=False schemas rather than crashing.
"""

import pytest
from unittest.mock import patch, MagicMock
from google.genai.errors import APIError
from config.settings import settings
from llms.gemini_provider import GeminiProvider
from llms.groq_provider import GroqProvider
from llms.openai_provider import OpenAIProvider


@patch("google.genai.Client")
def test_gemini_invalid_api_key(mock_client_class):
    """
    Verifies GeminiProvider catches invalid key APIError responses.
    """
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_client.models.generate_content.side_effect = APIError("API_KEY_INVALID", response_json={})
    
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = "invalid_key_value_string"
    
    with patch("time.sleep", return_value=None):
        provider = GeminiProvider()
        res = provider.generate("Hello")
        
        assert res["success"] is False
        assert "API_KEY_INVALID" in res["error"]
        
    settings.GEMINI_API_KEY = original_key


@patch("llms.groq_provider.Groq")
def test_groq_invalid_api_key(mock_groq_class):
    """
    Verifies GroqProvider handles invalid credential failures.
    """
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # Mock completions.create to raise exception
    mock_client.chat.completions.create.side_effect = Exception("Unauthorized: Invalid API Key")
    
    original_key = settings.GROQ_API_KEY
    settings.GROQ_API_KEY = "invalid_key_value_string"
    
    provider = GroqProvider()
    res = provider.generate("Hello")
    
    assert res["success"] is False
    assert "Invalid API Key" in res["error"] or "Unauthorized" in res["error"]
    
    settings.GROQ_API_KEY = original_key


@patch("llms.openai_provider.OpenAI")
def test_openai_invalid_api_key(mock_openai_class):
    """
    Verifies OpenAIProvider handles authentication failures.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.side_effect = Exception("AuthenticationError: Incorrect API key")
    
    original_key = settings.OPENAI_API_KEY
    settings.OPENAI_API_KEY = "invalid_key_value_string"
    
    provider = OpenAIProvider()
    res = provider.generate("Hello")
    
    assert res["success"] is False
    assert "Incorrect API key" in res["error"] or "AuthenticationError" in res["error"]
    
    settings.OPENAI_API_KEY = original_key
