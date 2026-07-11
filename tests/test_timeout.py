"""
Unit tests verifying timeout handling in LLM providers.
"""

import pytest
import requests
from unittest.mock import patch, MagicMock
from config.settings import settings
from llms.llama_provider import LlamaProvider
from llms.gemini_provider import GeminiProvider
from llms.groq_provider import GroqProvider
from llms.openai_provider import OpenAIProvider


@patch("requests.post")
def test_llama_timeout(mock_post):
    """
    Verifies LlamaProvider translates HTTP connection timeout.
    """
    mock_post.side_effect = requests.exceptions.Timeout("Ollama connection timed out.")
    
    provider = LlamaProvider()
    res = provider.generate("Hello", timeout=1.0)
    
    assert res["success"] is False
    assert "timeout" in res["error"].lower() or "timed out" in res["error"].lower()


@patch("google.genai.Client")
def test_gemini_timeout(mock_client_class):
    """
    Verifies GeminiProvider handles timeout failures.
    """
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    # Mock exception for timeout
    mock_client.models.generate_content.side_effect = Exception("DeadlineExceeded: connection timed out")
    
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = "test_valid_key_1234567"
    
    with patch("time.sleep", return_value=None):
        provider = GeminiProvider()
        res = provider.generate("Hello", timeout=1.0)
        
        assert res["success"] is False
        assert "timeout" in res["error"].lower() or "deadlineexceeded" in res["error"].lower()
        
    settings.GEMINI_API_KEY = original_key


@patch("llms.groq_provider.Groq")
def test_groq_timeout(mock_groq_class):
    """
    Verifies GroqProvider handles request timeouts.
    """
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_client.chat.completions.create.side_effect = Exception("TimeoutError: Groq request exceeded timeout")
    
    original_key = settings.GROQ_API_KEY
    settings.GROQ_API_KEY = "test_valid_key_1234567"
    
    provider = GroqProvider()
    res = provider.generate("Hello", timeout=1.0)
    
    assert res["success"] is False
    assert "timeout" in res["error"].lower() or "timeouterror" in res["error"].lower()
    
    settings.GROQ_API_KEY = original_key


@patch("llms.openai_provider.OpenAI")
def test_openai_timeout(mock_openai_class):
    """
    Verifies OpenAIProvider handles request timeouts.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.side_effect = Exception("APITimeoutError: OpenAI request timed out")
    
    original_key = settings.OPENAI_API_KEY
    settings.OPENAI_API_KEY = "test_valid_key_1234567"
    
    provider = OpenAIProvider()
    res = provider.generate("Hello", timeout=1.0)
    
    assert res["success"] is False
    assert "timeout" in res["error"].lower() or "apitimeouterror" in res["error"].lower()
    
    settings.OPENAI_API_KEY = original_key
