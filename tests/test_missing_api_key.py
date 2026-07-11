"""
Unit tests verifying behavior when API keys are missing from settings.
Ensures providers fail at the local validation level without invoking client endpoints.
"""

import pytest
from unittest.mock import patch
from config.settings import settings
from llms.gemini_provider import GeminiProvider
from llms.groq_provider import GroqProvider
from llms.openai_provider import OpenAIProvider


def test_gemini_missing_key():
    """
    Verifies GeminiProvider fails immediately when API key is empty.
    """
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = ""
    
    with patch("google.genai.Client") as mock_client:
        provider = GeminiProvider()
        res = provider.generate("Hello")
        
        assert res["success"] is False
        assert "missing" in res["error"].lower() or "not configured" in res["error"].lower()
        # Verify client was not instantiated
        mock_client.assert_not_called()
        
    settings.GEMINI_API_KEY = original_key


def test_groq_missing_key():
    """
    Verifies GroqProvider fails immediately when API key is empty.
    """
    original_key = settings.GROQ_API_KEY
    settings.GROQ_API_KEY = ""
    
    with patch("groq.Groq") as mock_client:
        provider = GroqProvider()
        res = provider.generate("Hello")
        
        assert res["success"] is False
        assert "missing" in res["error"].lower() or "not configured" in res["error"].lower()
        mock_client.assert_not_called()
        
    settings.GROQ_API_KEY = original_key


def test_openai_missing_key():
    """
    Verifies OpenAIProvider fails immediately when API key is empty.
    """
    original_key = settings.OPENAI_API_KEY
    settings.OPENAI_API_KEY = ""
    
    with patch("openai.OpenAI") as mock_client:
        provider = OpenAIProvider()
        res = provider.generate("Hello")
        
        assert res["success"] is False
        assert "missing" in res["error"].lower() or "not configured" in res["error"].lower()
        mock_client.assert_not_called()
        
    settings.OPENAI_API_KEY = original_key
