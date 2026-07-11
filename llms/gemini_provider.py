"""
Module for Google Gemini API integration using the new google-genai SDK.
Handles key validation, request timeout, retries, and returns a consistent response schema.
"""

import time
import logging
from typing import Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Client wrapper for Google Gemini API using the new google-genai SDK.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = "gemini-1.5-flash"

    def generate(self, prompt: str, system_prompt: str = "", timeout: float = 12.0) -> Dict[str, Any]:
        """
        Sends generation request to Gemini API with retries and timeout options.

        Parameters:
            prompt: User query content.
            system_prompt: System constraints instructions.
            timeout: Request execution timeout in seconds.

        Returns:
            Dict matching response schema:
            {
                "answer": str,
                "provider": "gemini",
                "tokens_used": int,
                "success": bool,
                "error": str | None
            }
        """
        if not self.api_key:
            logger.error("Gemini API key is missing.")
            return {
                "answer": "",
                "provider": "gemini",
                "tokens_used": 0,
                "success": False,
                "error": "Gemini API key is not configured in environment variables."
            }

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Sending prompt to Gemini (attempt {attempt}/{max_retries})...")
                
                # Create client with timeout HTTP option
                client = genai.Client(
                    api_key=self.api_key,
                    http_options={"timeout": timeout}
                )

                config = types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    temperature=0.2
                )

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                answer = response.text.strip() if response.text else ""
                if not answer:
                    logger.warning("Empty text response returned by Gemini API.")
                    raise ValueError("Empty response text.")

                tokens = 0
                if response.usage_metadata:
                    tokens = response.usage_metadata.total_token_count

                logger.info(f"Gemini generation completed. Tokens used: {tokens}")
                return {
                    "answer": answer,
                    "provider": "gemini",
                    "tokens_used": tokens,
                    "success": True,
                    "error": None
                }

            except APIError as api_err:
                logger.warning(f"Gemini API error (attempt {attempt}): {str(api_err)}")
                if attempt == max_retries:
                    return {
                        "answer": "",
                        "provider": "gemini",
                        "tokens_used": 0,
                        "success": False,
                        "error": f"Gemini APIError: {str(api_err)}"
                    }
                time.sleep(retry_delay * attempt)

            except Exception as e:
                logger.error(f"Unexpected Gemini provider failure: {str(e)}")
                if attempt == max_retries:
                    return {
                        "answer": "",
                        "provider": "gemini",
                        "tokens_used": 0,
                        "success": False,
                        "error": f"Gemini connection/timeout failure: {str(e)}"
                    }
                time.sleep(retry_delay * attempt)

        # Fallback return
        return {
            "answer": "",
            "provider": "gemini",
            "tokens_used": 0,
            "success": False,
            "error": "Gemini generation failed after max retries."
        }
