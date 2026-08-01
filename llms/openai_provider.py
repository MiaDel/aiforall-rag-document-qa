"""
Module for OpenAI API integration as a fallback provider.
Conforms to the standardized response dictionary schema.
"""

import logging
from typing import Dict, Any
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    Client wrapper for OpenAI API completions.
    """

    def generate(self, prompt: str, system_prompt: str = "", timeout: float = 12.0) -> Dict[str, Any]:
        """
        Sends generation request to OpenAI API.
        """
        # Read dynamically on every call so updated settings/secrets are immediately recognized
        api_key = settings.OPENAI_API_KEY
        model = settings.OPENAI_MODEL or "gpt-4o-mini"

        if not api_key:
            logger.error("OpenAI API key is missing.")
            return {
                "answer": "",
                "provider": "openai",
                "tokens_used": 0,
                "success": False,
                "error": "OpenAI API key is not configured in environment variables or secrets."
            }

        logger.info(f"Sending prompt to OpenAI API ({model})...")
        try:
            client = OpenAI(api_key=api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.2,
                timeout=timeout
            )
            
            answer = completion.choices[0].message.content.strip()
            if not answer:
                logger.warning("Empty response received from OpenAI.")
                return {
                    "answer": "",
                    "provider": "openai",
                    "tokens_used": 0,
                    "success": False,
                    "error": "Empty response from OpenAI API."
                }
            
            tokens = completion.usage.total_tokens if completion.usage else 0
            logger.info(f"OpenAI generation completed. Tokens used: {tokens}")
            return {
                "answer": answer,
                "provider": "openai",
                "tokens_used": tokens,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error during OpenAI generation: {str(e)}")
            return {
                "answer": "",
                "provider": "openai",
                "tokens_used": 0,
                "success": False,
                "error": f"OpenAI API failure: {str(e)}"
            }