"""
Module for Groq Cloud API integration.
Conforms to the standardized response dictionary schema.
"""

import logging
from typing import Dict, Any
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)


class GroqProvider:
    """
    Client wrapper for Groq API.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL

    def generate(self, prompt: str, system_prompt: str = "", timeout: float = 12.0) -> Dict[str, Any]:
        """
        Sends generation request to Groq API.

        Parameters:
            prompt: User message context.
            system_prompt: System instruction prompt.
            timeout: Maximum seconds to wait.

        Returns:
            Dict matching response schema:
            {
                "answer": str,
                "provider": "groq",
                "tokens_used": int,
                "success": bool,
                "error": str | None
            }
        """
        if not self.api_key:
            logger.error("Groq API key is missing.")
            return {
                "answer": "",
                "provider": "groq",
                "tokens_used": 0,
                "success": False,
                "error": "Groq API key is not configured in environment variables."
            }

        logger.info(f"Sending prompt to Groq Cloud ({self.model})...")
        try:
            client = Groq(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.2,
                timeout=timeout
            )
            
            answer = completion.choices[0].message.content.strip()
            if not answer:
                logger.warning("Empty response received from Groq.")
                return {
                    "answer": "",
                    "provider": "groq",
                    "tokens_used": 0,
                    "success": False,
                    "error": "Empty response from Groq API."
                }
            
            tokens = completion.usage.total_tokens if completion.usage else 0
            logger.info(f"Groq generation completed. Tokens used: {tokens}")
            return {
                "answer": answer,
                "provider": "groq",
                "tokens_used": tokens,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error during Groq generation: {str(e)}")
            return {
                "answer": "",
                "provider": "groq",
                "tokens_used": 0,
                "success": False,
                "error": f"Groq API failure: {str(e)}"
            }
