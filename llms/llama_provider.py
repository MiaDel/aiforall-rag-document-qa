"""
Module for local Llama3 LLM execution using Ollama API.
Includes timeout handling and connection error catch blocks.
Conforms to the standardized response dictionary schema.
"""

import logging
import requests
from typing import Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class LlamaProvider:
    """
    Client wrapper for local Llama3 execution via Ollama HTTP API.
    """

    def __init__(self):
        self.host = settings.OLLAMA_HOST.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.endpoint = f"{self.host}/api/generate"

    def generate(self, prompt: str, system_prompt: str = "", timeout: float = 120.0) -> Dict[str, Any]:
        """
        Queries Ollama API for text generation.

        Parameters:
            prompt: User message context.
            system_prompt: Optional instructions constraint.
            timeout: API call timeout.

        Returns:
            Dict matching response schema:
            {
                "answer": str,
                "provider": "llama3",
                "tokens_used": int,
                "success": bool,
                "error": str | None
            }
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        logger.info(f"Sending prompt to Ollama Llama3 ({self.model})...")
        try:
            response = requests.post(self.endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("response", "").strip()
            
            if not answer:
                logger.warning("Empty response received from local Llama3.")
                return {
                    "answer": "",
                    "provider": "llama3",
                    "tokens_used": 0,
                    "success": False,
                    "error": "Empty response from Ollama API."
                }
            
            # Exact token evaluation counts from Ollama response keys
            prompt_tokens = data.get("prompt_eval_count", 0)
            eval_tokens = data.get("eval_count", 0)
            tokens_used = prompt_tokens + eval_tokens

            logger.info(f"Llama3 generation completed. Tokens used: {tokens_used}")
            return {
                "answer": answer,
                "provider": "llama3",
                "tokens_used": tokens_used,
                "success": True,
                "error": None
            }

        except requests.exceptions.Timeout as t_err:
            logger.error(f"Ollama Llama3 generation timed out (limit={timeout}s).")
            return {
                "answer": "",
                "provider": "llama3",
                "tokens_used": 0,
                "success": False,
                "error": f"Ollama local model generation timed out: {str(t_err)}"
            }
            
        except requests.exceptions.ConnectionError as c_err:
            logger.error(f"Could not connect to Ollama at {self.host}.")
            return {
                "answer": "",
                "provider": "llama3",
                "tokens_used": 0,
                "success": False,
                "error": f"Connection to Ollama failed: {str(c_err)}"
            }
            
        except Exception as e:
            logger.exception(f"Unexpected error during Ollama generation: {str(e)}")
            return {
                "answer": "",
                "provider": "llama3",
                "tokens_used": 0,
                "success": False,
                "error": f"Ollama generation failed: {str(e)}"
            }
