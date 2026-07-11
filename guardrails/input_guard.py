"""
Module for checking user prompts against input guardrail violations:
- Prompt injection
- Jailbreak attempts
- Malicious instructions
- Code execution attempts
- Empty or oversized queries
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class InputGuardrail:
    """
    Evaluates incoming user prompts for security violations before passing them to retrieval or LLMs.
    """

    # Common prompt injection / jailbreak phrases
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:a\s+)?roleplay", re.IGNORECASE),
        re.compile(r"delete\s+database", re.IGNORECASE),
        re.compile(r"tell\s+me\s+(?:your\s+)?secrets", re.IGNORECASE),
        re.compile(r"bypass\s+restrictions", re.IGNORECASE),
        re.compile(r"disregard\s+rules", re.IGNORECASE),
        re.compile(r"do\s+anything\s+now", re.IGNORECASE),  # DAN jailbreaks
        re.compile(r"system\s+override", re.IGNORECASE)
    ]

    # Code execution patterns
    CODE_PATTERNS = [
        re.compile(r"\bimport\s+(?:os|sys|subprocess|shutil|socket|urllib)\b"),
        re.compile(r"\b__import__\b"),
        re.compile(r"\beval\s*\("),
        re.compile(r"\bexec\s*\("),
        re.compile(r"\bos\.system\b"),
        re.compile(r"\bsubprocess\.(?:Popen|run|call)\b"),
        re.compile(r"\brm\s+-rf\b")
    ]

    def __init__(self, max_query_length: int = 2000):
        self.max_query_length = max_query_length

    def validate(self, query: str) -> Tuple[bool, str | None]:
        """
        Validates the incoming query string.

        Parameters:
            query: The user prompt.

        Returns:
            Tuple of (is_safe: bool, refusal_reason: str | None).
        """
        # 1. Empty check
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.warning("Empty query intercepted by InputGuardrail.")
            return False, "Your question is empty. Please enter a valid question."

        # 2. Oversized check
        if len(cleaned_query) > self.max_query_length:
            logger.warning(f"Oversized query intercepted: {len(cleaned_query)} characters.")
            return False, f"Your question is too long (maximum allowed length is {self.max_query_length} characters)."

        # 3. Prompt injection & Jailbreak checks
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(cleaned_query):
                logger.error(f"Prompt injection pattern detected: '{pattern.pattern}'")
                return False, "This query violates safety guardrails. Jailbreak or injection attempt detected."

        # 4. Code execution checks
        for pattern in self.CODE_PATTERNS:
            if pattern.search(cleaned_query):
                logger.error(f"Malicious code execution pattern detected: '{pattern.pattern}'")
                return False, "This query violates safety guardrails. Code execution attempts are prohibited."

        logger.info("Query successfully passed input guardrails.")
        return True, None
