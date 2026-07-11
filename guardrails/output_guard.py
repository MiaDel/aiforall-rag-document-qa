"""
Module for checking LLM generated output against safety violations:
- Hallucinated citations
- Confidence score threshold violations
- Citation source verification
"""

import logging
from typing import List, Dict, Any, Tuple
from config.settings import settings

logger = logging.getLogger(__name__)


class OutputGuardrail:
    """
    Evaluates generated output and references to verify facts and citation validity.
    """

    def __init__(self, confidence_threshold: float | None = None):
        self.confidence_threshold = confidence_threshold or settings.SIMILARITY_THRESHOLD

    def validate_answer(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence_score: float,
        cited_files: List[str],
        cited_pages: List[int]
    ) -> Tuple[bool, str, List[str], List[int]]:
        """
        Validates LLM-generated outputs, ensuring alignment with grounding context.

        Parameters:
            answer_text: Generated response string.
            retrieved_chunks: Supporting facts retrieved from database.
            confidence_score: Average retrieval similarity score.
            cited_files: File names referenced in the output.
            cited_pages: Page numbers referenced in the output.

        Returns:
            Tuple of (is_valid: bool, validated_answer_text: str, validated_files: List[str], validated_pages: List[int]).
        """
        refusal_msg = "I could not find the answer in the uploaded documents."
        
        # 1. Confidence threshold check
        if confidence_score < self.confidence_threshold:
            logger.warning(
                f"Confidence score {confidence_score:.4f} falls below threshold "
                f"{self.confidence_threshold:.4f}. Triggering refusal."
            )
            return False, refusal_msg, [], []

        # If LLM itself returns a refusal phrasing, align it to standard refusal
        lower_answer = answer_text.lower()
        if "could not find the answer" in lower_answer or "not mention" in lower_answer or "no context" in lower_answer:
            logger.info("LLM reported facts not found. Rewriting to standard refusal.")
            return True, refusal_msg, [], []

        # 2. Hallucinated citations prevention
        # Build set of valid sources from the retrieved chunks
        valid_files = set()
        valid_pages_by_file = {}
        
        for item in retrieved_chunks:
            doc = item.get("document")
            if doc:
                f_name = doc.metadata.get("file_name")
                p_num = doc.metadata.get("page_number")
                if f_name:
                    valid_files.add(f_name)
                    if f_name not in valid_pages_by_file:
                        valid_pages_by_file[f_name] = set()
                    if p_num is not None:
                        valid_pages_by_file[f_name].add(p_num)

        # Verify cited files exist in actual context
        validated_files = []
        for file in cited_files:
            if file in valid_files:
                validated_files.append(file)
            else:
                logger.warning(f"Hallucinated file citation detected: '{file}'. Excluding.")

        # Verify cited pages exist in actual context
        validated_pages = []
        for page in cited_pages:
            # Check if this page was actually retrieved in any file
            page_found = False
            for f_name, page_set in valid_pages_by_file.items():
                if page in page_set:
                    page_found = True
                    break
            
            if page_found:
                validated_pages.append(page)
            else:
                logger.warning(f"Hallucinated page citation detected: page {page}. Excluding.")

        # If all citations are hallucinated, the answer is suspect; fallback to refusal
        if cited_files and not validated_files:
            logger.error("All cited files were hallucinated. Invalidating answer.")
            return False, refusal_msg, [], []

        logger.info("Generated answer successfully passed output guardrails.")
        return True, answer_text, validated_files, sorted(list(set(validated_pages)))
