import re
from observability.logger import logger

class TextCleaner:
    """
        Utility class for cleaning extracted document text before chunking.

        Responsibilities:
            - Normalize whitespace
            - Remove page numbers
            - Remove repeated headers/footers
            - Fix hyphenated line breaks
    """
    
    def clean(self, text: str) -> str:
        """Run the complete cleaning pipeline."""
        
        logger.info("Cleaning extracted text.")

        self._validate_input(text)

        text = self._normalize_whitespace(text)
        text = self._remove_headers_footers(text)
        text = self._remove_page_numbers(text)
        text = self._fix_hyphenation(text)
        text = self._remove_extra_blank_lines(text)
        
        logger.info("Text cleaning completed.")

        return text.strip()
    
    def _validate_input(self, text: str)-> None:

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if not text.strip():
            raise ValueError("Input text is empty.")
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving line breaks.
        """

        text = text.replace("\t", " ")

        text = re.sub(r"[ ]{2,}", " ", text)

        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        return text
    
    def _remove_page_numbers(self, text: str) -> str:
        """
        Remove standalone page number lines.
        """

        return re.sub(
            r"^\s*page\s+\d+\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        
        
    def _fix_hyphenation(self, text: str) -> str:
        """
        Join words split across lines with hyphens.
        """

        return re.sub(
            r"(\w+)-\r?\n(\w+)",
            r"\1\2",
            text,
        )
    
    def _remove_extra_blank_lines(self, text: str) -> str:
        """
        Remove excessive blank lines.
        """

        return re.sub(
            r"(\r?\n){3,}",
            "\n\n",
            text,
        )
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Placeholder for header/footer removal.
        """

        return text
    
    