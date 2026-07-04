from pathlib import Path

from observability.logger import logger
from config import SUPPORTED_FILE_TYPES
from context.indexers.pdf_parser import PDFParser
from context.indexers.docx_parser import DOCXParser
from context.indexers.image_parser import ImageParser

from exceptions import FileValidationError


class FileRouter:
    """
    Routes documents to the appropriate parser
    based on file extension.
    """

    def route(
        self,
        file_path: str,
    ) -> tuple[str, dict]:
        """
        Route a document to its parser.
        """

        path = self._validate_file(file_path)

        parser = self._get_parser(path)

        logger.info(
            f"Routing '{path.name}' to {parser.__class__.__name__}."
        )
        
        raw_text, metadata = parser.parse(str(path))
        logger.info(
            f"Successfully parsed '{path.name}' using {parser.__class__.__name__}."
        )
        return raw_text, metadata

    def _validate_file(
        self,
        file_path: str,
    ) -> Path:
        """
        Validate the input file before routing.
        """
        logger.info(f"Received document: {file_path}")
        if not isinstance(file_path, str):
            raise FileValidationError(
                "File path must be a string."
            )

        path = Path(file_path)

        if not path.exists():
            raise FileValidationError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise FileValidationError(
                f"{path} is not a valid file."
            )

        if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            raise FileValidationError(
                f"Unsupported file type: {path.suffix}"
            )

        if path.stat().st_size == 0:
            raise FileValidationError(
                "Input file is empty."
            )

        return path

    def _get_parser(
        self,
        path: Path,
    )-> PDFParser | DOCXParser | ImageParser:
        """
        Return the appropriate parser for the file type.
        """

        extension = path.suffix.lower()

        PARSERS = {
            ".pdf": PDFParser,
            ".docx": DOCXParser,
            ".png": ImageParser,
            ".jpg": ImageParser,
            ".jpeg": ImageParser,
            ".tiff": ImageParser,
        }
        parser_class = PARSERS.get(extension)

        if parser_class is None:
            raise FileValidationError(...)

        return parser_class()