import fitz
import pdfplumber
from pathlib import Path
from utils.models import ParsedPage
from utils.exceptions import FileValidationError
from utils.exceptions import PDFParserError
from typing import List
from PIL import ImageOps
import pytesseract
from pdf2image import convert_from_path
from utils.logger import logger

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

class PDFParser:
    """
    Handles extraction of text and metadata from PDF documents.
    Uses PyMuPDF as the primary parser and pdfplumber as a fallback.
    """
    def _create_parsed_page(
        self,
        text: str,
        page_number: int,
        pdf_path: Path,
        parser_name: str,
        width: float,
        height: float,
    ):
        return ParsedPage(
            page_number=page_number,
            text=text,
            metadata={
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "parser": parser_name,
                "page": page_number,
                "width": width,
                "height": height,
            },
        )
    
    def parse(self, file_path: str) -> tuple[str, dict]:
        """
        Parse a PDF using the best available parser.

        Strategy:
            1. Validate input
            2. Try PyMuPDF
            3. Fall back to pdfplumber
            4. Raise error if both fail
        """
        pages = None
        parser_used = None
        pymupdf_error = "No exception raised."
        pdfplumber_error = "No exception raised."
        ocr_error = "No exception raised."
        
        pdf_path = self._validate_file(file_path)

        try:
            pages = self._parse_with_pymupdf(pdf_path)
            parser_used = "pymupdf"
        except Exception as e:
            pymupdf_error = str(e)
            logger.warning(f"PyMuPDF failed: {e}")
            
        if not pages:
            
            try:
                pages = self._parse_with_pdfplumber(pdf_path)
                parser_used = "pdfplumber"
            except Exception as e:
                pdfplumber_error = str(e)
                logger.warning(f"pdfplumber failed: {e}")
                
            if not pages:       
                try:
                    pages = self._parse_with_ocr(pdf_path)

                    if pages:

                        raw_text = self._merge_pages(pages)

                        metadata = self._create_metadata(
                            pdf_path,
                            len(pages),
                            "ocr"
                        )

                        return raw_text, metadata
                except Exception as e:

                    ocr_error = str(e)

                    logger.warning(f"OCR failed: {e}")
                
        if not pages:

            raise PDFParserError(
            f"""
            Failed to parse PDF.

            PyMuPDF:
            {pymupdf_error}

            pdfplumber:
            {pdfplumber_error}

            OCR:
            {ocr_error}
            """
            )
        
        raw_text = self._merge_pages(pages)
        metadata = self._create_metadata(
            pdf_path,
            len(pages),
            parser_used,
        )

        return raw_text, metadata

    def _validate_file(self, file_path: str)-> Path:
        """Validate the input PDF."""
        
        if not isinstance(file_path, str):
            raise FileValidationError("File path must be a string.")

        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileValidationError(
                f"File does not exist: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise FileValidationError(
                f"{pdf_path} is not a valid file."
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise FileValidationError(
                f"Unsupported file type: {pdf_path.suffix}"
            )

        if pdf_path.stat().st_size == 0:
            raise FileValidationError(
                "PDF file is empty."
            )

        return pdf_path

    def _parse_with_pymupdf(self,pdf_path: Path) -> List[ParsedPage]:
        """
        Parse a PDF using PyMuPDF.

        Returns:
            List[ParsedPage]
        """
        doc = fitz.open(pdf_path)
        try:
            if doc.is_encrypted:
                raise PDFParserError(
                    "Encrypted or password protected PDF."
                )
            
            pages = []
            for page_number, page in enumerate(doc):
                text = page.get_text("text")
                
                text = text.strip()

                if not text:
                    continue
                
                parsed_page = self._create_parsed_page(
                    text=text,
                    page_number=page_number + 1,
                    pdf_path=pdf_path,
                    parser_name="pymupdf",
                    width=page.rect.width,
                    height=page.rect.height,
                )
                pages.append(parsed_page)
        finally:   
            doc.close()
        return pages
            
        
        

    def _parse_with_pdfplumber(self, pdf_path: Path)-> List[ParsedPage]:
        """Fallback extraction using pdfplumber."""
        with pdfplumber.open(pdf_path) as pdf:
            pages=[]
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if not text:
                    continue

                text = text.strip()

                if not text:
                    continue
                
                parsed_page = self._create_parsed_page(
                    text=text,
                    page_number=page_number + 1,
                    pdf_path=pdf_path,
                    parser_name="pdfplumber",
                    width=page.width,
                    height=page.height,
                )

                pages.append(parsed_page)
        return pages        
                

    def _parse_with_ocr(self, pdf_path: Path)-> List[ParsedPage]:
        """
        OCR fallback for scanned PDFs.
        """

        images = convert_from_path(
            pdf_path,
            dpi=300
        )

        pages = []

        for page_number, image in enumerate(images):

            gray = ImageOps.grayscale(image)

            text = pytesseract.image_to_string(gray)

            text = text.strip()

            if not text:
                continue

            parsed_page = self._create_parsed_page(
                text=text,
                page_number=page_number + 1,
                pdf_path=pdf_path,
                parser_name="ocr",
                width=image.width,
                height=image.height,
            )

            pages.append(parsed_page)

        return pages
        

    def _merge_pages(self, pages: List[ParsedPage]) -> str:
        """Merge extracted page content into a single string."""
        merged=[]
        for page in pages:
            merged.extend([
                    f"========== Page {page.page_number} ==========" ,
                    page.text,
                    ""   
                    ])
        merged_text = "\n".join(merged)
        return merged_text.strip()

    def _create_metadata(self,pdf_path: Path, page_count: int, parser: str,)-> dict:
        """Generate document-level metadata."""
        return {
            "filename": pdf_path.name,
            "source": str(pdf_path),
            "document_type": "pdf",
            "page_count": page_count,
            "parser": parser,
        }