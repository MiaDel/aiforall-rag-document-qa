class PDFParserError(Exception):
    """
    Base exception for all PDF parser related errors.
    """
    pass


class FileValidationError(PDFParserError):
    """
    Raised when the file path or file type is invalid.
    """
    pass


class CorruptPDFError(PDFParserError):
    """
    Raised when a PDF cannot be opened or is corrupted.
    """
    pass


class PasswordProtectedPDFError(PDFParserError):
    """
    Raised when the PDF requires a password.
    """
    pass


class EmptyPDFError(PDFParserError):
    """
    Raised when no readable text is found.
    """
    pass