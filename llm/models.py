from dataclasses import dataclass

@dataclass
class ParsedPage:
    """
    Represents the extracted content of a single PDF page.
    """

    page_number: int
    text: str
    parser: str