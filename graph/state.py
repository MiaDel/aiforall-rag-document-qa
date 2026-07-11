"""
State definitions for the LangGraph loan RAG agent.
"""

from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document as LCDocument


class LoanRAGState(TypedDict):
    """
    State tracking dictionary representing variables passed across LangGraph nodes.
    """
    query: str
    documents: List[LCDocument]
    contexts_str: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    evidence: List[str]
    provider_used: str
    tokens_used: int
    error: str | None
    is_safe: bool
