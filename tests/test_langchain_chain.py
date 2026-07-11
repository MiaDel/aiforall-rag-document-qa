"""
Unit tests for the LangChain LCEL RAG Integration.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document as LCDocument
from chains.rag_chain import LangChainRAGChain


@patch("chains.retrieval_chain.LangChainRAGRetriever._get_relevant_documents")
@patch("llms.llm_router.LLMRouter.generate_answer")
def test_langchain_chain_execution(mock_router, mock_retriever):
    """
    Verifies that the LangChain LCEL sequence compiles, executes, and yields structured outputs.
    """
    # Setup mock retriever to return 1 doc
    mock_retriever.return_value = [
        LCDocument(
            page_content="Interest rates are set by the central bank at 5%.",
            metadata={
                "doc_id": "doc_123",
                "file_name": "terms.pdf",
                "page_number": 1,
                "chunk_id": "doc_123_p1_c0",
                "section_title": "Rates",
                "source": "/workspace/terms.pdf"
            }
        )
    ]

    # Setup mock LLM router response
    mock_router.return_value = {
        "answer": "The interest rate is 5%.",
        "sources": [{"file": "terms.pdf", "page": 1}],
        "confidence": 0.95,
        "evidence": ["Interest rates are set by the central bank at 5."],
        "provider_used": "llama3",
        "tokens_used": 20,
        "chunk_references": []
    }

    # Instantiate and fetch LCEL runnable chain
    rag_chain = LangChainRAGChain()
    chain = rag_chain.get_chain()
    
    # Run query
    response = chain.invoke("What is the interest rate?")
    
    # Assert output schema matches the unified format
    assert isinstance(response, dict)
    assert response["answer"] == "The interest rate is 5%."
    assert response["sources"] == [{"file": "terms.pdf", "page": 1}]
    assert response["confidence"] == 0.95
    assert response["provider_used"] == "llama3"
    
    mock_retriever.assert_called_once()
