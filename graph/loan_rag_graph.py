"""
Module compiling the LangGraph loan RAG agent state machine.
Defines execution steps, edges, safety conditional routing, and compiles the graph.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graph.state import LoanRAGState
from graph.nodes import LoanRAGNodes

logger = logging.getLogger(__name__)


# 1. Routing Function for Conditional Edges
def route_after_guardrail(state: LoanRAGState) -> str:
    """
    Routes the execution path depending on whether the guardrail flagged the prompt.
    """
    if not state.get("is_safe", True):
        logger.warning("[Graph Router] Query blocked. Routing execution directly to END.")
        return "blocked"
    logger.info("[Graph Router] Prompt is safe. Routing to retriever node.")
    return "proceed"


class LoanRAGGraph:
    """
    Builder class for the LangGraph StateGraph workflow.
    """

    def __init__(self):
        self.nodes = LoanRAGNodes()
        self._assemble_graph()

    def _assemble_graph(self):
        """
        Builds nodes and hooks up edges in the StateGraph workflow.
        """
        # Initialize graph with our TypedState schema
        workflow = StateGraph(LoanRAGState)

        # 1. Register Nodes
        workflow.add_node("guardrail", self.nodes.guardrail_node)
        workflow.add_node("retriever", self.nodes.retriever_node)
        workflow.add_node("reranker", self.nodes.reranker_node)
        workflow.add_node("generator", self.nodes.llm_generation_node)
        workflow.add_node("citation", self.nodes.citation_node)

        # 2. Set Entrance Point
        workflow.set_entry_point("guardrail")

        # 3. Add Conditional Routing Edge
        workflow.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {
                "blocked": END,
                "proceed": "retriever"
            }
        )

        # 4. Add Linear Edges
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "generator")
        workflow.add_edge("generator", "citation")
        workflow.add_edge("citation", END)

        # Compile Runnable
        self.app = workflow.compile()
        logger.info("LangGraph loan RAG StateGraph compiled successfully.")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the compiled graph with a starting query.

        Parameters:
            query: The user prompt.

        Returns:
            The final State dictionary after execution.
        """
        initial_state: LoanRAGState = {
            "query": query,
            "documents": [],
            "contexts_str": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "evidence": [],
            "provider_used": "",
            "tokens_used": 0,
            "error": None,
            "is_safe": True
        }
        
        logger.info(f"Invoking compiled LangGraph with prompt: '{query}'")
        final_state = self.app.invoke(initial_state)
        return final_state
