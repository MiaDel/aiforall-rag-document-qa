"""
Module for routing queries to LLM providers in a prioritized fallback chain:
Llama3 Local -> Gemini -> Groq -> OpenAI.
Incorporates input-output guardrails, multi-document evidence synthesis,
latency logging to logs/rag.log, and connection health checks.
"""

import time
import requests
from typing import List, Dict, Any
from config.settings import settings
from utils.logger import setup_logger
from guardrails.input_guard import InputGuardrail
from guardrails.output_guard import OutputGuardrail
from llms.llama_provider import LlamaProvider
from llms.gemini_provider import GeminiProvider
from llms.groq_provider import GroqProvider
from llms.openai_provider import OpenAIProvider
from llms.reasoning_engine import MultiDocumentReasoningEngine

# central logger setup
logger = setup_logger("llm_router")


# Health checks exposed globally/class level
def check_llama_connection() -> bool:
    """
    Checks if Ollama service is reachable.
    """
    try:
        host = settings.OLLAMA_HOST.rstrip("/")
        # Check simple tags api
        response = requests.get(f"{host}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        logger.warning("Llama3 Local (Ollama) health check failed: unreachable.")
        return False


def check_gemini_connection() -> bool:
    """
    Checks if Gemini key is set.
    """
    # Simple validation that key is provided and length is normal
    key = settings.GEMINI_API_KEY
    is_valid = bool(key and len(key) > 10)
    if not is_valid:
        logger.warning("Gemini health check failed: Missing API Key.")
    return is_valid


def check_grok_connection() -> bool:
    """
    Checks if Groq key is set.
    """
    key = settings.GROQ_API_KEY
    is_valid = bool(key and len(key) > 10)
    if not is_valid:
        logger.warning("Grok/Groq health check failed: Missing API Key.")
    return is_valid


class LLMRouter:
    """
    Manages the RAG answer generation pipeline, priority-based fallback, and guardrails.
    """

    def __init__(self):
        self.llama_provider = LlamaProvider()
        self.gemini_provider = GeminiProvider()
        self.groq_provider = GroqProvider()
        self.openai_provider = OpenAIProvider()
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: str | None = None
    ) -> Dict[str, Any]:
        """
        Hardened generation pipeline integrating guardrails and multi-document reasoning.

        Returns:
            Dict containing the answer, sources, evidence snippets, confidence, and metadata.
        """
        start_pipeline = time.time()
        logger.info(f"Query received: '{query}'")

        # 1. Input Guardrail
        is_safe, refusal_reason = self.input_guard.validate(query)
        if not is_safe:
            logger.warning(f"Query blocked by input guardrail: {refusal_reason}")
            # Log metrics to rag.log
            logger.info(
                f"[RAG Metric] Query: '{query}' | Selected Provider: Guardrail | "
                f"Status: Blocked | Latency: 0.00s | Tokens: 0"
            )
            return {
                "answer": refusal_reason or "Blocked by input guardrails.",
                "sources": [],
                "confidence": 0.0,
                "evidence": [],
                "provider_used": "Input Guardrail",
                "tokens_used": 0,
                "chunk_references": []
            }

        # 2. Extract sources and evidence upfront
        sources = MultiDocumentReasoningEngine.identify_sources(retrieved_chunks)
        evidence = MultiDocumentReasoningEngine.extract_evidence_snippets(retrieved_chunks)
        
        # Calculate retrieval confidence score (average similarity score of retrieved chunks)
        if retrieved_chunks:
            scores = [item.get("score", 0.0) for item in retrieved_chunks]
            confidence_score = float(sum(scores) / len(scores))
        else:
            confidence_score = 0.0

        # 3. Context Validation (Refusal if empty context)
        refusal_msg = "I could not find the answer in the uploaded documents."
        if not retrieved_chunks:
            logger.warning("Empty retrieved contexts. Refusing generation.")
            logger.info(
                f"[RAG Metric] Query: '{query}' | Selected Provider: Context Guardrail | "
                f"Status: Refused | Latency: 0.00s | Tokens: 0"
            )
            return {
                "answer": refusal_msg,
                "sources": [],
                "confidence": 0.0,
                "evidence": [],
                "provider_used": "Context Guardrail",
                "tokens_used": 0,
                "chunk_references": []
            }

        # 4. Compile comparative document context
        contexts = MultiDocumentReasoningEngine.compile_comparative_context(retrieved_chunks)
        
        # Configure prompt guidelines
        sys_prompt = system_prompt or (
            "You are a Senior Loan Officer and Expert GenAI Assistant.\n"
            "Your task is to answer the user's loan document question based ONLY on the provided document fragments.\n"
            "Guidelines:\n"
            "1. Answer the question accurately using facts from the fragments.\n"
            "2. If you cannot answer using the fragments, say exactly: 'I could not find the answer in the uploaded documents.'\n"
            "3. Do not formulate answers from external knowledge. Remain grounded."
        )

        user_prompt = f"Document Contexts:\n{contexts}\n\nQuestion: {query}\nAnswer:"

        # 5. Priority Fallback Chain Execution
        routing_chain = [
            ("groq", self.groq_provider),
            ("openai", self.openai_provider),
            ("llama3", self.llama_provider),
            ("gemini", self.gemini_provider)
        ]

        llm_response = None
        for provider_name, provider_client in routing_chain:
            try:
                logger.info(f"Attempting generation with provider: '{provider_name}'")
                llm_response = provider_client.generate(prompt=user_prompt, system_prompt=sys_prompt)
                
                # Check if provider reported successful execution
                if llm_response.get("success"):
                    break
                else:
                    logger.warning(f"Provider '{provider_name}' returned failure: {llm_response.get('error')}. Fallback triggered.")
            except Exception as e:
                logger.warning(f"Unexpected provider error on '{provider_name}': {str(e)}. Fallback triggered.")
                continue

        # If all providers fail
        if not llm_response or not llm_response.get("success"):
            logger.critical("All LLM providers in fallback chain failed.")
            return {
                "answer": "All available LLM endpoints are currently unreachable. Please try again later.",
                "sources": sources,
                "confidence": confidence_score,
                "evidence": evidence,
                "provider_used": "None (All Failed)",
                "tokens_used": 0,
                "chunk_references": [item["document"].metadata for item in retrieved_chunks if item.get("document")]
            }

        # Extract values from successful provider response
        answer_text = llm_response.get("answer", "")
        provider_used = llm_response.get("provider", "Unknown")
        tokens_used = llm_response.get("tokens_used", 0)

        # 6. Output Guardrail Validation
        # Verify citations match context and checks confidence limits
        cited_files = [s["file"] for s in sources]
        cited_pages = [s["page"] for s in sources]
        
        is_valid_out, final_answer, val_files, val_pages = self.output_guard.validate_answer(
            answer_text=answer_text,
            retrieved_chunks=retrieved_chunks,
            confidence_score=confidence_score,
            cited_files=cited_files,
            cited_pages=cited_pages
        )

        # Adjust final citations based on validation outputs by filtering the original list
        validated_sources = [
            s for s in sources
            if s["file"] in val_files and s["page"] in val_pages
        ]
        
        # Calculate latency
        latency = time.time() - start_pipeline
        logger.info(f"[Timing] LLM response time: {latency:.4f}s")
        print(f"[Timing] LLM response time: {latency:.4f}s")
        logger.info(f"[RAG Metric] Query: '{query}' | Selected Provider: {provider_used} | Latency: {latency:.2f}s | Tokens: {tokens_used}")

        return {
            "answer": final_answer,
            "sources": validated_sources if is_valid_out and final_answer != refusal_msg else [],
            "confidence": confidence_score if final_answer != refusal_msg else 0.0,
            "evidence": evidence if final_answer != refusal_msg else [],
            "provider_used": provider_used,
            "tokens_used": tokens_used,
            "chunk_references": [item["document"].metadata for item in retrieved_chunks if item.get("document")]
        }
