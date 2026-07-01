"""Retrieval-Augmented Generation for healthcare policy intelligence.

This module uses an existing ``HealthcareVectorStore`` to retrieve relevant
healthcare document chunks and an OpenAI chat model to answer policy questions
with source citations. It does not regenerate embeddings or recreate FAISS
indexes.

Example:
    ```python
    from modules.rag import HealthcareRAGAssistant

    assistant = HealthcareRAGAssistant()

    response = assistant.answer_with_sources("When is lumbar MRI covered?")

    print(response)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class HealthcareRAGAssistant:
    """Answers healthcare policy questions using retrieval-augmented generation."""

    def __init__(
        self,
        vector_store: Any | None = None,
        llm: Any | None = None,
        vector_store_path: str | Path | None = None,
        preferred_sources: list[str] | None = None,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        max_retries: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the RAG assistant.

        Args:
            vector_store: Existing ``HealthcareVectorStore`` instance.
            llm: Optional LangChain-compatible chat model.
            vector_store_path: Optional path to a saved vector store.
            preferred_sources: Optional source-path or filename hints used to
                prioritize retrieval from the active policy document.
            llm_model: OpenAI model name.
            temperature: LLM temperature.
            request_timeout: LLM request timeout in seconds.
            max_retries: Number of retries for failed LLM calls.
            logger: Optional logger instance.

        Returns:
            None.

        Raises:
            ValueError: If ``max_retries`` is negative.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")

        self.vector_store = vector_store
        self.llm = llm
        self.vector_store_path = Path(vector_store_path) if vector_store_path else None
        self.preferred_sources = [source for source in (preferred_sources or []) if source]
        self.llm_model = llm_model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        self.queries_answered = 0
        self.total_latency_seconds = 0.0
        self.total_retrieved_documents = 0

    def initialize_llm(self) -> Any | None:
        """Initialize a chat model for healthcare RAG answers.

        Args:
            None.

        Returns:
            LangChain-compatible chat model when available; otherwise ``None``.

        Raises:
            This method logs configuration or initialization failures and
            returns ``None`` instead of raising.
        """
        if self.llm is not None:
            return self.llm

        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            groq_api_key = os.getenv("GROQ_API_KEY")

            if openai_api_key:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    api_key=openai_api_key,
                )
                self.logger.info("LLM initialized with OpenAI: %s", self.llm_model)
                return self.llm

            if groq_api_key:
                from langchain_groq import ChatGroq

                groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                self.llm = ChatGroq(
                    model=groq_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    api_key=groq_api_key,
                )
                self.logger.info("LLM initialized with Groq: %s", groq_model)
                return self.llm

            self.logger.error("Neither OPENAI_API_KEY nor GROQ_API_KEY is configured.")
            return None
        except Exception as error:
            self.logger.error("Failed to initialize LLM: %s", error)
            return None

    def load_vector_store(self) -> Any | None:
        """Load or return the existing healthcare vector store.

        Args:
            None.

        Returns:
            Loaded ``HealthcareVectorStore`` instance when available; otherwise
            ``None``.

        Raises:
            This method logs loading failures and returns ``None`` instead of
            raising.
        """
        if self.vector_store is not None:
            return self.vector_store

        if self.vector_store_path is None:
            self.logger.error("Vector store path is not configured.")
            return None

        try:
            from modules.vector_store import HealthcareVectorStore

            vector_store = HealthcareVectorStore()
            if not vector_store.load_index(self.vector_store_path):
                self.logger.error("Failed to load vector store: %s", self.vector_store_path)
                return None
            self.vector_store = vector_store
            self.logger.info("Vector store loaded: %s", self.vector_store_path)
            return self.vector_store
        except Exception as error:
            self.logger.error("Vector store loading failed: %s", error)
            return None

    def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant healthcare document chunks from the vector store.

        Args:
            query: Natural language healthcare policy question.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            Retrieved document chunks with text, scores, and metadata.
        """
        self.logger.info("Question received for retrieval.")
        if not isinstance(query, str) or not query.strip():
            self.logger.warning("Invalid query provided for retrieval.")
            return []

        vector_store = self.load_vector_store()
        if vector_store is None:
            self.logger.error("No vector store available for retrieval.")
            return []

        try:
            candidate_count = max(top_k * 3, 10)
            documents = vector_store.similarity_search_with_scores(query, k=candidate_count)
            documents = self._rank_documents(query, documents)
            documents = documents[:top_k]
            self.logger.info("Documents retrieved: %s", len(documents))
            self._log_retrieved_pages(documents)
            return documents
        except Exception as error:
            self.logger.error("Document retrieval failed: %s", error)
            return []

    def build_context(self, retrieved_documents: list[dict[str, Any]]) -> str:
        """Build a citation-preserving context string from retrieved chunks.

        Args:
            retrieved_documents: Retrieved chunks from the vector store.

        Returns:
            Context string containing source, page, and chunk text.
        """
        if not retrieved_documents:
            self.logger.warning("No retrieved documents available for context.")
            return ""

        seen: set[tuple[str, int | str, str]] = set()
        context_blocks: list[str] = []
        for index, document in enumerate(retrieved_documents, start=1):
            text = str(document.get("text", "")).strip()
            metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
            if not text:
                continue

            source = str(metadata.get("source", "Unknown source"))
            page = metadata.get("page", metadata.get("page_number", "Unknown page"))
            dedupe_key = (source, page, text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            context_blocks.append(
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {text}"
            )

        return "\n\n".join(context_blocks)

    def generate_answer(self, query: str) -> str:
        """Generate a RAG answer for a healthcare policy question.

        Args:
            query: Natural language healthcare policy question.

        Returns:
            Concise answer grounded in retrieved healthcare documents.
        """
        response = self.answer_with_sources(query)
        return str(response.get("answer", ""))

    def answer_with_sources(self, query: str) -> dict[str, Any]:
        """Answer a question and include source citations.

        Args:
            query: Natural language healthcare policy question.

        Returns:
            JSON-compatible answer payload containing question, answer,
            confidence, citations, and retrieved documents.
        """
        start_time = time.perf_counter()
        retrieved_documents = self.retrieve_documents(query)
        if not retrieved_documents:
            response = self._unavailable_response(query, [])
            self._record_usage(start_time, 0)
            return response

        context = self.build_context(retrieved_documents)
        if not context:
            response = self._unavailable_response(query, retrieved_documents)
            self._record_usage(start_time, len(retrieved_documents))
            return response

        llm = self.initialize_llm()
        if llm is None:
            response = self._fallback_response(query, retrieved_documents)
            self._record_usage(start_time, len(retrieved_documents))
            return response

        answer_text = self._call_llm(query, context, llm)
        if not answer_text:
            response = self._fallback_response(query, retrieved_documents)
            self._record_usage(start_time, len(retrieved_documents))
            return response

        citations = self._build_citations(retrieved_documents)
        evaluation = self.evaluate_response(
            {
                "question": query,
                "answer": answer_text,
                "citations": citations,
                "retrieved_documents": retrieved_documents,
            }
        )
        response = {
            "question": query,
            "answer": answer_text,
            "confidence": evaluation["confidence"],
            "citations": citations,
            "retrieved_documents": retrieved_documents,
        }
        self._record_usage(start_time, len(retrieved_documents))
        return response

    def answer_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """Answer multiple healthcare policy questions.

        Args:
            queries: List of natural language healthcare questions.

        Returns:
            List of answer payloads.
        """
        if not isinstance(queries, list):
            self.logger.warning("Batch queries must be provided as a list.")
            return []

        return [self.answer_with_sources(query) for query in queries]

    def evaluate_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Evaluate grounding, confidence, citation quality, and hallucination risk.

        Args:
            response: RAG answer payload.

        Returns:
            Evaluation dictionary with grounded status, confidence, and
            hallucination risk.
        """
        if not isinstance(response, dict):
            return {
                "grounded": False,
                "confidence": 0.0,
                "hallucination_risk": "High",
            }

        answer = str(response.get("answer", "")).strip()
        citations = response.get("citations", [])
        retrieved_documents = response.get("retrieved_documents", [])
        unavailable = "unavailable" in answer.lower() or "not present" in answer.lower()
        grounded = bool(answer and citations and retrieved_documents)

        if unavailable and not citations:
            return {
                "grounded": True,
                "confidence": 0.75,
                "hallucination_risk": "Low",
            }

        if grounded:
            average_score = self._average_retrieval_score(retrieved_documents)
            confidence = min(0.99, max(0.5, average_score if average_score else 0.85))
            return {
                "grounded": True,
                "confidence": round(confidence, 2),
                "hallucination_risk": "Low",
            }

        return {
            "grounded": False,
            "confidence": 0.25 if answer else 0.0,
            "hallucination_risk": "High",
        }

    def get_usage_statistics(self) -> dict[str, int | str]:
        """Return RAG assistant usage statistics.

        Args:
            None.

        Returns:
            Dictionary containing answered query count, average latency, average
            retrieved documents, and LLM model.
        """
        average_latency = (
            self.total_latency_seconds / self.queries_answered
            if self.queries_answered
            else 0.0
        )
        average_retrieved = (
            self.total_retrieved_documents / self.queries_answered
            if self.queries_answered
            else 0.0
        )
        return {
            "queries_answered": self.queries_answered,
            "average_latency": f"{average_latency:.1f} sec",
            "average_retrieved_documents": round(average_retrieved),
            "llm_model": self.llm_model,
        }

    def _call_llm(self, query: str, context: str, llm: Any) -> str:
        """Call the configured LLM with retrieved context.

        Args:
            query: User question.
            context: Retrieved context.
            llm: LangChain-compatible chat model.

        Returns:
            LLM answer text, or an empty string on failure.
        """
        messages = self._build_messages(query, context)
        for attempt in range(1, self.max_retries + 2):
            request_start = time.perf_counter()
            try:
                self.logger.info("Submitting LLM request. Attempt: %s", attempt)
                response = llm.invoke(messages)
                latency = time.perf_counter() - request_start
                self.logger.info("LLM latency: %.2f seconds", latency)
                return self._extract_response_content(response)
            except (RuntimeError, TimeoutError, ValueError) as error:
                self.logger.error("LLM request failed: %s", error)
        return ""

    def _build_messages(self, query: str, context: str) -> list[SystemMessage | HumanMessage]:
        """Build healthcare-specific RAG prompt messages.

        Args:
            query: User question.
            context: Retrieved context.

        Returns:
            LangChain system and human messages.
        """
        system_prompt = (
            "You are an expert healthcare policy analyst. "
            "Answer questions ONLY using the retrieved healthcare documents. "
            "Do not invent information. "
            "If the answer is not present in the retrieved documents, state "
            "that the information is unavailable. "
            "Lead with the direct policy answer first. "
            "Prefer coverage criteria, medical necessity, authorization, coding, "
            "or exclusion language only when it directly answers the question. "
            "Ignore unrelated chunks from other sections if they do not answer the question. "
            "If retrieved chunks conflict, prioritize the active policy document and the most directly relevant chunk. "
            "Always cite the source document and page number. "
            "Be concise, accurate, and explain your reasoning."
        )
        human_prompt = (
            "Retrieved healthcare documents:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Return a concise answer grounded only in the retrieved documents. "
            "Start with the exact answer to the question in 2-4 sentences, then cite the supporting source and page."
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from an LLM response.

        Args:
            response: LangChain response or response-like object.

        Returns:
            Text response content.

        Raises:
            ValueError: If no usable content exists.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text content.")
        return content.strip()

    def _build_citations(self, retrieved_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build unique source citations from retrieved documents.

        Args:
            retrieved_documents: Retrieved vector store results.

        Returns:
            Unique citation dictionaries containing source and page.
        """
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, Any]] = set()
        for document in retrieved_documents:
            metadata = document.get("metadata", {})
            source = metadata.get("source", "")
            page = metadata.get("page", metadata.get("page_number", ""))
            key = (str(source), page)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"source": source, "page": page})
        return citations

    def _fallback_response(
        self,
        query: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a grounded fallback response when the LLM is unavailable.

        Args:
            query: User question.
            retrieved_documents: Retrieved documents.

        Returns:
            JSON-compatible answer payload.
        """
        citations = self._build_citations(retrieved_documents)
        if not retrieved_documents:
            return self._unavailable_response(query, [])

        top_text = str(retrieved_documents[0].get("text", "")).strip()
        answer = (
            "Based on the most relevant retrieved policy section, "
            f"{top_text}"
        )
        evaluation = self.evaluate_response(
            {
                "question": query,
                "answer": answer,
                "citations": citations,
                "retrieved_documents": retrieved_documents,
            }
        )
        return {
            "question": query,
            "answer": answer,
            "confidence": evaluation["confidence"],
            "citations": citations,
            "retrieved_documents": retrieved_documents,
        }

    def _unavailable_response(
        self,
        query: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a response for unavailable information.

        Args:
            query: User question.
            retrieved_documents: Retrieved documents, usually empty.

        Returns:
            JSON-compatible unavailable answer payload.
        """
        return {
            "question": query,
            "answer": "The requested information is unavailable in the retrieved documents.",
            "confidence": 0.0,
            "citations": [],
            "retrieved_documents": retrieved_documents,
        }

    def _average_retrieval_score(self, retrieved_documents: list[dict[str, Any]]) -> float:
        """Calculate average retrieval score.

        Args:
            retrieved_documents: Retrieved vector store results.

        Returns:
            Average score, or zero if scores are unavailable.
        """
        scores = [
            float(document.get("score", 0.0))
            for document in retrieved_documents
            if isinstance(document.get("score", 0.0), (int, float))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _rank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank retrieved chunks toward the active document and question intent.

        Args:
            query: User policy question.
            documents: Retrieved vector-store results.

        Returns:
            Re-ranked result list with the strongest chunks first.
        """
        query_terms = {term for term in self._tokenize(query) if len(term) > 2}
        ranked: list[tuple[float, dict[str, Any]]] = []
        for document in documents:
            metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
            text = str(document.get("text", ""))
            source = str(metadata.get("source", ""))
            base_score = float(document.get("score", 0.0))
            score = base_score
            is_preferred_source = self._source_matches_preferred(source)

            lowered_text = text.lower()
            lowered_source = source.lower()
            score += 0.08 * sum(1 for term in query_terms if term in lowered_text)

            if is_preferred_source:
                score += 0.35

            if any(term in query.lower() for term in ["covered", "coverage", "eligible"]):
                if "covered" in lowered_text or "coverage" in lowered_text:
                    score += 0.2
                if "excluded" in lowered_text or "not covered" in lowered_text:
                    score -= 0.1

            if "prior authorization" in query.lower() and "prior authorization" in lowered_text:
                score += 0.25

            if any(term in query.lower() for term in ["cpt", "code", "coding", "icd", "hcpcs"]):
                if any(token in lowered_text for token in ["cpt", "icd", "hcpcs", "code"]):
                    score += 0.2

            if not is_preferred_source and any(
                marker in lowered_source for marker in ["old", "historical", "comparison"]
            ):
                score -= 0.35

            ranked.append((score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked]

    def _source_matches_preferred(self, source: str) -> bool:
        """Return whether a source matches the currently preferred policy file."""
        lowered_source = source.lower()
        return any(preferred.lower() in lowered_source for preferred in self.preferred_sources)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize query text with lightweight normalization."""
        cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
        return [token for token in cleaned.split() if token]

    def _record_usage(self, start_time: float, retrieved_count: int) -> None:
        """Record latency and retrieval statistics.

        Args:
            start_time: Query start time from ``time.perf_counter``.
            retrieved_count: Number of retrieved documents.

        Returns:
            None.
        """
        latency = time.perf_counter() - start_time
        self.queries_answered += 1
        self.total_latency_seconds += latency
        self.total_retrieved_documents += retrieved_count

    def _log_retrieved_pages(self, documents: list[dict[str, Any]]) -> None:
        """Log retrieved source/page pairs without sensitive content.

        Args:
            documents: Retrieved vector store documents.

        Returns:
            None.
        """
        pages = [
            {
                "source": document.get("metadata", {}).get("source", ""),
                "page": document.get("metadata", {}).get(
                    "page",
                    document.get("metadata", {}).get("page_number", ""),
                ),
            }
            for document in documents
        ]
        self.logger.info("Retrieved pages: %s", json.dumps(pages))
