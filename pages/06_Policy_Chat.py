"""Policy chat page with hosted low-memory fallback retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from utils import pandas_compat as pd
import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.table_view import render_table
from utils.runtime_mode import is_low_memory_demo_mode, trim_text_for_demo
from utils.session_manager import SessionManager
from utils.theme import apply_theme

load_dotenv()


def _all_chunks(payloads: list[dict[str, Any]]) -> list[Any]:
    """Return all uploaded document chunks."""
    chunks: list[Any] = []
    for payload in payloads:
        chunks.extend(payload.get("chunks", []))
    return chunks


def _payload_label(payload: dict[str, Any]) -> str:
    """Return a readable label for an uploaded policy payload."""
    return str(payload.get("filename") or Path(str(payload.get("source_path", ""))).name)


def _payload_by_label(payloads: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    """Return the payload matching the selected label."""
    for payload in payloads:
        if _payload_label(payload) == label:
            return payload
    return None


def _ensure_chunks(payload: dict[str, Any]) -> list[Document]:
    """Return stored chunks, or build lightweight chunks on demand."""
    existing_chunks = payload.get("chunks", [])
    if existing_chunks:
        return existing_chunks

    chunk_size = 900
    chunk_overlap = 150
    chunks: list[Document] = []
    for document in payload.get("documents", []):
        text = trim_text_for_demo(str(getattr(document, "page_content", "") or ""), max_chars=2200)
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(getattr(document, "metadata", {}) or {}),
                    )
                )
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)

    payload["chunks"] = chunks
    return chunks


def _retrieved_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build retrieved document DataFrame."""
    rows = []
    for result in results:
        metadata = result.get("metadata", {})
        rows.append(
            {
                "Score": round(float(result.get("score", 0.0)), 3),
                "Source": metadata.get("source", ""),
                "Page": metadata.get("page", metadata.get("page_number", "")),
                "Text": result.get("text", "")[:300],
            }
        )
    return pd.DataFrame(rows)


def _tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight retrieval."""
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
        if len(token) > 2
    }


def _lightweight_retrieve(question: str, chunks: list[Document], top_k: int = 4) -> list[dict[str, Any]]:
    """Retrieve likely matches with token overlap instead of embeddings."""
    question_tokens = _tokenize(question)
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(getattr(chunk, "page_content", "") or "").strip()
        if not text:
            continue

        overlap = len(question_tokens & _tokenize(text))
        if overlap <= 0:
            continue

        results.append(
            {
                "text": text,
                "score": overlap / max(len(question_tokens), 1),
                "metadata": dict(getattr(chunk, "metadata", {}) or {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _extractive_answer(retrieved_documents: list[dict[str, Any]]) -> str:
    """Build a concise answer from the best retrieved snippets."""
    if not retrieved_documents:
        return "The selected policy does not contain enough matching text to answer this question."

    snippets: list[str] = []
    for result in retrieved_documents[:3]:
        sentences = re.split(r"(?<=[.!?])\s+", result.get("text", ""))
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                snippets.append(cleaned)
                break

    return " ".join(snippets) if snippets else "The selected policy does not contain enough matching text to answer this question."


def _lightweight_response(question: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a hosted-demo answer payload without embeddings or FAISS."""
    chunks = _ensure_chunks(payload)
    retrieved_documents = _lightweight_retrieve(question, chunks)
    citations = [
        {
            "source": item.get("metadata", {}).get("source", payload.get("filename", "")),
            "page": item.get("metadata", {}).get("page", item.get("metadata", {}).get("page_number", "")),
        }
        for item in retrieved_documents
    ]
    top_score = float(retrieved_documents[0]["score"]) if retrieved_documents else 0.2
    return {
        "question": question,
        "answer": _extractive_answer(retrieved_documents),
        "confidence": round(min(max(top_score, 0.2), 0.95), 2),
        "citations": citations,
        "retrieved_documents": retrieved_documents,
        "mode": "lightweight_demo_retrieval",
    }


st.set_page_config(page_title="Policy Chat", page_icon="CHAT", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Policy Chat")
render_header(
    "Policy Chat",
    "Create a semantic index and ask source-grounded questions against uploaded healthcare content.",
)

payloads = st.session_state.get("uploaded_documents", [])
chunks = _all_chunks(payloads)
low_memory_demo = is_low_memory_demo_mode()

if not chunks and not payloads:
    st.warning("Upload and extract PDFs before creating a policy chat index.")
else:
    payload_labels = [_payload_label(payload) for payload in payloads]
    selected_label = st.selectbox(
        "Active policy document",
        payload_labels,
        help="Policy Chat answers only from this selected uploaded file, regardless of filename.",
    )
    selected_payload = _payload_by_label(payloads, selected_label)
    selected_chunks = _ensure_chunks(selected_payload) if selected_payload else []
    selected_source_path = str(selected_payload.get("source_path", "")) if selected_payload else ""

    if st.session_state.get("policy_chat_source") != selected_source_path:
        st.session_state["vector_store"] = None
        st.session_state["embeddings"] = None
        st.session_state["rag_response"] = None
        st.session_state["policy_chat_source"] = selected_source_path

    st.caption(
        f"Scoped to `{selected_label}` only. "
        "The selected file is treated as the active policy for this chat session."
    )
    if low_memory_demo:
        st.info(
            "Hosted demo mode is active here. The app uses lightweight retrieval for the selected file only, "
            "so the cloud version stays responsive without downloading a large embedding model."
        )

    col1, col2 = st.columns([0.35, 0.65])
    with col1:
        if st.button("Build or Reuse Vector Index", type="primary"):
            if low_memory_demo:
                st.session_state["embeddings"] = [
                    {
                        "text": chunk.page_content[:400],
                        "embedding": [],
                        "metadata": dict(chunk.metadata),
                    }
                    for chunk in selected_chunks
                ]
                st.session_state["vector_store"] = {
                    "mode": "lightweight_demo_retrieval",
                    "chunks": len(selected_chunks),
                    "embedding_dimension": 0,
                    "model": "token-overlap",
                }
                st.success("Lightweight hosted retrieval index prepared.")
            elif st.session_state.get("vector_store") and st.session_state.get("embeddings"):
                st.info("Existing embeddings and vector store reused.")
            else:
                try:
                    from modules.embeddings import HealthcareEmbeddingGenerator
                    from modules.vector_store import HealthcareVectorStore

                    with st.spinner("Generating embeddings and creating FAISS index. First run may take a minute while the embedding model downloads."):
                        embedder = HealthcareEmbeddingGenerator()
                        embeddings = embedder.generate_embeddings(selected_chunks)
                        if not embeddings:
                            st.error(
                                "Embedding generation returned no vectors. "
                                "Verify that sentence-transformers is installed and that the embedding model can download successfully."
                            )
                            st.stop()
                        prepared = embedder.prepare_for_vector_store(selected_chunks, embeddings)
                        vector_store = HealthcareVectorStore(embedding_model=embedder.model)
                        index = vector_store.create_index(prepared)
                    if index is None:
                        st.error(
                            "FAISS index creation failed after embeddings were generated. "
                            "Check the server logs for the exact vector store error."
                        )
                    else:
                        st.session_state["embeddings"] = embeddings
                        st.session_state["vector_store"] = vector_store
                        st.success("Vector index created.")
                except Exception as error:
                    st.error(f"Vector index creation failed: {error}")
    with col2:
        st.info(
            "Only the selected policy is indexed in hosted demo mode."
            if low_memory_demo
            else "Embeddings are generated only for the selected active policy document."
        )

vector_store = st.session_state.get("vector_store")
if vector_store:
    stats = vector_store if isinstance(vector_store, dict) else vector_store.get_index_statistics()
    render_metric_row(
        [
            {"title": "Chunks Indexed", "value": str(stats.get("chunks", 0)), "delta": "Lightweight" if low_memory_demo else "FAISS", "icon": "IDX", "color": "#1455a0"},
            {"title": "Embedding Dim", "value": str(stats.get("embedding_dimension", 0)), "delta": "Demo" if low_memory_demo else "Vectors", "icon": "VEC", "color": "#047857"},
            {"title": "Model", "value": str(stats.get("model", "")), "delta": "Retriever" if low_memory_demo else "Embedding", "icon": "MOD", "color": "#7c3aed"},
        ]
    )
    question = st.text_input("Ask a policy question", placeholder="When is lumbar MRI covered?")
    if st.button("Ask Question") and question:
        try:
            with st.spinner("Retrieving sources and generating answer"):
                if low_memory_demo:
                    response = _lightweight_response(question, selected_payload or {})
                else:
                    import importlib
                    import modules.rag as rag_module

                    rag_module = importlib.reload(rag_module)
                    HealthcareRAGAssistant = rag_module.HealthcareRAGAssistant

                    try:
                        assistant = HealthcareRAGAssistant(
                            vector_store=vector_store,
                            preferred_sources=[selected_source_path, selected_label],
                        )
                    except TypeError:
                        assistant = HealthcareRAGAssistant(vector_store=vector_store)
                        assistant.preferred_sources = [selected_source_path, selected_label]
                    response = assistant.answer_with_sources(question)
            st.session_state["rag_response"] = response
            st.success("Question answered.")
        except Exception as error:
            st.error(f"Policy chat failed: {error}")

response = st.session_state.get("rag_response")
if response:
    st.markdown("#### Answer")
    st.write(response.get("answer", ""))
    st.metric("Confidence", f"{response.get('confidence', 0):.2f}")
    retrieved = response.get("retrieved_documents", [])
    if retrieved:
        render_table(_retrieved_frame(retrieved), title="Retrieved Chunks", search=True)
    render_json_viewer(response, "RAG Response JSON", expanded=False)
else:
    st.info("Build an index and ask a question to see RAG results.")

render_footer()
