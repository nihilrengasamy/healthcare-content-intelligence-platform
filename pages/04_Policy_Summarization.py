"""Policy summarization page."""

from __future__ import annotations

import html
import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from utils.session_manager import SessionManager
from utils.theme import apply_theme

load_dotenv()


def _all_documents(payloads: list[dict[str, Any]]) -> list[Any]:
    """Return all page documents from uploaded payloads."""
    documents: list[Any] = []
    for payload in payloads:
        documents.extend(payload.get("documents", []))
    return documents


def _payload_text(payload: dict[str, Any]) -> str:
    """Return combined text from an uploaded document payload."""
    documents = payload.get("documents", [])
    return "\n".join(
        getattr(document, "page_content", "")
        for document in documents
        if getattr(document, "page_content", "")
    )


def _combined_documents(payloads: list[dict[str, Any]]) -> list[Any]:
    """Return one LangChain document per uploaded PDF for cleaner Groq summaries."""
    from langchain_core.documents import Document

    documents: list[Any] = []
    for payload in payloads:
        text = _payload_text(payload)
        metadata = payload.get("metadata", {})
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "filename": payload.get("filename", "uploaded_policy.pdf"),
                    "source": payload.get("source_path", ""),
                    "pages": metadata.get("pages", 0),
                    "document_level_summary": True,
                },
            )
        )
    return documents


def _summary_errors(summaries: list[dict[str, Any]]) -> list[str]:
    """Return Groq errors captured in summary payloads."""
    errors: list[str] = []
    for item in summaries:
        payload = item.get("summary", {})
        if isinstance(payload, dict) and payload.get("_error"):
            errors.append(str(payload["_error"]))
    return errors


def _has_summary_content(summary: dict[str, Any]) -> bool:
    """Return whether a structured summary contains useful content."""
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _normalize_text(value: Any) -> str:
    """Return cleaned display text for summary fields."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()

    return " ".join(text.split())


def _normalize_list(values: Any) -> list[str]:
    """Return cleaned bullet items for list-like summary fields."""
    if not isinstance(values, list):
        values = [values] if values else []

    cleaned_items: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _normalize_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_items.append(text)
    return cleaned_items


def _polish_bullet_text(text: str) -> str:
    """Lightly polish extracted bullet text for analyst-facing display."""
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""

    replacements = {
        "This synthetic policy describes ": "",
        "This synthetic policy ": "",
        "Covered Service ": "",
        "Medical Necessity ": "",
        "Routine repeat imaging ": "Repeat imaging ",
        "Claims must include ": "Include ",
    }
    for source, target in replacements.items():
        if cleaned.startswith(source):
            cleaned = target + cleaned[len(source) :]
            break

    return cleaned[0].upper() + cleaned[1:] if cleaned else ""


def _is_authorization_like(text: str) -> bool:
    """Return whether text looks like authorization rather than eligibility."""
    lowered = text.casefold()
    markers = [
        "prior authorization",
        "authorization is required",
        "authorization required",
        "outpatient",
        "emergency department",
        "inpatient services",
    ]
    return any(marker in lowered for marker in markers)


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into display-friendly sentences."""
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]


def _render_bullet_section(title: str, values: Any, empty_text: str = "Not specified.") -> None:
    """Render a summary section as clean bullet points."""
    st.markdown(f"#### {title}")
    items = [_polish_bullet_text(item) for item in _normalize_list(values)]
    items = [item for item in items if item]
    if not items:
        st.caption(empty_text)
        return

    st.markdown("\n".join(f"- {item}" for item in items))


def _render_text_section(title: str, value: Any, empty_text: str = "Not specified.") -> None:
    """Render a summary section as clean paragraph text."""
    st.markdown(f"#### {title}")
    text = _normalize_text(value)
    if not text:
        st.caption(empty_text)
        return
    st.write(text)


def _join_examples(items: list[str], limit: int = 2) -> str:
    """Join a short list of items into readable prose."""
    selected = items[:limit]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return f"{selected[0]} and {selected[1]}"


def _build_display_executive_summary(summary: dict[str, Any]) -> str:
    """Build a display-ready executive summary when the stored field is blank."""
    executive_summary = _normalize_text(summary.get("executive_summary", ""))
    if executive_summary:
        return executive_summary

    purpose = _normalize_text(summary.get("purpose", ""))
    coverage = _normalize_list(summary.get("covered_services", []))
    medical_necessity = _normalize_list(summary.get("medical_necessity", []))
    exclusions = _normalize_list(summary.get("excluded_services", []))
    authorization = _normalize_text(summary.get("prior_authorization", ""))

    sentences: list[str] = []
    if purpose:
        sentences.append(purpose.rstrip(".") + ".")
    if coverage:
        sentences.append(f"Coverage focuses on {_join_examples(coverage)}.")
    if medical_necessity:
        sentences.append(
            f"Medical necessity criteria emphasize {_join_examples(medical_necessity)}."
        )
    if authorization:
        sentences.append(f"Authorization guidance indicates {authorization.rstrip('.')}.")
    if exclusions:
        sentences.append(f"Key exclusions include {_join_examples(exclusions)}.")

    return " ".join(sentences).strip()


def _get_display_eligibility(summary: dict[str, Any]) -> list[str]:
    """Return eligibility items that are distinct from authorization rules."""
    eligibility_items = [
        _polish_bullet_text(item)
        for item in _normalize_list(summary.get("eligibility_criteria", []))
    ]
    eligibility_items = [item for item in eligibility_items if item]
    return [item for item in eligibility_items if not _is_authorization_like(item)]


st.set_page_config(page_title="Policy Summarization", page_icon="SUM", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Policy Summarization")
render_header(
    "Policy Summarization",
    "Generate structured summaries for healthcare policy content.",
)

payloads = st.session_state.get("uploaded_documents", [])
show_prompt = False
if not payloads:
    st.warning("Upload and extract PDFs before summarization.")
else:
    try:
        from modules.prompt_manager import PromptManager

        prompt = PromptManager().get_prompt("summarization")
        show_prompt = st.toggle("Developer Mode", value=False, help="Show the underlying prompt template used for summarization.")
        if show_prompt:
            with st.expander("Prompt Template", expanded=False):
                st.code(prompt if isinstance(prompt, str) else str(prompt))
    except Exception as error:
        st.info(f"Prompt manager unavailable: {error}")

    entered_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Used only for this running Streamlit session. Do not paste keys into screenshots.",
    )
    if entered_api_key:
        os.environ["GROQ_API_KEY"] = entered_api_key

    groq_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    groq_model_default = os.getenv("GROQ_MODEL", groq_models[0])
    groq_model_index = groq_models.index(groq_model_default) if groq_model_default in groq_models else 0
    model_name = st.selectbox(
        "Groq Model",
        groq_models,
        index=groq_model_index,
        help="Use llama-3.1-8b-instant for a fast demo, or choose a larger Groq model if available.",
    )

    has_api_key = bool(os.getenv("GROQ_API_KEY"))
    if not has_api_key:
        st.warning(
            "GROQ_API_KEY is required for Groq summarization. Enter your key above "
            "or add it to your local .env file, then click Generate Groq Summary."
        )

    if st.button("Generate Groq Summary", type="primary", disabled=not has_api_key):
        try:
            from modules.summarizer import HealthcareSummarizer

            with st.spinner("Generating clean structured summaries with Groq"):
                summaries = HealthcareSummarizer(
                    model_name=model_name,
                    max_workers=1,
                ).summarize_documents(_combined_documents(payloads))

            errors = _summary_errors(summaries)
            has_content = any(
                _has_summary_content(item.get("summary", {}))
                for item in summaries
                if isinstance(item.get("summary", {}), dict)
            )
            st.session_state["summary"] = summaries

            if errors:
                st.error("Groq summarization failed. Check your API key, billing, model access, or network.")
                with st.expander("Groq error details", expanded=True):
                    for error in errors:
                        st.code(error)
            elif not has_content:
                st.error("Groq returned an empty summary. Try another model or regenerate.")
            else:
                st.success("Groq summary generated.")
        except Exception as error:
            st.error(f"Groq summarization failed: {error}")

summary = st.session_state.get("summary")
if summary:
    render_metric_row(
        [
            {"title": "Summaries", "value": str(len(summary)), "delta": "Generated", "icon": "SUM", "color": "#1455a0"},
            {"title": "Source Pages", "value": str(len(_all_documents(payloads))), "delta": "Processed", "icon": "PAGE", "color": "#047857"},
            {"title": "Status", "value": "Ready", "delta": "Stored", "icon": "OK", "color": "#7c3aed"},
        ]
    )
    selected_index = st.selectbox("Select summary", list(range(len(summary))), format_func=lambda index: summary[index].get("document", f"Summary {index + 1}"))
    selected_summary = summary[selected_index].get("summary", {})
    executive_summary = _build_display_executive_summary(selected_summary)
    executive_sentences = _split_sentences(executive_summary)[:5]
    purpose = _normalize_text(selected_summary.get("purpose", ""))
    eligibility_items = _get_display_eligibility(selected_summary)

    st.markdown("### Executive Summary")
    if executive_sentences:
        summary_body = "".join(
            f"<p style='margin: 0 0 0.65rem 0; line-height: 1.7;'>{html.escape(sentence)}</p>"
            for sentence in executive_sentences
        )
        st.markdown(
            f"""
            <div class="hcip-card" style="padding: 1rem 1.2rem; margin-bottom: 1rem;">
                {summary_body}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Executive summary is not available for this document yet.")

    if purpose:
        st.caption(purpose)

    col1, col2 = st.columns(2)
    with col1:
        _render_bullet_section("Coverage", selected_summary.get("covered_services", []))
        _render_bullet_section("Medical Necessity", selected_summary.get("medical_necessity", []))
        if eligibility_items:
            _render_bullet_section("Eligibility", eligibility_items)
    with col2:
        _render_bullet_section("Exclusions", selected_summary.get("excluded_services", []))
        _render_text_section("Authorization", selected_summary.get("prior_authorization", ""))
        _render_bullet_section("Coding", selected_summary.get("coding_requirements", []))
        _render_bullet_section("Dates", selected_summary.get("key_dates", []))

    if _normalize_list(selected_summary.get("important_changes", [])):
        _render_bullet_section("Important Changes", selected_summary.get("important_changes", []))

    if selected_summary.get("_error"):
        st.error("This summary contains an error diagnostic instead of Groq content.")
        st.code(selected_summary["_error"])
    if show_prompt:
        render_json_viewer(summary, "Summary JSON", expanded=False)
else:
    st.info("Summary output will appear here.")

render_footer()
