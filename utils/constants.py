"""Constants for the Streamlit frontend."""

from __future__ import annotations


SESSION_KEYS = [
    "uploaded_documents",
    "document_classification",
    "summary",
    "comparison",
    "embeddings",
    "vector_store",
    "rag_response",
    "rules",
    "features",
    "rule_results",
    "ml_prediction",
    "claim_decision",
    "explanation",
    "evaluation",
    "analytics",
]


PLACEHOLDER_STATUS = {
    "Ready": "#1f7a4d",
    "Waiting": "#6b7280",
    "Processing": "#1d4ed8",
    "Completed": "#047857",
    "Failed": "#b91c1c",
}


FEATURE_CARDS = [
    ("Document Intelligence", "Upload, classify, summarize, and compare healthcare content."),
    ("Policy Q&A", "Retrieve relevant policy sections and prepare grounded answers."),
    ("Rules & Features", "Extract structured policy rules and model-ready features."),
    ("Decision Support", "Combine rules, ML signals, decisions, explanations, and evaluation."),
]

