"""Centralized Streamlit session state management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st

from utils.constants import SESSION_KEYS


class SessionManager:
    """Initializes and manages frontend session state keys."""

    DEFAULTS: dict[str, Any] = {
        "uploaded_documents": [],
        "document_classification": None,
        "summary": None,
        "comparison": None,
        "embeddings": None,
        "vector_store": None,
        "rag_response": None,
        "rules": [],
        "features": [],
        "rule_results": [],
        "ml_prediction": None,
        "claim_decision": None,
        "explanation": None,
        "evaluation": None,
        "analytics": {},
    }
    @classmethod
    def initialize(cls) -> None:
        """Initialize all required session state keys.

        Returns:
            None.
        """
        for key in SESSION_KEYS:
            if key not in st.session_state:
                st.session_state[key] = deepcopy(cls.DEFAULTS.get(key))

    @classmethod
    def get_status_summary(cls) -> dict[str, str]:
        """Return a friendly status summary for session data.

        Returns:
            Mapping of session keys to status labels.
        """
        cls.initialize()
        summary: dict[str, str] = {}
        for key in SESSION_KEYS:
            value = st.session_state.get(key)
            if value in (None, [], {}):
                summary[key] = "Waiting"
            else:
                summary[key] = "Ready"
        return summary

    @classmethod
    def reset(cls) -> None:
        """Reset frontend workflow state.

        Returns:
            None.
        """
        for key, value in cls.DEFAULTS.items():
            st.session_state[key] = deepcopy(value)
