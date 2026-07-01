"""Theme helpers for the Streamlit frontend."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st


def apply_theme() -> None:
    """Apply enterprise healthcare styling.

    Returns:
        None.
    """
    st.markdown(
        dedent(
            """
            <style>
            :root {
                --hcip-blue: #1455a0;
                --hcip-blue-dark: #0f3d73;
                --hcip-gray-50: #f8fafc;
                --hcip-gray-100: #eef2f7;
                --hcip-gray-600: #475569;
                --hcip-border: #d7dee8;
                --hcip-success: #047857;
                --hcip-warning: #b45309;
                --hcip-danger: #b91c1c;
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
                max-width: 1680px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            html {
                font-size: 17px;
            }
            body, [data-testid="stAppViewContainer"] {
                color: #0f172a;
            }
            .hcip-header {
                border-bottom: 1px solid var(--hcip-border);
                padding-bottom: 0.85rem;
                margin-bottom: 1.25rem;
            }
            .hcip-kicker {
                color: var(--hcip-blue);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0;
                text-transform: uppercase;
                margin-bottom: 0.25rem;
            }
            .hcip-title {
                color: #0f172a;
                font-size: 2rem;
                font-weight: 750;
                margin: 0;
                letter-spacing: 0;
            }
            .hcip-subtitle {
                color: var(--hcip-gray-600);
                font-size: 1rem;
                margin-top: 0.35rem;
            }
            .hcip-card {
                background: white;
                border: 1px solid var(--hcip-border);
                border-radius: 8px;
                padding: 1rem;
                min-height: 100%;
            }
            .hcip-metric {
                background: white;
                border: 1px solid var(--hcip-border);
                border-radius: 8px;
                padding: 1rem;
            }
            .hcip-metric-label {
                color: var(--hcip-gray-600);
                font-size: 0.82rem;
                font-weight: 650;
            }
            .hcip-metric-value {
                color: #0f172a;
                font-size: 1.8rem;
                font-weight: 760;
            }
            .hcip-badge {
                display: inline-block;
                border-radius: 999px;
                color: white;
                font-size: 0.78rem;
                font-weight: 700;
                padding: 0.18rem 0.55rem;
            }
            .hcip-muted {
                color: var(--hcip-gray-600);
            }
            .hcip-footer {
                color: var(--hcip-gray-600);
                border-top: 1px solid var(--hcip-border);
                margin-top: 2rem;
                padding-top: 1rem;
                font-size: 0.82rem;
            }
            div[data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid var(--hcip-border);
            }
            div[data-testid="stSidebar"] * {
                font-size: 1rem;
            }
            .stButton > button,
            .stDownloadButton > button {
                font-size: 1rem;
                padding: 0.55rem 1rem;
            }
            .stTextInput input,
            .stTextArea textarea,
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect div[data-baseweb="select"] > div {
                font-size: 1rem;
            }
            .stMarkdown p,
            .stMarkdown li,
            .stCaption,
            .stAlert,
            .stDataFrame,
            .stTable {
                font-size: 1rem;
            }
            @media (min-width: 1200px) {
                .stApp {
                    zoom: 1.05;
                }
            }
            @media (max-width: 1024px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .stApp {
                    zoom: 1;
                }
            }
            </style>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
