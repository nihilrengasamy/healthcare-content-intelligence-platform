# Contributing

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local Checks

Run syntax checks before submitting changes:

```powershell
python -m compileall -q app.py modules pages components utils config
python -m pytest
```

## Engineering Guidelines

- Preserve the existing module boundaries and healthcare AI workflow.
- Do not log API keys, PHI, raw claims, or sensitive clinical notes.
- Mock OpenAI and other external calls in tests.
- Keep deterministic modules free of LLM and network dependencies.
- Add tests for new behavior and regression fixes.
- Prefer small, focused changes over broad refactors.

## Review Checklist

- The Streamlit app launches with `streamlit run app.py`.
- New configuration belongs in `.env.example` or `config/settings.py`.
- Generated files, local indexes, models, and logs stay out of version control.
- Business logic changes are documented in the README or module docs.
