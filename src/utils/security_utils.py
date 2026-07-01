"""Security utility helpers for safe document and prompt handling."""


class SecurityUtils:
    """Provides security-oriented helpers for uploaded content and prompts."""

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize an uploaded filename."""
        # TODO: Remove unsafe path characters and normalize file name.
        pass

    def redact_sensitive_text(self, text: str) -> str:
        """Redact sensitive text from content."""
        # TODO: Redact PHI/PII-like patterns for demo safety.
        pass

