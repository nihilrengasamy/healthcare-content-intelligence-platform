"""Rule extraction service for healthcare policies and contracts."""


class RuleExtractor:
    """Extracts structured rules from unstructured healthcare content."""

    def extract_rules(self, document_id: str) -> list[dict]:
        """Extract rules from a document."""
        # TODO: Use GPT to extract structured rules with citations.
        pass

    def normalize_rules(self, rules: list[dict]) -> list[dict]:
        """Normalize extracted rules into the internal schema."""
        # TODO: Map raw LLM output to rule schema.
        pass

