"""Source attribution service for policy-backed outputs."""


class SourceAttribution:
    """Links answers, rules, features, and decisions back to source content."""

    def attribute_sources(self, output: dict, citations: list[dict]) -> dict:
        """Attach source attribution to an output."""
        # TODO: Map generated claims to cited document chunks.
        pass

