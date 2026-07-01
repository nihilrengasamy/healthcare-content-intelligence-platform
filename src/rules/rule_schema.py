"""Rule schema definitions for extracted healthcare policy rules."""


class RuleSchema:
    """Defines the canonical structure for executable healthcare rules."""

    def validate_shape(self, rule: dict) -> bool:
        """Validate basic rule structure."""
        # TODO: Validate required fields and supported operators.
        pass

