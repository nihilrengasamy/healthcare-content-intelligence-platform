"""Rule validation service."""


class RuleValidator:
    """Validates extracted rules before execution."""

    def validate_rule(self, rule: dict) -> dict:
        """Validate one rule."""
        # TODO: Check schema, operators, fields, and citation support.
        pass

    def validate_rules(self, rules: list[dict]) -> list[dict]:
        """Validate multiple rules."""
        # TODO: Validate extracted rule set.
        pass

