"""Rule engine for evaluating healthcare claims."""


class RuleEngine:
    """Applies extracted and approved rules to claim records."""

    def evaluate_claim(self, claim: dict, rules: list[dict]) -> dict:
        """Evaluate a claim against rules."""
        # TODO: Apply all eligible rules and return evaluation result.
        pass

    def apply_rules(self, claim: dict, rules: list[dict]) -> list[dict]:
        """Apply rules to a claim."""
        # TODO: Return triggered and non-triggered rule results.
        pass

    def return_triggered_rules(self, evaluation: dict) -> list[dict]:
        """Return triggered rules from an evaluation."""
        # TODO: Extract triggered rule details from evaluation result.
        pass

