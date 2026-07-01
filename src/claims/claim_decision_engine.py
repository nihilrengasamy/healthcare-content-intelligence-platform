"""Claim decision engine combining rule and ML outputs."""


class ClaimDecisionEngine:
    """Combines deterministic rule outcomes and ML predictions into decisions."""

    def make_decision(self, claim: dict, rule_result: dict, model_result: dict) -> dict:
        """Make a final claim decision."""
        # TODO: Combine rules, model score, thresholds, and policy context.
        pass

    def combine_rule_and_model_outputs(self, rule_result: dict, model_result: dict) -> dict:
        """Combine rule engine and ML model outputs."""
        # TODO: Normalize signals into a single decision payload.
        pass

