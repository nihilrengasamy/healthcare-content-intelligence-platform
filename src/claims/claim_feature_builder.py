"""Claim feature builder for rules and ML scoring."""


class ClaimFeatureBuilder:
    """Builds rule-ready and model-ready claim feature payloads."""

    def build_rule_features(self, claim: dict) -> dict:
        """Build features for rule evaluation."""
        # TODO: Normalize claim attributes for rule engine.
        pass

    def build_model_features(self, claim: dict) -> object:
        """Build features for ML prediction."""
        # TODO: Convert claim into model feature vector.
        pass

