"""Constants used across the Healthcare Content Intelligence Platform."""


class DocumentTypes:
    """Supported healthcare content document types."""

    BILLING_CODING_POLICY = "billing_coding_policy"
    CLINICAL_PRACTICE_GUIDELINE = "clinical_practice_guideline"
    PAYER_PROVIDER_CONTRACT = "payer_provider_contract"
    COVERAGE_POLICY = "coverage_policy"
    REGULATORY_DOCUMENT = "regulatory_document"


class DecisionOutcomes:
    """Supported claim decision outcomes."""

    APPROVE = "approve"
    DENY = "deny"
    FLAG_FOR_REVIEW = "flag_for_review"


# TODO: Add embedding, chunking, prompt, rule, and model constants.

