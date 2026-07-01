"""Prompt templates for summarization, RAG, extraction, and explanations."""


class PromptTemplates:
    """Stores prompt template accessors for LLM workflows."""

    def get_policy_qa_prompt(self) -> str:
        """Return the policy Q&A prompt template."""
        # TODO: Define grounded Q&A prompt.
        pass

    def get_rule_extraction_prompt(self) -> str:
        """Return the rule extraction prompt template."""
        # TODO: Define structured rule extraction prompt.
        pass

    def get_explanation_prompt(self) -> str:
        """Return the explanation prompt template."""
        # TODO: Define claim decision explanation prompt.
        pass

