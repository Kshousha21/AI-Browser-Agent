"""Closed, content-free failures; never expose provider or validation bodies."""

CODES = frozenset({
    "rag_invalid_request", "rag_context_budget", "rag_invalid_answer",
    "rag_unknown_source", "rag_invalid_evidence", "rag_unexpected_tool_call",
    "rag_remote_context_denied", "rag_provider_not_allowed", "rag_configuration_error",
    "rag_provider_failed", "rag_provider_timeout", "rag_retrieval_failed",
    "rag_invalid_page", "rag_visual_unavailable", "rag_render_failed",
})


class RagError(RuntimeError):
    def __init__(self, code: str):
        self.code = code if code in CODES else "rag_provider_failed"
        super().__init__(self.code)
