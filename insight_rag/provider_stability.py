"""Provider compatibility hardening for the Insight PDF-RAG runtime.

The cloud provider may occasionally emit benign JSON representations that are
semantically clear but do not exactly match ARIA's canonical Pydantic contract.
This module normalizes only those representations before validation; it does not
weaken retrieval, grounding, citation checks, or document scope.
"""

from __future__ import annotations

from llm_provider import LLMProvider

from .semantic_router import CompactRouterOutput, SemanticRouter

_PATCHED = False

_SCOPE_ALIASES = {
    "DOCUMENT_QA": "DOCUMENT",
    "DOCUMENT_SUMMARY": "DOCUMENT",
    "DOCUMENT_SEARCH": "DOCUMENT",
    "DOCUMENT_METADATA": "DOCUMENT",
    "DOCUMENT_COMPARE": "DOCUMENT",
    "CHAT": "CONVERSATION",
    "PREVIOUS_ANSWER": "CONVERSATION",
    "SYSTEM_INFO": "SYSTEM",
    "CLARIFY": "CLARIFICATION",
}


def _normalise_provider_payload(payload):
    if not isinstance(payload, dict):
        return payload
    candidate = dict(payload)
    for field in ("document_keys", "target_pages", "table_operations"):
        if candidate.get(field) is None:
            candidate[field] = []
    raw_scope = str(candidate.get("scope") or "").upper()
    if raw_scope in _SCOPE_ALIASES:
        candidate["scope"] = _SCOPE_ALIASES[raw_scope]
    return candidate


def apply_provider_stability_patches() -> None:
    """Apply narrow cloud-output compatibility fixes once per Python process."""
    global _PATCHED
    if _PATCHED:
        return

    original_validate_rich = SemanticRouter._validate_rich
    original_messages = SemanticRouter._messages
    original_compact_validate = CompactRouterOutput.model_validate
    original_chat_structured = LLMProvider.chat_structured

    def validate_rich(parsed: dict):
        return original_validate_rich(_normalise_provider_payload(parsed))

    def compact_model_validate(cls, obj, *args, **kwargs):
        return original_compact_validate(_normalise_provider_payload(obj), *args, **kwargs)

    def messages(question: str, *, manifest: str, history_text: str) -> list[dict]:
        routed = original_messages(question, manifest=manifest, history_text=history_text)
        if routed and routed[0].get("role") == "system":
            routed[0]["content"] += (
                "\n\nJSON routing rules:\n"
                "- scope is the broad domain and task is the operation. PDF count uses "
                "scope=DOCUMENT and task=DOCUMENT_METADATA; never use DOCUMENT_METADATA as scope.\n"
                "- For array fields such as document_keys, target_pages and table_operations, use [] "
                "when there are no values. Never invent page numbers or document keys.\n"
                "Semantic examples:\n"
                "- A whole-document request like explaining or teaching the PDF is "
                "DOCUMENT/DOCUMENT_SUMMARY with broad_query=true.\n"
                "- If a previous assistant answer exists, a request to explain/simplify/rephrase that answer "
                "without new facts is CONVERSATION/PREVIOUS_ANSWER with previous_action=TRANSFORM.\n"
                "Classify equivalent wording by meaning, not by exact phrase matching."
            )
        return routed

    def chat_structured(
        self,
        role,
        messages,
        *,
        json_schema,
        schema_name="structured_response",
        temperature=0.0,
        num_predict=900,
        timeout=None,
        reasoning_effort="low",
    ):
        if role == "rag_scope":
            num_predict = min(int(num_predict), 450)
        return original_chat_structured(
            self,
            role,
            messages,
            json_schema=json_schema,
            schema_name=schema_name,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )

    SemanticRouter._validate_rich = staticmethod(validate_rich)
    SemanticRouter._messages = staticmethod(messages)
    CompactRouterOutput.model_validate = classmethod(compact_model_validate)
    LLMProvider.chat_structured = chat_structured
    _PATCHED = True
