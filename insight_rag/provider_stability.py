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
    original_normalise = SemanticRouter._normalise
    original_compact_validate = CompactRouterOutput.model_validate
    original_chat_structured = LLMProvider.chat_structured

    def validate_rich(parsed: dict):
        return original_validate_rich(_normalise_provider_payload(parsed))

    def compact_model_validate(cls, obj, *args, **kwargs):
        return original_compact_validate(_normalise_provider_payload(obj), *args, **kwargs)

    def normalise(self, output, *, key_to_id, selected_set, documents, latency_ms):
        """Resolve harmless bad document-key guesses when scope is unambiguous.

        DeepSeek can occasionally return a filename-like token or generic label in
        ``document_keys`` instead of one of the manifest keys (D1, D2, ...).  The
        core router correctly treats unknown keys as unsafe when several PDFs are
        eligible.  But when exactly one selected/eligible PDF exists, asking
        "summarize my PDF" is not ambiguous: the UI selection is the stronger,
        deterministic scope constraint.  Clear only the invalid model key in that
        one-document case and let the core normalizer bind the sole eligible PDF.

        Inventory count/list requests are also authoritative metadata operations;
        they do not need a model-selected content key at all.
        """
        all_ids = [str(doc.get("document_id")) for doc in documents if doc.get("document_id")]
        eligible_ids = [did for did in all_ids if not selected_set or did in selected_set]
        requested_keys = [str(key).upper() for key in (getattr(output, "document_keys", None) or [])]
        invalid_keys = [key for key in requested_keys if key not in key_to_id]

        inventory_request = (
            getattr(output, "scope", None) == "DOCUMENT"
            and getattr(output, "task", None) == "DOCUMENT_METADATA"
            and getattr(output, "metadata_kind", None) in {"INVENTORY_COUNT", "INVENTORY_LIST"}
        )
        single_document_request = (
            getattr(output, "scope", None) == "DOCUMENT"
            and len(eligible_ids) == 1
            and bool(invalid_keys)
        )

        if inventory_request or single_document_request:
            output = output.model_copy(update={"document_keys": []})

        return original_normalise(
            self,
            output,
            key_to_id=key_to_id,
            selected_set=selected_set,
            documents=documents,
            latency_ms=latency_ms,
        )

    def messages(question: str, *, manifest: str, history_text: str) -> list[dict]:
        routed = original_messages(question, manifest=manifest, history_text=history_text)
        if routed and routed[0].get("role") == "system":
            routed[0]["content"] += (
                "\n\nJSON routing rules:\n"
                "- scope is the broad domain and task is the operation. PDF count uses "
                "scope=DOCUMENT and task=DOCUMENT_METADATA; never use DOCUMENT_METADATA as scope.\n"
                "- For array fields such as document_keys, target_pages and table_operations, use [] "
                "when there are no values. Never invent page numbers or document keys.\n"
                "- document_keys may contain only exact manifest keys such as D1/D2. If exactly one "
                "selected PDF exists and the user says 'my PDF', 'the PDF' or equivalent wording, "
                "use that selected manifest key rather than a filename or generic word like PDF.\n"
                "Semantic examples:\n"
                "- A whole-document request like explaining, teaching or summarizing the PDF is "
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
    SemanticRouter._normalise = normalise
    CompactRouterOutput.model_validate = classmethod(compact_model_validate)
    LLMProvider.chat_structured = chat_structured
    _PATCHED = True
