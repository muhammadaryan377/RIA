"""Groq compatibility hardening for the Insight PDF-RAG runtime.

Provider quirks are handled here without weakening ARIA's semantic policy.  Live
Groq runs have shown three harmless-but-disruptive representations:

* optional array fields can be emitted as ``null`` instead of ``[]``;
* a model can put a valid *task* value such as ``DOCUMENT_METADATA`` in the
  ``scope`` field even though the intended scope is ``DOCUMENT``;
* token-per-minute 429 responses can request a wait longer than five seconds.

The compatibility layer accepts those representations, normalises them into
ARIA's canonical contract, and keeps retries bounded.  It does not add a
keyword/regex intent classifier and it does not weaken grounding/citation checks.
"""

from __future__ import annotations

import logging
import re
import time

from llm_provider import LLMProvider

from .semantic_router import CompactRouterOutput, SemanticRouter


logger = logging.getLogger(__name__)
_PATCHED = False
_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)

# Some structured-output models occasionally copy the task label into scope.
# These aliases are semantic equivalents, not new application scopes.
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


def _nullable_array(schema: dict, field: str) -> None:
    """Allow a structured-output array field to be represented as JSON null."""
    prop = schema.get("properties", {}).get(field)
    if not isinstance(prop, dict):
        return
    current = prop.get("type")
    if current == "array":
        prop["type"] = ["array", "null"]
    elif isinstance(current, list) and "array" in current and "null" not in current:
        prop["type"] = [*current, "null"]


def _allow_scope_aliases(schema: dict) -> None:
    """Let constrained decoding emit known task-as-scope aliases for normalisation."""
    prop = schema.get("properties", {}).get("scope")
    if not isinstance(prop, dict):
        return
    values = list(prop.get("enum") or [])
    for alias in _SCOPE_ALIASES:
        if alias not in values:
            values.append(alias)
    prop["enum"] = values


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


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read Groq's retry hint from headers, then fall back to the error text."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                value = float(raw)
                if value > 0:
                    return value
        except (TypeError, ValueError):
            pass

    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        try:
            value = float(match.group(1))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return None


def apply_groq_stability_patches() -> None:
    """Apply narrow provider-compatibility fixes once per Python process."""
    global _PATCHED
    if _PATCHED:
        return

    original_strict_schema = SemanticRouter._strict_json_schema
    original_compact_schema = SemanticRouter._compact_json_schema
    original_validate_rich = SemanticRouter._validate_rich
    original_messages = SemanticRouter._messages
    original_compact_validate = CompactRouterOutput.model_validate
    original_chat_structured = LLMProvider.chat_structured

    def strict_schema() -> dict:
        schema = original_strict_schema()
        for field in ("document_keys", "target_pages", "table_operations"):
            _nullable_array(schema, field)
        _allow_scope_aliases(schema)
        return schema

    def compact_schema() -> dict:
        schema = original_compact_schema()
        for field in ("document_keys", "target_pages"):
            _nullable_array(schema, field)
        _allow_scope_aliases(schema)
        return schema

    def validate_rich(parsed: dict):
        return original_validate_rich(_normalise_provider_payload(parsed))

    def compact_model_validate(cls, obj, *args, **kwargs):
        return original_compact_validate(_normalise_provider_payload(obj), *args, **kwargs)

    def messages(question: str, *, manifest: str, history_text: str) -> list[dict]:
        routed = original_messages(question, manifest=manifest, history_text=history_text)
        if routed and routed[0].get("role") == "system":
            routed[0]["content"] += (
                "\n\nStructured-output rules:\n"
                "- scope is the broad domain and task is the operation. For example, PDF count uses "
                "scope=DOCUMENT and task=DOCUMENT_METADATA; never use DOCUMENT_METADATA as scope.\n"
                "- For array fields such as document_keys, target_pages and table_operations, prefer [] "
                "when there are no values. Never invent page numbers or document keys.\n"
                "Semantic examples:\n"
                "- 'explain my PDF', 'teach me this PDF', 'explain all topics in this document' or an "
                "equivalent whole-document explanation is DOCUMENT/DOCUMENT_SUMMARY with broad_query=true.\n"
                "- If a previous assistant answer exists, 'explain it', 'explain that', 'what does that "
                "mean?', 'say it simply' or an equivalent request that needs no new facts is "
                "CONVERSATION/PREVIOUS_ANSWER with previous_action=TRANSFORM.\n"
                "These are semantic examples only; classify equivalent wording by meaning, not exact phrases."
            )
        return routed

    def chat_structured(self, role, messages, *, json_schema, schema_name="structured_response",
                        temperature=0.0, num_predict=900, timeout=None,
                        reasoning_effort="low"):
        # The route JSON is small. A lower output ceiling reduces TPM reservation
        # without changing the schema or model.
        if role == "rag_scope":
            num_predict = min(int(num_predict), 450)
        return original_chat_structured(
            self, role, messages, json_schema=json_schema, schema_name=schema_name,
            temperature=temperature, num_predict=num_predict, timeout=timeout,
            reasoning_effort=reasoning_effort,
        )

    SemanticRouter._strict_json_schema = staticmethod(strict_schema)
    SemanticRouter._compact_json_schema = staticmethod(compact_schema)
    SemanticRouter._validate_rich = staticmethod(validate_rich)
    SemanticRouter._messages = staticmethod(messages)
    CompactRouterOutput.model_validate = classmethod(compact_model_validate)
    LLMProvider.chat_structured = chat_structured

    def cloud_with_retry(self, fn, attempts=4):
        """Retry 429s using Groq's requested delay, with a bounded upper limit."""
        last_exc = None
        attempts = max(1, int(attempts))
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                status = getattr(exc, "status_code", None)
                if status is None and response is not None:
                    status = getattr(response, "status_code", None)

                if status != 429 or attempt >= attempts - 1:
                    raise

                hinted = _retry_after_seconds(exc)
                if hinted is not None:
                    wait = min(max(hinted + 0.5, 1.5), 20.0)
                else:
                    wait = min(1.5 * (2**attempt), 8.0)

                logger.warning(
                    "Rate limit (429); retrying in %.1fs (attempt %d/%d)",
                    wait, attempt + 1, attempts,
                )
                time.sleep(wait)

        raise last_exc  # pragma: no cover

    LLMProvider._cloud_with_retry = cloud_with_retry
    _PATCHED = True
