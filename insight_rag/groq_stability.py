"""Groq compatibility hardening for the Insight PDF-RAG runtime.

This module keeps provider-specific tolerance in one place instead of weakening
ARIA's semantic routing or grounding rules.  It addresses two behaviours seen
with Groq structured outputs in live runs:

* optional array fields can be emitted as ``null`` even when the semantic value
  is an empty list;
* token-per-minute 429 responses can ask the client to wait longer than the
  previous five-second retry cap.

The patches are intentionally narrow and idempotent. They do not change route
semantics, retrieval, grounding, citation verification, or model selection.
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


def _normalise_array_nulls(payload):
    if not isinstance(payload, dict):
        return payload
    candidate = dict(payload)
    for field in ("document_keys", "target_pages", "table_operations"):
        if candidate.get(field) is None:
            candidate[field] = []
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

    def strict_schema() -> dict:
        schema = original_strict_schema()
        for field in ("document_keys", "target_pages", "table_operations"):
            _nullable_array(schema, field)
        return schema

    def compact_schema() -> dict:
        schema = original_compact_schema()
        for field in ("document_keys", "target_pages"):
            _nullable_array(schema, field)
        return schema

    def validate_rich(parsed: dict):
        return original_validate_rich(_normalise_array_nulls(parsed))

    def compact_model_validate(cls, obj, *args, **kwargs):
        return original_compact_validate(_normalise_array_nulls(obj), *args, **kwargs)

    def messages(question: str, *, manifest: str, history_text: str) -> list[dict]:
        routed = original_messages(question, manifest=manifest, history_text=history_text)
        if routed and routed[0].get("role") == "system":
            routed[0]["content"] += (
                "\nFor array fields such as document_keys, target_pages and table_operations, "
                "prefer [] when there are no values. Never invent page numbers or document keys."
            )
        return routed

    SemanticRouter._strict_json_schema = staticmethod(strict_schema)
    SemanticRouter._compact_json_schema = staticmethod(compact_schema)
    SemanticRouter._validate_rich = staticmethod(validate_rich)
    SemanticRouter._messages = staticmethod(messages)
    CompactRouterOutput.model_validate = classmethod(compact_model_validate)

    def cloud_with_retry(self, fn, attempts=4):
        """Retry 429s using Groq's requested delay, with a bounded upper limit."""
        last_exc = None
        for attempt in range(max(1, int(attempts))):
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
                    # Small safety margin prevents retrying on the exact boundary.
                    wait = min(max(hinted + 0.5, 1.5), 20.0)
                else:
                    # Bounded exponential fallback when no server hint is exposed.
                    wait = min(1.5 * (2**attempt), 8.0)

                logger.warning(
                    "Rate limit (429); retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    attempts,
                )
                time.sleep(wait)

        raise last_exc  # pragma: no cover

    LLMProvider._cloud_with_retry = cloud_with_retry
    _PATCHED = True
