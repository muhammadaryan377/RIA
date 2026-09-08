"""Bounded retry for transient broad-summary verification false negatives.

A broad PDF summary is a multi-stage cloud operation (route -> generate -> verify).
Even with deterministic prompts, a hosted model can occasionally reject one run
and accept the same fully grounded request immediately afterwards.  This patch
retries only that narrow failure mode once, before the visible turn is persisted.
It never converts a failed verification into success: the second run must pass
all normal routing, retrieval, citation-integrity and grounding checks itself.
"""

from __future__ import annotations

from .diagnostics import record
from .scope_layer import InsightPDFRAG as RoutedInsightPDFRAG

_PATCHED = False


def apply_summary_retry_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_chat = RoutedInsightPDFRAG._chat

    def chat_with_summary_retry(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        first = original_chat(
            self,
            question,
            conversation_id=conversation_id,
            document_ids=document_ids,
        )

        routing = first.get("routing") or {}
        is_broad_summary = (
            first.get("scope") == "DOCUMENT"
            and routing.get("task") == "DOCUMENT_SUMMARY"
            and first.get("intent") == "document_query"
        )
        transient_verify_failure = first.get("evidence_status") == "verification_failed"

        if not (is_broad_summary and transient_verify_failure):
            return first

        record("summary_retry", "attempted")
        second = original_chat(
            self,
            question,
            conversation_id=conversation_id,
            document_ids=document_ids,
        )
        if second.get("evidence_status") == "supported" and second.get("sources"):
            record("summary_retry", "recovered")
            second.setdefault("grounding", {})["summary_retry"] = "recovered"
            return second

        # Keep the original fail-closed result. The retry is availability hardening,
        # not a path for bypassing the verifier.
        record("summary_retry", "failed")
        first.setdefault("grounding", {})["summary_retry"] = "failed"
        return first

    RoutedInsightPDFRAG._chat = chat_with_summary_retry
    _PATCHED = True
