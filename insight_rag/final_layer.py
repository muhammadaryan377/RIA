"""Final response-quality layer for ARIA Insight PDF-RAG.

The lower layers already handle intent routing, context planning, adaptive
retrieval, evidence gates, conversation actions, and graceful failures.  This
last layer enforces a clean user-facing contract: only citations actually used
in the answer are shown, fabricated source labels are removed, and grounded
answers expose compact diagnostics for evaluation without leaking internals to
normal prose.
"""

from __future__ import annotations

import re

from .conversation_layer import InsightPDFRAG as ConversationalInsightPDFRAG


_CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


class InsightPDFRAG(ConversationalInsightPDFRAG):
    """Production export for ARIA's Insight Agent PDF capability."""

    @staticmethod
    def _finalize_sources(answer: str, sources: list[dict]) -> tuple[str, list[dict], str]:
        """Return citation-safe answer and only the sources the answer references.

        LLMs can occasionally emit a source label that was not in the supplied
        context.  Never surface such a label as if it were real evidence.
        When the answer contains valid labels, hide unrelated retrieved chunks so
        the UI stays concise.  If the model omits labels entirely, retain at most
        four retrieved sources for transparency rather than inventing a citation.
        """
        if not sources:
            cleaned = _CITATION_RE.sub("", answer or "").strip()
            return cleaned, [], "not_applicable"

        by_id = {
            str(source.get("source_id") or "").upper(): source
            for source in sources
            if source.get("source_id")
        }
        cited = [match.upper() for match in _CITATION_RE.findall(answer or "")]
        valid_cited: list[str] = []
        for source_id in cited:
            if source_id in by_id and source_id not in valid_cited:
                valid_cited.append(source_id)

        def replace_invalid(match: re.Match) -> str:
            label = match.group(1).upper()
            return match.group(0) if label in by_id else ""

        cleaned = _CITATION_RE.sub(replace_invalid, answer or "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

        if valid_cited:
            filtered = [by_id[source_id] for source_id in valid_cited]
            return cleaned, filtered, "cited"

        # Do not invent source markers.  Keep a small evidence list so the user
        # can still inspect where the grounded answer came from.
        return cleaned, sources[:4], "evidence_available_uncited"

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        result = super().chat(
            question,
            conversation_id=conversation_id,
            document_ids=document_ids,
        )

        intent = str(result.get("intent") or "")
        evidence_status = str(result.get("evidence_status") or "")
        sources = list(result.get("sources") or [])

        grounded_turn = (
            evidence_status in {"supported", "carried_forward", "model_unavailable"}
            and (intent == "document_query" or intent == "text_search" or intent.startswith("conversation_"))
        )

        if grounded_turn:
            answer, sources, citation_status = self._finalize_sources(
                str(result.get("answer") or ""),
                sources,
            )
            result["answer"] = answer
            result["sources"] = sources
            result["retrieved_chunks"] = len(sources)
            result["citation_status"] = citation_status
        else:
            result["citation_status"] = "not_applicable"

        # Compact machine-readable grounding metadata is useful for tests and
        # future evaluation, while user-facing answers remain natural.
        result["grounding"] = {
            "intent": intent,
            "evidence_status": evidence_status,
            "citation_status": result["citation_status"],
            "used_pdf_evidence": bool(result.get("sources")),
            "retrieval": result.get("retrieval") or "not_used",
        }
        return result
