"""Final response-quality layer for ARIA Insight PDF-RAG.

The lower layers handle routing, context planning, adaptive retrieval, evidence
gates, conversation actions and graceful failures. This layer enforces the
user-facing citation contract and adds an auditable grounding-quality index.
"""

from __future__ import annotations

import re

from .conversation_layer import InsightPDFRAG as ConversationalInsightPDFRAG
from .enterprise_quality import grounding_quality
from .grounding import REFUSAL, citation_integrity


_CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


class InsightPDFRAG(ConversationalInsightPDFRAG):
    """Production export for ARIA's Insight Agent PDF capability."""

    @staticmethod
    def _finalize_sources(answer: str, sources: list[dict]) -> tuple[str, list[dict], str]:
        """Return citation-safe answer and only the sources the answer references.

        LLMs can occasionally emit a source label that was not in the supplied
        context. Never surface such a label as if it were real evidence. When the
        answer contains valid labels, hide unrelated retrieved chunks so the UI
        stays concise. If the model omits labels entirely, retain at most four
        retrieved sources for transparency rather than inventing a citation.
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

        # Do not invent source markers. Keep a small evidence list so the user can
        # still inspect where the grounded answer came from.
        return cleaned, sources[:4], "evidence_available_uncited"

    def finalize_result(self, result: dict) -> dict:
        """Finalise every route before the outer scope layer persists the turn."""
        intent = str(result.get("intent") or "")
        evidence_status = str(result.get("evidence_status") or "")
        sources = list(result.get("sources") or [])
        grounded = evidence_status in {"supported", "carried_forward"} and bool(sources)
        citation_status = "not_applicable"
        if grounded:
            integrity = citation_integrity(str(result.get("answer") or ""), sources)
            if integrity != "valid":
                result["answer"] = REFUSAL
                result["evidence_status"] = "verification_failed"
                sources = []
                citation_status = integrity
            else:
                answer, sources, citation_status = self._finalize_sources(result["answer"], sources)
                result["answer"] = answer
        elif evidence_status in {"model_unavailable", "verification_failed", "verification_unavailable"}:
            sources = []

        result["sources"] = sources
        result["retrieved_chunks"] = len(sources)
        result["citation_status"] = citation_status

        context_plan = dict(result.get("context_plan") or {})
        quality = grounding_quality(
            evidence_status=str(result.get("evidence_status") or ""),
            citation_status=citation_status,
            sources=sources,
            requested_document_ids=list(result.get("document_ids") or []),
            cross_document=bool(context_plan.get("cross_document")),
        )
        result["grounding_quality"] = quality
        result["grounding"] = {
            "intent": intent,
            "evidence_status": result.get("evidence_status"),
            "citation_status": citation_status,
            "used_pdf_evidence": bool(sources),
            "retrieval": result.get("retrieval") or "not_used",
            "quality_level": quality.get("level"),
            "quality_score": quality.get("score"),
            "quality_is_calibrated_probability": False,
        }
        return result
