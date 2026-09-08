"""Provider-stable semantic evidence gating for PDF-RAG."""

from __future__ import annotations

from .diagnostics import record
from .friendly import InsightPDFRAG as GroundedInsightPDFRAG
from .grounding import verify_evidence

_PATCHED = False


def apply_verification_stability_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def semantic_evidence_check(self, question, docs, plan):
        if not docs:
            return False
        if plan.cross_document:
            found = {str(doc.metadata.get("document_id")) for doc in docs}
            if set(plan.document_ids) - found:
                return False
        # Broad summaries are validated after generation, where claim-level
        # citations provide a stronger check than a pre-generation sufficiency gate.
        if plan.task == "DOCUMENT_SUMMARY":
            return True

        context, _ = self._build_context(docs)
        decision = verify_evidence(self.llm, question=question, context=context)
        if decision == "verified":
            record("evidence_verifier", "supported")
            return True
        if decision == "verification_unavailable":
            record("evidence_verifier", "unavailable")
            return False
        record("evidence_verifier", "unsupported")
        return False

    GroundedInsightPDFRAG._semantic_evidence_check = semantic_evidence_check
    _PATCHED = True
