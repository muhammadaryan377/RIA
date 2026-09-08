"""Regressions from live broad-summary failures in the PDF chat UI."""

from types import SimpleNamespace

from insight_rag.friendly import InsightPDFRAG as GroundedRAG
from insight_rag.grounding import _evidence_for_citations, verify_answer


class _Model:
    def __init__(self, response="SUPPORTED"):
        self.response = response
        self.calls = []

    def chat(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if role == "rag_verify":
            return self.response
        return "Topic one is explained here [S1]."


def test_verifier_keeps_only_cited_source_blocks():
    context = (
        "[S1] first.pdf — page 1 — text\nAlpha evidence.\n\n"
        "[S2] first.pdf — page 2 — text\nBeta evidence."
    )
    selected = _evidence_for_citations("Beta is discussed [S2].", context)
    assert "Beta evidence" in selected
    assert "Alpha evidence" not in selected


def test_verify_answer_sends_only_cited_evidence():
    model = _Model()
    context = (
        "[S1] first.pdf — page 1 — text\nAlpha evidence.\n\n"
        "[S2] first.pdf — page 2 — text\nBeta evidence."
    )
    result = verify_answer(
        model,
        question="Explain beta",
        answer="Beta is discussed [S2].",
        context=context,
    )
    assert result == "verified"
    verifier_prompt = model.calls[-1][1][1]["content"]
    assert "Beta evidence" in verifier_prompt
    assert "Alpha evidence" not in verifier_prompt


def test_document_summary_uses_citation_complete_prompt_and_lower_output_cap():
    model = _Model()
    dummy = SimpleNamespace(
        llm=model,
        _history_for_rewrite=lambda history: "",
    )
    plan = SimpleNamespace(task="DOCUMENT_SUMMARY")

    answer = GroundedRAG._secure_generate_answer(
        dummy,
        question="explain my pdf",
        context="[S1] first.pdf — page 1 — text\nA topic is described here.",
        table_facts="",
        history=[],
        plan=plan,
    )

    assert "[S1]" in answer
    role, messages, kwargs = model.calls[-1]
    assert role == "rag"
    assert kwargs["num_predict"] == 650
    assert "Every factual sentence, bullet, or short paragraph MUST contain" in messages[0]["content"]
    assert "bounded whole-document summary/explanation" in messages[1]["content"]
