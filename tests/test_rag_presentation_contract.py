"""Presentation-contract regressions from live PDF chat conversations."""

import json

from insight_rag.conversation_layer import InsightPDFRAG as ConversationRAG
from insight_rag.presentation import PresentationDirective, reformat_grounded_answer


class _PresentationModel:
    def __init__(self, *, verdict="SUPPORTED", fail_reshape=False):
        self.verdict = verdict
        self.fail_reshape = fail_reshape
        self.calls = []

    def chat_structured(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        name = kwargs.get("schema_name")
        if name == "aria_answer_presentation":
            return json.dumps({"shape": "ONE_LINE", "detail": "BRIEF", "simple": False})
        if name == "aria_previous_answer_transform":
            return json.dumps({"text": "The PDF covers core data science concepts and tools."})
        if name == "aria_answer_reshape":
            if self.fail_reshape:
                raise RuntimeError("temporary provider formatting failure")
            return json.dumps({"text": "The PDF covers core data science concepts and tools."})
        if name == "aria_grounding_verdict":
            return json.dumps({"verdict": self.verdict})
        raise AssertionError(f"unexpected structured schema: {name}")

    def chat(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if role == "rag_verify":
            return self.verdict
        return "The PDF covers core data science concepts and tools."


def _previous_answer():
    return {
        "role": "assistant",
        "content": "- Data science concepts are introduced. [S1]\n- Tools and workflow stages are explained. [S2]",
        "sources": [
            {"source_id": "S1", "document_id": "doc-1", "filename": "first.pdf", "page": 1},
            {"source_id": "S2", "document_id": "doc-1", "filename": "first.pdf", "page": 2},
        ],
    }


def _rag_with(model):
    rag = object.__new__(ConversationRAG)
    rag.llm = model
    return rag


def test_previous_answer_one_line_transform_preserves_sources_and_shape():
    rag = _rag_with(_PresentationModel())
    transformed = rag._transform_previous_answer("in just one line", _previous_answer())

    assert "\n" not in transformed
    assert transformed.startswith("The PDF covers core data science concepts and tools.")
    assert transformed.endswith("[S1] [S2]")


def test_previous_answer_one_line_has_deterministic_safe_fallback_after_verifier_rejection():
    rag = _rag_with(_PresentationModel(verdict="UNSUPPORTED"))
    transformed = rag._transform_previous_answer("in just one line", _previous_answer())

    # A purely presentational one-line request must not become a generic refusal.
    # The fallback changes only whitespace/bullet layout and preserves the existing
    # citation placement instead of moving labels away from their original claims.
    assert "\n" not in transformed
    assert "Data science concepts are introduced." in transformed
    assert "Tools and workflow stages are explained." in transformed
    assert "[S1]" in transformed and "[S2]" in transformed
    assert transformed.index("[S1]") < transformed.index("Tools and workflow stages")


def test_grounded_one_line_reformat_preserves_existing_citations():
    model = _PresentationModel()
    answer = "First grounded point. [S1]\nSecond grounded point. [S2]"
    directive = PresentationDirective(shape="ONE_LINE", detail="BRIEF", simple=False)

    reshaped = reformat_grounded_answer(
        model,
        request="summarize in one line",
        answer=answer,
        directive=directive,
    )

    assert "\n" not in reshaped
    assert reshaped.endswith("[S1] [S2]")


def test_one_line_reformat_provider_failure_falls_back_without_losing_citations():
    model = _PresentationModel(fail_reshape=True)
    answer = "First grounded point. [S1]\nSecond grounded point. [S2]"
    directive = PresentationDirective(shape="ONE_LINE", detail="BRIEF", simple=False)

    reshaped = reformat_grounded_answer(
        model,
        request="one line please",
        answer=answer,
        directive=directive,
    )

    assert "\n" not in reshaped
    assert "First grounded point." in reshaped
    assert "Second grounded point." in reshaped
    assert reshaped.endswith("[S1] [S2]")
