"""Tests for schema-validated semantic routing in ARIA Insight PDF-RAG."""

import json

from insight_rag.semantic_router import SemanticRouter


DOCS = [
    {"document_id": "doc-a", "filename": "Alpha.pdf", "pages": 5, "tables": 1},
    {"document_id": "doc-b", "filename": "Beta.pdf", "pages": 8, "tables": 2},
]


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.models = {}

    def chat(self, role, messages, **kwargs):
        assert role == "rag_scope"
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def payload(**overrides):
    base = {
        "scope": "DOCUMENT",
        "task": "DOCUMENT_QA",
        "confidence": 0.95,
        "needs_clarification": False,
        "clarification_question": None,
        "document_keys": ["D1"],
        "target_pages": [],
        "search_term": None,
        "exact_search": False,
        "broad_query": False,
        "cross_document": False,
        "needs_rewrite": False,
        "needs_query_decomposition": False,
        "prefer_tables": False,
        "previous_action": "NONE",
        "transform_instruction": None,
        "metadata_kind": "NONE",
        "table_operations": [],
        "table_filter_operator": "NONE",
        "table_filter_value": None,
        "table_top_n": None,
        "reason": "test route",
    }
    base.update(overrides)
    return base


def test_document_route_resolves_model_document_key_to_owned_id():
    router = SemanticRouter(FakeLLM(payload()))
    decision = router.decide(
        "Explain the metric from my report",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert decision.scope == "DOCUMENT"
    assert decision.task == "DOCUMENT_QA"
    assert decision.document_ids == ("doc-a",)
    assert decision.requires_retrieval is True


def test_conversation_route_never_requires_document_retrieval():
    router = SemanticRouter(
        FakeLLM(payload(scope="CONVERSATION", task="CHAT", document_keys=[], confidence=0.99))
    )
    decision = router.decide(
        "Nice to talk with you",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
    )
    assert decision.scope == "CONVERSATION"
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_system_route_is_separate_from_pdf_evidence():
    router = SemanticRouter(
        FakeLLM(payload(scope="SYSTEM", task="SYSTEM_INFO", document_keys=[], confidence=0.98))
    )
    decision = router.decide(
        "Explain your retrieval architecture",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
    )
    assert decision.scope == "SYSTEM"
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_out_of_scope_route_does_not_become_document_query():
    router = SemanticRouter(
        FakeLLM(payload(scope="OUT_OF_SCOPE", task="OUT_OF_SCOPE", document_keys=[], confidence=0.99))
    )
    decision = router.decide(
        "Tell me an unrelated world fact",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
    )
    assert decision.scope == "OUT_OF_SCOPE"
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_low_confidence_sensitive_route_fails_to_clarification():
    router = SemanticRouter(FakeLLM(payload(confidence=0.20)))
    decision = router.decide(
        "I mean that one",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a", "doc-b"],
    )
    assert decision.scope == "CLARIFICATION"
    assert decision.needs_clarification is True
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_selected_document_boundary_fails_closed_for_unselected_key():
    router = SemanticRouter(FakeLLM(payload(document_keys=["D2"])))
    decision = router.decide(
        "Use the report",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert decision.scope == "CLARIFICATION"
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_inconsistent_scope_task_pair_fails_closed():
    router = SemanticRouter(
        FakeLLM(payload(scope="SYSTEM", task="DOCUMENT_QA", document_keys=["D1"], confidence=0.99))
    )
    decision = router.decide(
        "ambiguous malformed route",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert decision.scope == "CLARIFICATION"
    assert decision.task == "CLARIFY"
    assert decision.requires_retrieval is False
    assert decision.document_ids == ()


def test_exact_search_is_local_and_does_not_require_vector_retrieval():
    router = SemanticRouter(
        FakeLLM(
            payload(
                task="DOCUMENT_SEARCH",
                search_term="exact phrase",
                exact_search=True,
                confidence=0.99,
            )
        )
    )
    decision = router.decide(
        "exact search",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert decision.scope == "DOCUMENT"
    assert decision.task == "DOCUMENT_SEARCH"
    assert decision.requires_retrieval is False
    assert decision.search_term == "exact phrase"


def test_invalid_router_output_fails_closed():
    router = SemanticRouter(FakeLLM("not json"))
    decision = router.decide(
        "anything",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
    )
    assert decision.scope == "CLARIFICATION"
    assert decision.classifier_used is False
    assert decision.requires_retrieval is False
