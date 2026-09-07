"""Regression tests for ARIA Insight PDF-RAG context engineering."""

from insight_rag.context_engine import ContextEngineer
from insight_rag.final_layer import InsightPDFRAG
from insight_rag.semantic_router import RouteDecision


def _docs():
    return [
        {
            "document_id": "doc-a",
            "filename": "First 3 topic.pdf",
            "pages": 7,
            "tables": 1,
            "total_chunks": 12,
        },
        {
            "document_id": "doc-b",
            "filename": "Annual Sales 2025.pdf",
            "pages": 10,
            "tables": 3,
            "total_chunks": 22,
        },
    ]


def _decision(**overrides):
    base = dict(
        scope="DOCUMENT",
        task="DOCUMENT_QA",
        confidence=0.96,
        document_ids=("doc-a",),
        requires_retrieval=True,
    )
    base.update(overrides)
    return RouteDecision(**base)


def test_metadata_plan_never_becomes_content_retrieval():
    route = _decision(
        task="DOCUMENT_METADATA",
        document_ids=("doc-a", "doc-b"),
        metadata_kind="INVENTORY_LIST",
        requires_retrieval=False,
    )
    plan = ContextEngineer().plan(
        "list them",
        history=[],
        documents=_docs(),
        selected_document_ids=None,
        route_decision=route,
    )
    assert plan.intent == "document_inventory"


def test_search_without_target_becomes_clarification():
    route = _decision(
        task="DOCUMENT_SEARCH",
        search_term=None,
        exact_search=False,
    )
    plan = ContextEngineer().plan(
        "find it",
        history=[],
        documents=_docs(),
        selected_document_ids=["doc-a"],
        route_decision=route,
    )
    assert plan.intent == "clarification"


def test_exact_phrase_search_is_router_driven():
    route = _decision(
        task="DOCUMENT_SEARCH",
        search_term="data science lifecycle",
        exact_search=True,
        requires_retrieval=False,
    )
    plan = ContextEngineer().plan(
        "search request",
        history=[],
        documents=[_docs()[0]],
        selected_document_ids=["doc-a"],
        route_decision=route,
    )
    assert plan.intent == "text_search"
    assert plan.text_search_term == "data science lifecycle"


def test_cross_pdf_comparison_is_router_driven():
    route = _decision(
        task="DOCUMENT_COMPARE",
        document_ids=("doc-a", "doc-b"),
        cross_document=True,
        needs_query_decomposition=True,
    )
    plan = ContextEngineer().plan(
        "compare",
        history=[],
        documents=_docs(),
        selected_document_ids=["doc-a", "doc-b"],
        route_decision=route,
    )
    assert plan.cross_document is True
    assert plan.complex_query is True


def test_invalid_model_source_labels_are_removed():
    answer, sources, citation_status = InsightPDFRAG._finalize_sources(
        "Revenue increased [S1], but profit doubled [S9].",
        [
            {"source_id": "S1", "filename": "Annual Sales 2025.pdf", "page": 2},
            {"source_id": "S2", "filename": "Annual Sales 2025.pdf", "page": 3},
        ],
    )
    assert "[S1]" in answer
    assert "[S9]" not in answer
    assert [source["source_id"] for source in sources] == ["S1"]
    assert citation_status == "cited"


def test_uncited_grounded_answer_keeps_only_small_evidence_set():
    answer, sources, citation_status = InsightPDFRAG._finalize_sources(
        "The document discusses data science.",
        [
            {"source_id": f"S{i}", "filename": "First 3 topic.pdf", "page": i}
            for i in range(1, 9)
        ],
    )
    assert answer == "The document discusses data science."
    assert len(sources) == 4
    assert citation_status == "evidence_available_uncited"
