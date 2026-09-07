"""Unit tests for deterministic context engineering after semantic routing."""

from langchain_core.documents import Document

from insight_rag.context_engine import ContextEngineer
from insight_rag.semantic_router import RouteDecision


DOCS = [
    {
        "document_id": "doc-a",
        "filename": "First 3 topic.pdf",
        "pages": 7,
        "tables": 1,
        "total_chunks": 14,
    },
    {
        "document_id": "doc-b",
        "filename": "Sales Report 2025.pdf",
        "pages": 12,
        "tables": 4,
        "total_chunks": 28,
    },
]


def decision(**overrides):
    base = dict(
        scope="DOCUMENT",
        task="DOCUMENT_QA",
        confidence=0.95,
        document_ids=("doc-a",),
        requires_retrieval=True,
    )
    base.update(overrides)
    return RouteDecision(**base)


def test_inventory_route_becomes_local_inventory_plan():
    engine = ContextEngineer()
    route = decision(
        task="DOCUMENT_METADATA",
        document_ids=("doc-a", "doc-b"),
        metadata_kind="INVENTORY_COUNT",
        requires_retrieval=False,
    )
    plan = engine.plan(
        "count my documents",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
        route_decision=route,
    )
    assert plan.intent == "document_inventory"
    answer = engine.inventory_answer(plan.metadata_kind, documents=DOCS)
    assert "2 PDFs" in answer


def test_router_clarification_is_preserved_without_guessing():
    engine = ContextEngineer()
    route = decision(
        scope="CLARIFICATION",
        task="CLARIFY",
        confidence=0.80,
        document_ids=(),
        needs_clarification=True,
        clarification_question="Which PDF should I use?",
        requires_retrieval=False,
    )
    plan = engine.plan(
        "that one",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a", "doc-b"],
        route_decision=route,
    )
    assert plan.intent == "clarification"
    assert plan.clarification == "Which PDF should I use?"


def test_exact_search_plan_uses_router_search_term():
    engine = ContextEngineer()
    route = decision(
        task="DOCUMENT_SEARCH",
        search_term="data science lifecycle",
        exact_search=True,
        requires_retrieval=False,
    )
    plan = engine.plan(
        "locate it",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
        route_decision=route,
    )
    assert plan.intent == "text_search"
    assert plan.text_search_term == "data science lifecycle"
    assert plan.exact_search is True


def test_page_scope_and_summary_flags_are_preserved():
    route = decision(
        task="DOCUMENT_SUMMARY",
        target_pages=(4,),
        broad_query=True,
    )
    plan = ContextEngineer().plan(
        "summarize",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
        route_decision=route,
    )
    assert plan.page_numbers == (4,)
    assert plan.broad_query is True


def test_unselected_document_id_cannot_enter_content_plan():
    route = decision(document_ids=("doc-b",))
    plan = ContextEngineer().plan(
        "question",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
        route_decision=route,
    )
    assert plan.document_ids == ("doc-a",)


def test_cross_document_comparison_and_table_plan_are_preserved():
    route = decision(
        task="DOCUMENT_COMPARE",
        document_ids=("doc-a", "doc-b"),
        cross_document=True,
        prefer_tables=True,
        table_operations=("COMPARE", "MAX"),
    )
    plan = ContextEngineer().plan(
        "compare",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a", "doc-b"],
        route_decision=route,
    )
    assert plan.cross_document is True
    assert plan.prefer_tables is True
    assert plan.table_operations == ("COMPARE", "MAX")


def test_evidence_selection_deduplicates_and_diversifies():
    engine = ContextEngineer()
    docs = [
        Document(
            page_content="Revenue was 100.",
            metadata={"chunk_id": "a1", "document_id": "doc-a", "page": 1, "content_type": "text"},
        ),
        Document(
            page_content="Revenue was 100.",
            metadata={"chunk_id": "a2", "document_id": "doc-a", "page": 1, "content_type": "text"},
        ),
        Document(
            page_content="Revenue was 200.",
            metadata={"chunk_id": "b1", "document_id": "doc-b", "page": 2, "content_type": "text"},
        ),
    ]
    selected = engine.select_evidence(
        "comparison",
        docs,
        selected_document_ids=["doc-a", "doc-b"],
        cross_document=True,
    )
    assert len(selected) == 2
    assert {document.metadata["document_id"] for document in selected} == {"doc-a", "doc-b"}


def test_broad_evidence_selection_balances_pages():
    docs = [
        Document(
            page_content=f"content {page}",
            metadata={"chunk_id": f"p{page}", "document_id": "doc-a", "page": page, "content_type": "text"},
        )
        for page in range(1, 8)
    ]
    selected = ContextEngineer.select_evidence(
        "summary",
        docs,
        selected_document_ids=["doc-a"],
        broad_query=True,
    )
    assert {document.metadata["page"] for document in selected} == set(range(1, 8))
