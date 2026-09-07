"""Unit tests for deterministic context engineering in Insight PDF-RAG."""

from langchain_core.documents import Document

from insight_rag.context_engine import ContextEngineer


DOCS = [
    {
        "document_id": "doc-a",
        "filename": "First 3 topic.pdf",
        "pages": 7,
        "tables": 1,
        "total_chunks": 14,
        "created_at": "2026-09-07T10:00:00Z",
    },
    {
        "document_id": "doc-b",
        "filename": "Sales Report 2025.pdf",
        "pages": 12,
        "tables": 4,
        "total_chunks": 28,
        "created_at": "2026-09-06T10:00:00Z",
    },
]


def test_inventory_question_does_not_use_rag():
    engine = ContextEngineer()
    plan = engine.plan(
        "well how many pdfs are you have",
        history=[],
        documents=DOCS,
        selected_document_ids=None,
    )
    assert plan.intent == "document_inventory"
    answer = engine.inventory_answer(
        "how many pdfs do you have",
        documents=DOCS,
        selected_document_ids=None,
    )
    assert "2 PDFs" in answer
    assert "First 3 topic.pdf" in answer


def test_vague_find_request_asks_for_clarification():
    engine = ContextEngineer()
    plan = engine.plan(
        "can you find the text",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert plan.intent == "clarification"
    assert "What exact text" in (plan.clarification or "")


def test_exact_find_extracts_term_without_running_inventory_logic():
    engine = ContextEngineer()
    plan = engine.plan(
        'find "data science lifecycle"',
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert plan.intent == "text_search"
    assert plan.text_search_term == "data science lifecycle"
    assert plan.document_ids == ("doc-a",)


def test_page_range_is_resolved_and_capped():
    engine = ContextEngineer()
    assert engine.extract_page_numbers("summarize pages 2-4") == [2, 3, 4]
    assert engine.extract_page_numbers("what is on p.7?") == [7]


def test_this_pdf_resolves_from_recent_sources():
    engine = ContextEngineer()
    history = [
        {"role": "user", "content": "what was revenue?", "sources": []},
        {
            "role": "assistant",
            "content": "Revenue was reported in the sales PDF.",
            "sources": [{"document_id": "doc-b", "filename": "Sales Report 2025.pdf", "page": 3}],
        },
    ]
    plan = engine.plan(
        "what is on page 4 of this pdf?",
        history=history,
        documents=DOCS,
        selected_document_ids=None,
    )
    assert plan.document_ids == ("doc-b",)
    assert plan.page_numbers == (4,)


def test_explicit_filename_overrides_broad_selection():
    engine = ContextEngineer()
    plan = engine.plan(
        "summarize Sales Report 2025.pdf",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a", "doc-b"],
    )
    assert plan.document_ids == ("doc-b",)


def test_cross_document_comparison_is_detected():
    engine = ContextEngineer()
    plan = engine.plan(
        "compare the main findings in both PDFs",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a", "doc-b"],
    )
    assert plan.cross_document is True


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
        "compare revenue in both PDFs",
        docs,
        selected_document_ids=["doc-a", "doc-b"],
        cross_document=True,
    )
    assert len(selected) == 2
    assert {d.metadata["document_id"] for d in selected} == {"doc-a", "doc-b"}
