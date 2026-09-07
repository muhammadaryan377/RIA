"""Regression tests for ARIA Insight PDF-RAG context engineering."""

from insight_rag.context_engine import ContextEngineer
from insight_rag.final_layer import InsightPDFRAG


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


def test_inventory_question_never_becomes_pdf_content_question():
    plan = ContextEngineer().plan(
        "well how many pdfs are you have",
        history=[],
        documents=_docs(),
        selected_document_ids=None,
    )
    assert plan.intent == "document_inventory"


def test_vague_find_text_requests_clarification_instead_of_random_retrieval():
    plan = ContextEngineer().plan(
        "can you find the text",
        history=[],
        documents=_docs(),
        selected_document_ids=["doc-a"],
    )
    assert plan.intent == "clarification"
    assert "exact text" in (plan.clarification or "").lower()


def test_exact_phrase_search_is_detected():
    plan = ContextEngineer().plan(
        'find "data science lifecycle" in this pdf',
        history=[],
        documents=[_docs()[0]],
        selected_document_ids=["doc-a"],
    )
    assert plan.intent == "text_search"
    assert plan.text_search_term == "data science lifecycle"


def test_page_target_is_preserved_in_plan():
    plan = ContextEngineer().plan(
        "summarize page 4",
        history=[],
        documents=[_docs()[0]],
        selected_document_ids=["doc-a"],
    )
    assert 4 in plan.page_numbers


def test_singular_pdf_reference_with_multiple_active_docs_requires_clarification():
    plan = ContextEngineer().plan(
        "what is this pdf about?",
        history=[],
        documents=_docs(),
        selected_document_ids=["doc-a", "doc-b"],
    )
    assert plan.intent == "clarification"


def test_explicit_filename_resolves_document_scope():
    plan = ContextEngineer().plan(
        "what was revenue in Annual Sales 2025.pdf?",
        history=[],
        documents=_docs(),
        selected_document_ids=None,
    )
    assert plan.document_ids == ("doc-b",)


def test_cross_pdf_comparison_is_detected():
    plan = ContextEngineer().plan(
        "compare revenue across both PDFs",
        history=[],
        documents=_docs(),
        selected_document_ids=["doc-a", "doc-b"],
    )
    assert plan.cross_document is True


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
