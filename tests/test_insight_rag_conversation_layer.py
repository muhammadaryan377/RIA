"""Tests for previous-answer helpers and broad evidence balancing."""

from langchain_core.documents import Document

from insight_rag.context_engine import ContextEngineer
from insight_rag.conversation_layer import InsightPDFRAG


def test_previous_answer_source_summary_uses_grounded_source_metadata():
    answer = InsightPDFRAG._source_summary(
        [
            {
                "source_id": "S1",
                "document_id": "doc-a",
                "filename": "sales.pdf",
                "page": 3,
                "content_type": "text",
            },
            {
                "source_id": "S2",
                "document_id": "doc-a",
                "filename": "sales.pdf",
                "page": 4,
                "content_type": "table",
                "table_index": 1,
            },
        ]
    )
    assert "sales.pdf, page 3" in answer
    assert "sales.pdf, page 4, table 1" in answer


def test_previous_answer_source_summary_deduplicates_same_location():
    answer = InsightPDFRAG._source_summary(
        [
            {"filename": "sales.pdf", "page": 3, "table_index": None},
            {"filename": "sales.pdf", "page": 3, "table_index": None},
        ]
    )
    assert answer.count("sales.pdf, page 3") == 1


def test_broad_summary_evidence_is_balanced_across_pages():
    docs = []
    for page in range(1, 8):
        docs.append(
            Document(
                page_content=f"Main content from page {page}",
                metadata={
                    "chunk_id": f"p{page}-a",
                    "document_id": "doc-a",
                    "page": page,
                    "content_type": "text",
                },
            )
        )
        docs.append(
            Document(
                page_content=f"Secondary content from page {page}",
                metadata={
                    "chunk_id": f"p{page}-b",
                    "document_id": "doc-a",
                    "page": page,
                    "content_type": "text",
                },
            )
        )

    selected = ContextEngineer.select_evidence(
        "summary",
        docs,
        selected_document_ids=["doc-a"],
        broad_query=True,
    )
    assert 7 <= len(selected) <= 8
    assert {doc.metadata["page"] for doc in selected} == set(range(1, 8))
