"""Tests for previous-answer operations and broad-summary evidence balancing."""

from langchain_core.documents import Document

from insight_rag.conversation_layer import AdvancedContextEngineer, detect_conversation_action


def grounded_history():
    return [
        {"role": "user", "content": "What is revenue?", "sources": []},
        {
            "role": "assistant",
            "content": "Revenue was 100. [S1]",
            "sources": [
                {
                    "source_id": "S1",
                    "document_id": "doc-a",
                    "filename": "sales.pdf",
                    "page": 3,
                    "content_type": "text",
                }
            ],
        },
    ]


def test_previous_answer_source_followup_is_detected():
    action = detect_conversation_action("where did you find that?", grounded_history())
    assert action is not None
    assert action.kind == "show_sources"


def test_previous_answer_repeat_is_detected():
    action = detect_conversation_action("repeat that", grounded_history())
    assert action is not None
    assert action.kind == "repeat"


def test_previous_answer_transform_is_detected():
    for text in (
        "make it shorter",
        "explain that in simple words",
        "put it in bullet points",
        "translate that to Urdu",
    ):
        action = detect_conversation_action(text, grounded_history())
        assert action is not None, text
        assert action.kind == "transform", text


def test_transform_is_not_triggered_without_previous_answer():
    assert detect_conversation_action("make it shorter", []) is None


def test_regular_pdf_question_is_not_misclassified_as_transform():
    assert detect_conversation_action("summarize this PDF in bullet points", grounded_history()) is None


def test_broad_summary_evidence_is_balanced_across_pages():
    engine = AdvancedContextEngineer()
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

    selected = engine.select_evidence(
        "Summarize this PDF",
        docs,
        selected_document_ids=["doc-a"],
    )
    assert len(selected) == 7
    assert {doc.metadata["page"] for doc in selected} == set(range(1, 8))
