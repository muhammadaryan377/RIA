"""Deterministic table reasoning tests for Insight PDF-RAG."""

from langchain_core.documents import Document

from insight_rag.table_reasoner import build_table_facts, markdown_to_frame


TABLE = """
| Product | Sales | Profit |
| --- | --- | --- |
| Laptop | 120000 | 25000 |
| Phone | 95000 | 18000 |
| Tablet | 60000 | 9000 |
""".strip()


def table_doc():
    return Document(
        page_content=TABLE,
        metadata={
            "document_id": "doc-1",
            "filename": "sales.pdf",
            "page": 8,
            "table_index": 1,
            "content_type": "table",
            "chunk_id": "chunk-1",
        },
    )


def test_markdown_table_parses_to_frame():
    frame = markdown_to_frame(TABLE)
    assert frame is not None
    assert list(frame.columns) == ["Product", "Sales", "Profit"]
    assert len(frame) == 3


def test_highest_profit_includes_row_label():
    facts = build_table_facts("Which product had the highest profit?", [table_doc()])
    assert "Profit max=25000" in facts
    assert "Product=Laptop" in facts


def test_total_sales_is_computed_locally():
    facts = build_table_facts("What is the total sales?", [table_doc()])
    assert "Sales sum=275000" in facts


def test_average_profit_is_computed_locally():
    facts = build_table_facts("What is the average profit?", [table_doc()])
    assert "Profit mean=17333.3" in facts


def test_threshold_filter_returns_matching_rows():
    facts = build_table_facts("Which products have sales above 80000?", [table_doc()])
    assert "Product=Laptop" in facts
    assert "Product=Phone" in facts
    assert "Product=Tablet" not in facts


def test_non_numeric_question_does_not_generate_math_facts():
    assert build_table_facts("Describe the products in the table", [table_doc()]) == ""
