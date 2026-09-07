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


def test_max_operation_includes_row_label():
    facts = build_table_facts(
        "profit by product",
        [table_doc()],
        operations=("MAX",),
    )
    assert "Profit max=25000" in facts
    assert "Product=Laptop" in facts


def test_sum_operation_is_computed_locally():
    facts = build_table_facts(
        "sales",
        [table_doc()],
        operations=("SUM",),
    )
    assert "Sales sum=275000" in facts


def test_mean_operation_is_computed_locally():
    facts = build_table_facts(
        "profit",
        [table_doc()],
        operations=("MEAN",),
    )
    assert "Profit mean=17333.3" in facts


def test_structured_filter_returns_matching_rows():
    facts = build_table_facts(
        "sales by product",
        [table_doc()],
        operations=("FILTER",),
        filter_operator="GT",
        filter_value=80000,
    )
    assert "Product=Laptop" in facts
    assert "Product=Phone" in facts
    assert "Product=Tablet" not in facts


def test_rank_operation_returns_top_rows():
    facts = build_table_facts(
        "sales by product",
        [table_doc()],
        operations=("RANK",),
        top_n=2,
    )
    assert "Product=Laptop" in facts
    assert "Product=Phone" in facts
    assert "Product=Tablet" not in facts


def test_no_structured_operation_produces_no_math_facts():
    assert build_table_facts("describe products", [table_doc()], operations=()) == ""
