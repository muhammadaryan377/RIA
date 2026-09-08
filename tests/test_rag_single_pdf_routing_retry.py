"""Regressions from live single-PDF routing and broad-summary instability."""

from insight_rag.semantic_router import RouterOutput, SemanticRouter
from insight_rag.summary_retry import _PATCHED as SUMMARY_RETRY_PATCHED


def _summary_route(document_keys):
    return RouterOutput(
        scope="DOCUMENT",
        task="DOCUMENT_SUMMARY",
        confidence=0.98,
        document_keys=document_keys,
        broad_query=True,
        reason="whole document summary",
    )


def test_single_selected_pdf_ignores_invalid_model_document_key():
    router = SemanticRouter.__new__(SemanticRouter)
    decision = router._normalise(
        _summary_route(["PDF"]),
        key_to_id={"D1": "doc-1"},
        selected_set={"doc-1"},
        documents=[{"document_id": "doc-1", "filename": "First 3 topic.pdf"}],
        latency_ms=1.0,
    )

    assert decision.scope == "DOCUMENT"
    assert decision.task == "DOCUMENT_SUMMARY"
    assert decision.document_ids == ("doc-1",)
    assert decision.needs_clarification is False


def test_unknown_document_key_still_fails_closed_when_multiple_pdfs_are_eligible():
    router = SemanticRouter.__new__(SemanticRouter)
    decision = router._normalise(
        _summary_route(["PDF"]),
        key_to_id={"D1": "doc-1", "D2": "doc-2"},
        selected_set={"doc-1", "doc-2"},
        documents=[
            {"document_id": "doc-1", "filename": "one.pdf"},
            {"document_id": "doc-2", "filename": "two.pdf"},
        ],
        latency_ms=1.0,
    )

    assert decision.scope == "CLARIFICATION"
    assert decision.task == "CLARIFY"
    assert decision.needs_clarification is True


def test_inventory_request_does_not_depend_on_model_document_key():
    router = SemanticRouter.__new__(SemanticRouter)
    output = RouterOutput(
        scope="DOCUMENT",
        task="DOCUMENT_METADATA",
        confidence=0.99,
        document_keys=["MY_PDFS"],
        metadata_kind="INVENTORY_COUNT",
        reason="count uploaded PDFs",
    )
    decision = router._normalise(
        output,
        key_to_id={"D1": "doc-1", "D2": "doc-2"},
        selected_set=set(),
        documents=[{"document_id": "doc-1"}, {"document_id": "doc-2"}],
        latency_ms=1.0,
    )

    assert decision.scope == "DOCUMENT"
    assert decision.task == "DOCUMENT_METADATA"
    assert decision.document_ids == ("doc-1", "doc-2")
    assert decision.needs_clarification is False


def test_summary_retry_patch_is_enabled_by_public_package_import():
    # Importing insight_rag applies the retry before the public class is exported.
    assert SUMMARY_RETRY_PATCHED is True
