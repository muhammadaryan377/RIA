"""Regression tests for live Groq compatibility hardening."""

from types import SimpleNamespace

import pytest

from insight_rag.groq_stability import _retry_after_seconds
from insight_rag.semantic_router import CompactRouterOutput, SemanticRouter
from llm_provider import LLMProvider


def _rich_route(**overrides):
    payload = {
        "scope": "DOCUMENT",
        "task": "DOCUMENT_SUMMARY",
        "confidence": 0.95,
        "needs_clarification": False,
        "clarification_question": None,
        "document_keys": ["D1"],
        "target_pages": None,
        "search_term": None,
        "exact_search": False,
        "broad_query": True,
        "cross_document": False,
        "needs_rewrite": False,
        "needs_query_decomposition": False,
        "prefer_tables": False,
        "previous_action": "NONE",
        "transform_instruction": None,
        "metadata_kind": "NONE",
        "table_operations": None,
        "table_filter_operator": "NONE",
        "table_filter_value": None,
        "table_top_n": None,
        "reason": "summary",
    }
    payload.update(overrides)
    return payload


def test_groq_schema_allows_null_optional_arrays():
    strict = SemanticRouter._strict_json_schema()
    compact = SemanticRouter._compact_json_schema()

    assert "null" in strict["properties"]["target_pages"]["type"]
    assert "null" in strict["properties"]["document_keys"]["type"]
    assert "null" in strict["properties"]["table_operations"]["type"]
    assert "null" in compact["properties"]["target_pages"]["type"]


def test_rich_router_normalises_null_arrays_to_empty_lists():
    parsed = SemanticRouter._validate_rich(_rich_route(document_keys=None))
    assert parsed.document_keys == []
    assert parsed.target_pages == []
    assert parsed.table_operations == []


def test_compact_router_normalises_null_arrays_to_empty_lists():
    parsed = CompactRouterOutput.model_validate({
        "scope": "DOCUMENT",
        "task": "DOCUMENT_SUMMARY",
        "confidence": 0.9,
        "needs_clarification": False,
        "clarification_question": None,
        "document_keys": None,
        "target_pages": None,
        "search_term": None,
        "exact_search": False,
        "needs_rewrite": False,
        "metadata_kind": "NONE",
        "previous_action": "NONE",
        "transform_instruction": None,
        "reason": "summary",
    })
    assert parsed.document_keys == []
    assert parsed.target_pages == []


def test_task_as_scope_alias_is_accepted_and_normalised():
    strict = SemanticRouter._strict_json_schema()
    assert "DOCUMENT_METADATA" in strict["properties"]["scope"]["enum"]

    parsed = SemanticRouter._validate_rich(_rich_route(
        scope="DOCUMENT_METADATA",
        task="DOCUMENT_METADATA",
        broad_query=False,
        metadata_kind="INVENTORY_COUNT",
        document_keys=None,
    ))
    assert parsed.scope == "DOCUMENT"
    assert parsed.task == "DOCUMENT_METADATA"
    assert parsed.document_keys == []


def test_router_prompt_covers_whole_pdf_explanation_and_previous_answer_pronouns():
    messages = SemanticRouter._messages(
        "explain it", manifest="D1: filename='First 3 topic.pdf'; pages=10; tables=0; selected=yes",
        history_text="ASSISTANT: You currently have 1 PDF: First 3 topic.pdf.",
    )
    policy = messages[0]["content"]
    assert "explain my PDF" in policy
    assert "DOCUMENT/DOCUMENT_SUMMARY" in policy
    assert "'explain it'" in policy
    assert "CONVERSATION/PREVIOUS_ANSWER" in policy
    assert "scope=DOCUMENT and task=DOCUMENT_METADATA" in policy


class _RateLimitError(RuntimeError):
    status_code = 429

    def __init__(self, message="rate limited", retry_after=None):
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(status_code=429, headers=headers)


def test_retry_hint_prefers_header_then_error_text():
    assert _retry_after_seconds(_RateLimitError(retry_after=11.7)) == pytest.approx(11.7)
    assert _retry_after_seconds(_RateLimitError("Please try again in 8.13s")) == pytest.approx(8.13)


def test_cloud_retry_waits_for_server_hint(monkeypatch):
    waits = []
    monkeypatch.setattr("insight_rag.groq_stability.time.sleep", waits.append)

    calls = 0

    def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _RateLimitError(retry_after=11.7)
        return "ok"

    provider = LLMProvider.__new__(LLMProvider)
    assert provider._cloud_with_retry(request) == "ok"
    assert calls == 2
    assert waits == [pytest.approx(12.2)]


def test_cloud_retry_does_not_retry_non_429(monkeypatch):
    waits = []
    monkeypatch.setattr("insight_rag.groq_stability.time.sleep", waits.append)

    class BadRequest(RuntimeError):
        status_code = 400

    provider = LLMProvider.__new__(LLMProvider)
    with pytest.raises(BadRequest):
        provider._cloud_with_retry(lambda: (_ for _ in ()).throw(BadRequest("bad schema")))
    assert waits == []
