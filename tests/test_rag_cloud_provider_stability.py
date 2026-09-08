"""Regression tests for DeepSeek cloud-provider compatibility."""

from types import SimpleNamespace

import pytest

from insight_rag.provider_stability import _normalise_provider_payload
from insight_rag.semantic_router import CompactRouterOutput, SemanticRouter
from llm_provider import DEEPSEEK_MODEL, LLMProvider, PROVIDERS


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


def test_cloud_provider_metadata_uses_deepseek():
    cloud = PROVIDERS["cloud"]
    assert "DeepSeek" in cloud["label"]
    assert cloud["models"]["sql"] == DEEPSEEK_MODEL
    assert cloud["models"]["sql"].startswith("deepseek-")


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


def test_task_as_scope_alias_is_normalised_before_validation():
    payload = _normalise_provider_payload(_rich_route(
        scope="DOCUMENT_METADATA",
        task="DOCUMENT_METADATA",
        broad_query=False,
        metadata_kind="INVENTORY_COUNT",
        document_keys=None,
    ))
    parsed = SemanticRouter._validate_rich(payload)
    assert parsed.scope == "DOCUMENT"
    assert parsed.task == "DOCUMENT_METADATA"
    assert parsed.document_keys == []


def test_router_prompt_covers_whole_pdf_and_previous_answer_transform():
    messages = SemanticRouter._messages(
        "explain it",
        manifest="D1: filename='First 3 topic.pdf'; pages=10; tables=0; selected=yes",
        history_text="ASSISTANT: Previous grounded answer.",
    )
    policy = messages[0]["content"]
    assert "DOCUMENT/DOCUMENT_SUMMARY" in policy
    assert "CONVERSATION/PREVIOUS_ANSWER" in policy
    assert "scope=DOCUMENT and task=DOCUMENT_METADATA" in policy


class _RateLimitError(RuntimeError):
    status_code = 429

    def __init__(self, message="rate limited", retry_after=None):
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(status_code=429, headers=headers)


def test_retry_hint_prefers_header_then_error_text():
    provider = LLMProvider.__new__(LLMProvider)
    assert provider._retry_after_seconds(_RateLimitError(retry_after=11.7)) == pytest.approx(11.7)
    assert provider._retry_after_seconds(_RateLimitError("Please try again in 8.13s")) == pytest.approx(8.13)


def test_cloud_retry_waits_for_server_hint(monkeypatch):
    waits = []
    monkeypatch.setattr("llm_provider.time.sleep", waits.append)
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
    monkeypatch.setattr("llm_provider.time.sleep", waits.append)

    class BadRequest(RuntimeError):
        status_code = 400

    provider = LLMProvider.__new__(LLMProvider)
    with pytest.raises(BadRequest):
        provider._cloud_with_retry(lambda: (_ for _ in ()).throw(BadRequest("bad request")))
    assert waits == []
