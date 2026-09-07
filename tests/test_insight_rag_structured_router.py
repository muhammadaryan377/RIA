"""Tests for strict structured-output routing used by ARIA PDF-RAG."""

import json

from insight_rag.semantic_router import SemanticRouter


DOCS = [
    {"document_id": "doc-a", "filename": "Alpha.pdf", "pages": 5, "tables": 1},
]


def _payload(**overrides):
    value = {
        "scope": "CONVERSATION",
        "task": "CHAT",
        "confidence": 0.99,
        "needs_clarification": False,
        "clarification_question": None,
        "document_keys": [],
        "target_pages": [],
        "search_term": None,
        "exact_search": False,
        "broad_query": False,
        "cross_document": False,
        "needs_rewrite": False,
        "needs_query_decomposition": False,
        "prefer_tables": False,
        "previous_action": "NONE",
        "transform_instruction": None,
        "metadata_kind": "NONE",
        "table_operations": [],
        "table_filter_operator": "NONE",
        "table_filter_value": None,
        "table_top_n": None,
        "reason": "social interaction",
    }
    value.update(overrides)
    return value


class StructuredFakeLLM:
    def __init__(self, response):
        self.response = response
        self.models = {}
        self.structured_calls = 0
        self.plain_calls = 0

    def chat_structured(self, role, messages, **kwargs):
        self.structured_calls += 1
        assert role == "rag_scope"
        schema = kwargs["json_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        return json.dumps(self.response)

    def chat(self, role, messages, **kwargs):
        self.plain_calls += 1
        raise AssertionError("plain chat fallback should not run when structured routing succeeds")


def test_router_prefers_strict_structured_output_path():
    llm = StructuredFakeLLM(_payload())
    decision = SemanticRouter(llm).decide(
        "Hello there",
        history=[],
        documents=DOCS,
        selected_document_ids=["doc-a"],
    )
    assert decision.scope == "CONVERSATION"
    assert decision.task == "CHAT"
    assert decision.requires_retrieval is False
    assert llm.structured_calls == 1
    assert llm.plain_calls == 0


def test_router_strict_schema_contains_nullable_control_fields():
    schema = SemanticRouter._strict_json_schema()
    props = schema["properties"]
    assert props["clarification_question"]["type"] == ["string", "null"]
    assert props["search_term"]["type"] == ["string", "null"]
    assert props["table_filter_value"]["type"] == ["number", "null"]
    assert props["table_top_n"]["type"] == ["integer", "null"]
