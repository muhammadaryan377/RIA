"""Focused tests for ARIA multi-source RAG routing and structured retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from insight_rag.multisource import InsightMultiSourceRAG
from insight_rag.source_router import MultiSourceRouter
from insight_rag.structured_retrieval import (
    StructuredDataRetriever,
    StructuredSourceConfig,
)


class FakeRouterLLM:
    provider = "cloud"

    def __init__(self, payload=None, fail=False):
        self.models = {}
        self.payload = payload or {}
        self.fail = fail

    def chat_structured(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("router offline")
        return json.dumps(self.payload)

    def chat(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("router offline")
        return json.dumps(self.payload)


def _route_payload(scope, task, **overrides):
    payload = {
        "scope": scope,
        "task": task,
        "confidence": 0.98,
        "rewritten_question": None,
        "previous_action": "NONE",
        "transform_instruction": None,
        "clarification_question": None,
        "reason": "test",
    }
    payload.update(overrides)
    return payload


def test_source_router_routes_structured_query():
    llm = FakeRouterLLM(_route_payload("STRUCTURED", "STRUCTURED_QUERY"))
    router = MultiSourceRouter(llm)

    decision = router.decide(
        "show total sales by region",
        history=[],
        documents=[],
        structured={
            "available": True,
            "source_type": "relational",
            "dialect": "postgresql",
            "database": "northwind",
        },
    )

    assert decision.scope == "STRUCTURED"
    assert decision.task == "STRUCTURED_QUERY"
    assert decision.confidence == 0.98


def test_source_router_explicit_mode_has_safe_fallback():
    llm = FakeRouterLLM(fail=True)
    router = MultiSourceRouter(llm)

    decision = router.decide(
        "total orders",
        history=[],
        documents=[],
        structured={
            "available": True,
            "source_type": "relational",
            "dialect": "postgresql",
            "database": "northwind",
        },
        source_mode="structured",
    )

    assert decision.scope == "STRUCTURED"
    assert decision.task == "STRUCTURED_QUERY"
    assert decision.confidence == 1.0


class _FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class FakeGoalAgent:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.engine = _FakeEngine()
        FakeGoalAgent.last_instance = self

    def process_goal(self, question, output_path):
        payload = {
            "status": "success",
            "user_goal": question,
            "sql_used": (
                "SELECT region, SUM(amount) AS revenue "
                "FROM sales GROUP BY region ORDER BY revenue DESC"
            ),
            "row_count": 2,
            "join_path": ["sales"],
            "kpi_alignment": {"kpis": ["revenue"], "dimensions": ["region"]},
            "data": [
                {"region": "North", "revenue": 1250.0},
                {"region": "South", "revenue": 900.0},
            ],
            "warnings": [],
        }
        Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
        return output_path


def _config(tmp_path):
    schema = {
        "database": "northwind",
        "schema": "public",
        "tables": {
            "sales": {
                "columns": [
                    {"column": "region", "data_type": "TEXT"},
                    {"column": "amount", "data_type": "NUMERIC"},
                ],
                "primary_key": [],
                "row_count": 20,
            }
        },
        "relationship_edges": [],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return StructuredSourceConfig(
        source_type="relational",
        dialect="postgresql",
        database="northwind",
        schema_path=str(schema_path),
        db_uri="postgresql://example.invalid/northwind",
        processed_path=str(tmp_path / "processed.json"),
    )


def test_structured_retriever_converts_goal_agent_result_to_evidence(tmp_path):
    retriever = StructuredDataRetriever(
        provider=object(),
        config=_config(tmp_path),
        agent_factory=FakeGoalAgent,
    )

    result = retriever.retrieve("revenue by region")

    assert result["status"] == "supported"
    assert result["row_count"] == 2
    assert result["columns"] == ["region", "revenue"]
    evidence = result["items"][0]
    assert evidence["source_kind"] == "database"
    assert "Validated SQL:" in evidence["text"]
    assert '"North"' in evidence["text"]
    assert FakeGoalAgent.last_instance.engine.disposed is True


def test_structured_schema_is_deterministic_evidence(tmp_path):
    retriever = StructuredDataRetriever(
        provider=object(),
        config=_config(tmp_path),
        agent_factory=FakeGoalAgent,
    )

    result = retriever.retrieve("what tables do we have", schema_only=True)

    assert result["status"] == "supported"
    text = result["items"][0]["text"]
    assert "sales" in text
    assert "region" in text
    assert "amount" in text


def test_multisource_evidence_labels_keep_source_provenance():
    context, sources = InsightMultiSourceRAG._label_evidence(
        [
            {
                "source_kind": "pdf",
                "filename": "targets.pdf",
                "page": 2,
                "content_type": "text",
                "document_id": "doc1",
                "chunk_id": "c1",
                "text": "Target revenue is 1000.",
            },
            {
                "source_kind": "database",
                "database": "northwind",
                "dialect": "postgresql",
                "content_type": "query_result",
                "sql": "SELECT SUM(amount) FROM sales",
                "row_count": 1,
                "columns": ["sum"],
                "text": "Exact result: 1250",
            },
        ]
    )

    assert "[S1]" in context and "[S2]" in context
    assert sources[0]["source_kind"] == "pdf"
    assert sources[1]["source_kind"] == "database"
    assert sources[1]["source_id"] == "S2"
