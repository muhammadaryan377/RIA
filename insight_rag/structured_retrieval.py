"""Grounded structured-data retrieval for ARIA Insight RAG.

Structured sources are not embedded as arbitrary row chunks.  ARIA reuses the
existing Schema Agent/Goal Agent contract: schema-aware planning -> validated
read-only SQL -> exact query result.  The result is then converted into evidence
for the Insight Agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from goal_agent import GoalAgent


MAX_STRUCTURED_CONTEXT_CHARS = int(
    os.getenv("ARIA_RAG_STRUCTURED_MAX_CONTEXT_CHARS", "16000")
)
MAX_STRUCTURED_ROWS = int(os.getenv("ARIA_RAG_STRUCTURED_MAX_ROWS", "80"))


@dataclass(frozen=True)
class StructuredSourceConfig:
    source_type: str
    dialect: str
    database: str
    schema_path: str
    db_uri: str
    processed_path: str

    @property
    def available(self) -> bool:
        return bool(self.db_uri and self.schema_path and self.processed_path and Path(self.schema_path).exists())

    def public_metadata(self) -> dict:
        return {
            "available": self.available,
            "source_type": self.source_type,
            "dialect": self.dialect,
            "database": self.database,
            "schema_file": Path(self.schema_path).name if self.schema_path else None,
        }


class StructuredDataRetriever:
    """Execute schema-grounded structured retrieval through ARIA's Goal Agent."""

    def __init__(
        self,
        *,
        provider,
        config: StructuredSourceConfig,
        agent_factory: Callable = GoalAgent,
    ):
        self.provider = provider
        self.config = config
        self.agent_factory = agent_factory

    @property
    def available(self) -> bool:
        return self.config.available and self.provider is not None

    def _load_schema(self) -> dict:
        path = Path(self.config.schema_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema mapping not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _column_name(column) -> str:
        if isinstance(column, str):
            return column
        if isinstance(column, dict):
            return str(column.get("column") or column.get("name") or column.get("column_name") or "")
        return ""

    def _schema_evidence(self) -> dict:
        mapping = self._load_schema()
        tables = mapping.get("tables") or {}
        lines = [
            f"Structured source: database={self.config.database}; dialect={self.config.dialect}; "
            f"schema={mapping.get('schema') or 'unknown'}",
            f"Tables available: {len(tables) if isinstance(tables, dict) else 0}",
        ]
        if isinstance(tables, dict):
            for table_name, table_info in list(tables.items())[:80]:
                info = table_info if isinstance(table_info, dict) else {}
                columns = [
                    self._column_name(value)
                    for value in (info.get("columns") or [])
                ]
                columns = [value for value in columns if value]
                pk = info.get("primary_key") or info.get("primary_keys") or []
                if isinstance(pk, str):
                    pk = [pk]
                row_count = info.get("row_count")
                suffix = f"; rows={row_count}" if row_count is not None else ""
                lines.append(
                    f"- {table_name}: columns={', '.join(columns[:80]) or '(unknown)'}"
                    f"; primary_key={', '.join(map(str, pk)) or '(none declared)'}{suffix}"
                )

        relationships = (
            mapping.get("relationship_edges")
            or mapping.get("declared_relationships")
            or mapping.get("relationships")
            or []
        )
        if relationships:
            lines.append("Relationships:")
            for relationship in relationships[:120]:
                if isinstance(relationship, str):
                    lines.append(f"- {relationship}")
                elif isinstance(relationship, dict):
                    source = (
                        relationship.get("source")
                        or relationship.get("from")
                        or relationship.get("source_table")
                    )
                    target = (
                        relationship.get("target")
                        or relationship.get("to")
                        or relationship.get("target_table")
                    )
                    source_col = (
                        relationship.get("source_column")
                        or relationship.get("from_column")
                        or relationship.get("column")
                    )
                    target_col = (
                        relationship.get("target_column")
                        or relationship.get("to_column")
                        or relationship.get("referenced_column")
                    )
                    relation_text = f"{source or '?'}"
                    if source_col:
                        relation_text += f".{source_col}"
                    relation_text += " -> "
                    relation_text += f"{target or '?'}"
                    if target_col:
                        relation_text += f".{target_col}"
                    lines.append(f"- {relation_text}")

        text = "\n".join(lines)
        if len(text) > MAX_STRUCTURED_CONTEXT_CHARS:
            text = text[: MAX_STRUCTURED_CONTEXT_CHARS - 20] + "\n[truncated schema]"
        return {
            "source_kind": "database" if self.config.dialect != "sqlite" else "tabular",
            "content_type": "schema",
            "database": self.config.database,
            "dialect": self.config.dialect,
            "schema_file": Path(self.config.schema_path).name,
            "text": text,
            "row_count": None,
            "sql": None,
            "columns": [],
        }

    @staticmethod
    def _columns_from_rows(rows: list[dict]) -> list[str]:
        ordered: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in row:
                name = str(key)
                if name not in ordered:
                    ordered.append(name)
        return ordered

    def _result_evidence(self, payload: dict) -> dict:
        rows = payload.get("data") or []
        if not isinstance(rows, list):
            rows = []
        rows = [row for row in rows if isinstance(row, dict)]
        row_count = int(payload.get("row_count", len(rows)) or 0)
        columns = self._columns_from_rows(rows)
        visible_rows = rows[:MAX_STRUCTURED_ROWS]

        header = [
            f"Structured query source: database={self.config.database}; dialect={self.config.dialect}",
            f"User analytical goal: {payload.get('user_goal') or payload.get('question') or ''}",
            f"Validated SQL: {payload.get('sql_used') or '(not available)'}",
            f"Returned rows: {row_count}",
            f"Columns: {', '.join(columns) if columns else '(none)'}",
        ]
        if payload.get("join_path"):
            header.append("Join path: " + " -> ".join(map(str, payload.get("join_path") or [])))
        if payload.get("kpi_alignment"):
            header.append(
                "KPI alignment: "
                + json.dumps(payload.get("kpi_alignment"), ensure_ascii=False, default=str)
            )
        body = json.dumps(visible_rows, ensure_ascii=False, default=str, indent=2)
        text = "\n".join(header) + "\nExact query result rows:\n" + body
        if row_count > len(visible_rows):
            text += (
                f"\n[Context contains the first {len(visible_rows)} of {row_count} returned rows. "
                "The SQL and total row count above remain authoritative.]"
            )
        if len(text) > MAX_STRUCTURED_CONTEXT_CHARS:
            text = text[: MAX_STRUCTURED_CONTEXT_CHARS - 32] + "\n[structured evidence truncated]"

        return {
            "source_kind": "database" if self.config.dialect != "sqlite" else "tabular",
            "content_type": "query_result",
            "database": self.config.database,
            "dialect": self.config.dialect,
            "schema_file": Path(self.config.schema_path).name,
            "sql": payload.get("sql_used"),
            "row_count": row_count,
            "columns": columns,
            "join_path": payload.get("join_path") or [],
            "kpi_alignment": payload.get("kpi_alignment"),
            "warnings": payload.get("warnings") or [],
            "text": text,
        }

    def retrieve(self, question: str, *, schema_only: bool = False) -> dict:
        if not self.available:
            return {
                "status": "unavailable",
                "message": "No usable structured source is connected.",
                "items": [],
            }

        if schema_only:
            try:
                item = self._schema_evidence()
                return {
                    "status": "supported",
                    "items": [item],
                    "database": self.config.database,
                    "dialect": self.config.dialect,
                }
            except Exception as exc:
                return {
                    "status": "service_unavailable",
                    "message": f"Structured schema could not be read: {type(exc).__name__}.",
                    "items": [],
                }

        agent = None
        try:
            agent = self.agent_factory(
                schema_json_path=self.config.schema_path,
                db_uri=self.config.db_uri,
                provider=self.provider,
                dialect=self.config.dialect,
            )
            output_path = Path(self.config.processed_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = agent.process_goal(question, output_path=str(output_path))

            # The production GoalAgent returns the output path.  Test/custom
            # implementations may return the payload directly.
            if isinstance(result, dict):
                payload = result
            else:
                candidate = Path(str(result or output_path))
                if not candidate.exists():
                    candidate = output_path
                if not candidate.exists():
                    raise RuntimeError("Goal Agent did not produce processed data")
                payload = json.loads(candidate.read_text(encoding="utf-8"))

            status = str(payload.get("status") or "success").lower()
            if status in {"needs_clarification", "clarification", "ambiguous"}:
                return {
                    "status": "clarification",
                    "message": (
                        payload.get("message")
                        or payload.get("question")
                        or "Please clarify the structured-data question."
                    ),
                    "items": [],
                    "payload": payload,
                }

            if status in {"query_failed", "failed", "error"}:
                return {
                    "status": "insufficient",
                    "message": payload.get("message") or "The structured query could not be completed safely.",
                    "items": [],
                    "payload": payload,
                }

            item = self._result_evidence(payload)
            return {
                "status": "supported",
                "items": [item],
                "sql": payload.get("sql_used"),
                "row_count": item["row_count"],
                "columns": item["columns"],
                "warnings": payload.get("warnings") or [],
                "payload": payload,
            }
        except Exception as exc:
            return {
                "status": "service_unavailable",
                "message": f"Structured retrieval failed safely: {type(exc).__name__}.",
                "items": [],
            }
        finally:
            engine = getattr(agent, "engine", None) if agent is not None else None
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
