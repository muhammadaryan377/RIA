"""Component: parses the generated SQL into the structured output contract and graceful failure envelope (spec section 9)."""

import difflib
import json
import re
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from llm_provider import LLMProvider, create_provider
from core.validation import unknown_sql_tables, empty_sql_tables, unknown_sql_columns
from core.config import SCHEMA_DIR, BASE_DIR
from schema_engine.lexical import singularize

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None


class SqlContractMixin:
    # ---- _sql_alias_map (_sql_alias_map) ----

    def _sql_alias_map(self, sql):
        """Resolve every token usable in a qualified reference to its table."""
        alias_map = {}
        struct = self._sql_structure(sql)
        alias_map.update({a.lower(): t for a, t in struct["alias_to_name"].items()})
        alias_map.update({t.lower(): t for t in struct["in_from"]})
        return alias_map

    # ---- _where_clause (_where_clause) ----

    def _where_clause(self, sql):
        """Return the top-level WHERE clause text (best-effort, paren-aware)."""
        if not sql:
            return ""
        depth = 0
        n = len(sql)
        start = None
        i = 0
        while i < n:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if depth == 0:
                m = re.match(r"\bWHERE\b", sql[i:], re.IGNORECASE)
                if m:
                    start = i + m.end()
                    break
            i += 1
        if start is None:
            return ""
        i = start
        depth = 0
        while i < n:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ";":
                return sql[start:i]
            elif depth == 0:
                k = re.match(r"\b(GROUP|ORDER|HAVING|LIMIT|OFFSET|UNION)\b", sql[i:], re.IGNORECASE)
                if k:
                    return sql[start:i]
            i += 1
        return sql[start:]

    # ---- _split_top_level_and (_split_top_level_and) ----

    def _split_top_level_and(self, where):
        """Split a WHERE clause on top-level AND (paren-aware, case-insensitive)."""
        parts = []
        depth = 0
        n = len(where)
        last = 0
        i = 0
        while i < n:
            ch = where[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and where[i:i + 3].upper() == "AND":
                left_ok = i == 0 or not where[i - 1].isalnum()
                right_ok = i + 3 >= n or not where[i + 3].isalnum()
                if left_ok and right_ok:
                    parts.append(where[last:i])
                    i += 3
                    last = i
                    continue
            i += 1
        parts.append(where[last:])
        return [p.strip() for p in parts if p.strip()]

    # ---- _extract_sql_filters (_extract_sql_filters) ----

    def _extract_sql_filters(self, sql):
        """Best-effort extraction of WHERE filters from the final SQL (spec §12:
        the contract filters must reflect the validated query, never [] when the
        SQL actually filters)."""
        filters = []
        where = self._where_clause(sql)
        if not where:
            return filters
        for part in self._split_top_level_and(where):
            m = re.match(
                r"\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*"
                r"(=|!=|<>|>=|<=|>|<|LIKE|ILIKE|IS\s+NOT|IS)\s+(.+?)\s*$",
                part, re.IGNORECASE,
            )
            if m:
                filters.append({
                    "column": m.group(1),
                    "operator": m.group(2).upper(),
                    "value": m.group(3).strip().rstrip(";").strip(),
                })
        return filters

    # ---- _extract_sql_group_by (_extract_sql_group_by) ----

    def _extract_sql_group_by(self, sql):
        """GROUP BY columns from the final SQL (spec §12)."""
        if not sql:
            return []
        m = re.search(
            r"\bGROUP\s+BY\s+(.+?)(?=\s+(HAVING|ORDER|LIMIT|OFFSET)\b|;|$)",
            sql, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return []
        return [c.strip() for c in m.group(1).split(",") if c.strip()]

    # ---- _extract_sql_measures (_extract_sql_measures) ----

    def _extract_sql_measures(self, sql):
        """Aggregate expressions in the final SQL SELECT clause (spec §12)."""
        if not sql:
            return []
        m = re.search(r"\bSELECT\s+(.+?)(?=\s+FROM\b)", sql, re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        items = []
        cur, depth = "", 0
        for ch in m.group(1):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                items.append(cur)
                cur = ""
            else:
                cur += ch
        items.append(cur)
        measures = []
        for it in items:
            it = it.strip()
            if re.match(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", it, re.IGNORECASE):
                measures.append(it)
        return measures

    # ---- _extract_sql_joins (_extract_sql_joins) ----

    def _extract_sql_joins(self, sql):
        """Explicit joins used by the final SQL (spec §12)."""
        joins = []
        if not sql:
            return joins
        alias_map = self._sql_alias_map(sql)
        for (_start, _end, _table, on_start, on_end) in self._join_clauses(sql):
            if on_start is None:
                continue
            cond = sql[on_start:on_end]
            for eq in re.finditer(r"([A-Za-z_]\w*)\.(\w+)\s*=\s*([A-Za-z_]\w*)\.(\w+)", cond):
                a, ca, b, cb = eq.groups()
                joins.append({
                    "left_table": alias_map.get(a.lower(), a),
                    "left_column": ca,
                    "right_table": alias_map.get(b.lower(), b),
                    "right_column": cb,
                })
        return joins

    # ---- _parse_sql_plan (_parse_sql_plan) ----

    def _parse_sql_plan(self, sql):
        """Deterministically extract order_by and limit from the final SQL."""
        order_by, limit = [], None
        if not sql:
            return order_by, limit
        m = re.search(r"\bORDER\s+BY\s+(.+?)(?=\s+(LIMIT|GROUP\s+BY|HAVING|;)\s*|$)",
                      sql, re.IGNORECASE | re.DOTALL)
        if m:
            order_by = [col.strip() for col in m.group(1).split(",") if col.strip()]
        m = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        if m:
            limit = int(m.group(1))
        return order_by, limit

    # ---- _build_output_contract (_build_output_contract) ----

    def _build_output_contract(self, user_goal, kpi_map, join_path, final_sql,
                               cleaned_data, success=True, message=None,
                               status="success", warnings=None,
                               execution_time_ms=None):
        """Enrich processed_data.json with the structured Goal-Agent contract
        (goal intent, data selection, analysis plan) while keeping the existing
        flat fields the Insight Agent and UI already consume."""
        data_columns = list(cleaned_data[0].keys()) if cleaned_data else []
        order_by, limit = self._parse_sql_plan(final_sql)
        sql_filters = self._extract_sql_filters(final_sql or "")
        sql_group_by = self._extract_sql_group_by(final_sql or "")
        sql_joins = self._extract_sql_joins(final_sql or "")
        sql_measures = self._extract_sql_measures(final_sql or "")

        joins = []
        if sql_joins:
            for j in sql_joins:
                joins.append(
                    f"{j['left_table']}.{j['left_column']} -> "
                    f"{j['right_table']}.{j['right_column']}"
                )
        else:
            for a, b in zip(join_path, join_path[1:]):
                joins.append(f"{a} -> {b}")

        return {
            "status": status,
            "goal": {
                "original_question": self._active_original_goal or user_goal,
                "intent": self._detect_intent(user_goal),
                "analysis_type": self._detect_analysis_type(user_goal),
            },
            "validation": {
                "structural": bool(final_sql and self._sql_structure(final_sql)["in_from"]),
                "semantic": bool(final_sql),
                "relationship": not bool(
                    [w for w in (warnings or []) if "Relationship" in w]
                ),
            },
            "data_selection": {
                "tables": list(join_path),
                "columns": data_columns,
                "filters": sql_filters,
                "joins": joins,
            },
            "analysis_plan": {
                "measures": sql_measures or [k["description"] for k in kpi_map.get("kpis", [])],
                "dimensions": kpi_map.get("dimensions", []),
                "aggregation": sorted(
                    {m.split("(", 1)[0].strip().upper() for m in sql_measures}
                ) or sorted({k["aggregate"] for k in kpi_map.get("kpis", [])}),
                "group_by": sql_group_by or kpi_map.get("dimensions", []),
                "order_by": order_by,
                "limit": limit,
            },
            "sql": final_sql,
            "execution": {
                "success": bool(success),
                "row_count": len(cleaned_data),
                "execution_time_ms": execution_time_ms,
            },
            "data": cleaned_data,
            "suggested_questions": self._template_suggestions(3),
            "warnings": warnings or [],
            "message": message,
        }

    # ---- _graceful_failure (_graceful_failure) ----

    def _graceful_failure(self, user_goal, output_path, exc):
        """Write a valid processed_data.json explaining WHY the goal could not
        be answered, so the endpoint never returns a 500 for a goal we could
        not answer (provider outage, unfixable SQL, database hiccup)."""
        exc_str = str(exc)
        # Sanitize: strip internal paths, connection strings, hostnames.
        
        sanitized = re.sub(r"(?:/[^/\s:]+){2,}", "[path]", exc_str)
        sanitized = re.sub(r"postgresql://[^\s]+", "[connection-string]", sanitized)
        sanitized = re.sub(r"mysql://[^\s]+", "[connection-string]", sanitized)
        sanitized = re.sub(r"sqlite:///[^\s]+", "[connection-string]", sanitized)
        sanitized = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "[host]", sanitized)

        if self._llm_unavailable(exc) or "429" in exc_str:
            message = (
                "The language model is temporarily unavailable or rate-limited. "
                "Please try again shortly or switch the model provider."
            )
        else:
            message = f"Could not answer this goal: {sanitized}"
        output = {
            "user_goal": user_goal,
            "kpi_alignment": {},
            "join_path": [],
            "sql_used": None,
            "row_count": 0,
            "data": [],
            "message": message,
            "preprocessing": {"notes": ["No data retrieved; the goal could not be answered."]},
            "missing_values_handled": "preserved (NULLs kept as-is)",
            "timestamp": datetime.now().isoformat(),
        }
        output.update(self._build_output_contract(
            user_goal, {"kpis": [], "dimensions": []}, [], None, [],
            success=False, message=message, status="query_failed",
        ))
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        logging.error("Goal Agent graceful failure for %r: %s", user_goal, exc)
        return output_path

