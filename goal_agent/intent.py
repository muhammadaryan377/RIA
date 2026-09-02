"""Component: intent parsing - goal IR, KPI mapping, clarification gate and metrics overview (spec sections 3, 7, 9)."""

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

logger = logging.getLogger(__name__)

class IntentMixin:
    # ---- _needs_clarification (_needs_clarification) ----

    def _needs_clarification(self, user_goal):
        """Return (True, question) only when the goal cannot be grounded to ANY
        table or column in the connected schema, even after normalization,
        synonym matching, singular/plural handling and analytical-intent
        checks (spec §7).

        The old behavior failed whenever no literal table/column name appeared
        in the question. That was too primitive: a generic analytical question
        ("top brands by model year", "overall sales") must NOT be rejected just
        because the user phrased a concept rather than a physical column.
        """
        tokens = self._normalize_tokens(user_goal)
        if not tokens:
            return True, (
                "Please provide a business question, for example "
                "'total sales by month' or 'top customers'."
            )

        goal_l = str(user_goal).lower()

        # 1) Table-level + column-level semantic matching (synonym-aware).
        tables = self._match_tables(goal_l)
        columns = []
        for _score, t in tables[:6]:
            columns += self._match_columns(goal_l, t)
        for t in self.tables:
            columns += self._match_columns(goal_l, t)

        if tables or columns:
            return False, None

        # 2) A genuinely analytical goal (metric concept + operation wording)
        #    must not be pre-blocked just because it does not name a physical
        #    column ("overall sales", "top brands by model year"). Only a vague
        #    goal with neither a grounded subject nor a resolvable metric
        #    concept + operation is ambiguous.
        if self._has_resolvable_analytical_structure(goal_l):
            return False, None

        # 3) Nothing at all matched: graceful clarification that names what was
        #    looked for, never a raw exception or matching failure.
        table_names = ", ".join(sorted(self.tables))[:300] if self.tables else "none"
        return True, (
            "I couldn't find a table or column in the connected database that "
            "clearly corresponds to your question. Available data covers: "
            f"{table_names}. Please rephrase using one of those areas (for "
            "example 'total sales by month' or 'top products')."
        )

    # ---- _clarify (_clarify) ----

    def _clarify(self, user_goal, question, output_path="processed_data.json"):
        """Write the spec §3 needs_clarification contract without running SQL."""
        output = {
            "status": "needs_clarification",
            "question": question,
            "goal": {
                "original_question": user_goal,
                "intent": self._detect_intent(user_goal),
                "analysis_type": self._detect_analysis_type(user_goal),
            },
            "data_selection": {"tables": [], "columns": [], "filters": [], "joins": []},
            "analysis_plan": {
                "measures": [], "dimensions": [], "aggregation": [],
                "group_by": [], "order_by": [], "limit": None,
            },
            "sql": None,
            "execution": {"success": False, "row_count": 0, "execution_time_ms": None},
            "data": [],
            "suggested_questions": [],
            "warnings": [],
            "message": question,
        }
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        return output_path

    # ---- _build_kpi_index (_build_kpi_index) ----

    def _build_kpi_index(self):
        """Keyword -> (aggregate, description) map used for KPI alignment.

        Spec §3: these are EXPLICIT wording hints only. A word is resolved only
        when the user actually says it ("top products by quantity"); it is never
        a silently applied default metric. Generic measure words (quantity,
        amount, value, price) are listed so explicit wording grounds even on
        schemas with no 'sales' column.
        """
        return {
            "total": ("SUM", "aggregate total"),
            "sum": ("SUM", "aggregate total"),
            "amount": ("SUM", "aggregate total"),
            "value": ("SUM", "aggregate total"),
            "quantity": ("SUM", "aggregate total"),
            "qty": ("SUM", "aggregate total"),
            "units": ("SUM", "aggregate total"),
            "price": ("SUM", "aggregate total"),
            "cost": ("SUM", "aggregate total"),
            "count": ("COUNT", "row count"),
            "how many": ("COUNT", "row count"),
            "average": ("AVG", "average value"),
            "avg": ("AVG", "average value"),
            "mean": ("AVG", "average value"),
            "max": ("MAX", "maximum value"),
            "maximum": ("MAX", "maximum value"),
            "min": ("MIN", "minimum value"),
            "minimum": ("MIN", "minimum value"),
            "profit": ("SUM", "profit KPI"),
            "margin": ("SUM", "profit KPI"),
            "revenue": ("SUM", "revenue KPI"),
            "sales": ("SUM", "sales KPI"),
            "growth": ("AVG", "growth KPI"),
        }

    # ---- _detect_metrics_overview (_detect_metrics_overview) ----

    def _detect_metrics_overview(self, user_goal):
        """Return the entity table to summarize when the goal is an open-ended
        'most important metrics for X' style question, else None.

        Deterministic: no LLM. Concrete goals ('total sales by customer') never
        match because they do not use metric/overview vocabulary.
        """
        if not re.search(self._OVERVIEW_RE, user_goal):
            return None
        goal_tokens = self._normalize_tokens(user_goal)
        for table_name in self._get_relevant_tables(user_goal, max_tables=3):
            if self._natural_label(table_name).replace(" ", "") in goal_tokens:
                return table_name
            if table_name.lower() in goal_tokens:
                return table_name
        return None

    # ---- _build_metrics_overview (_build_metrics_overview) ----

    def _build_metrics_overview(self, entity_table):
        """Deterministic metric set for an entity: entity count + one count and
        one sum per fact table that references it. Returns [(label, sql)]."""
        entity_label = self._natural_label(entity_table)
        metrics = [
            (f"Total {self._plural_label(entity_table)}",
             f'SELECT COUNT(*) FROM "{entity_table}"'),
        ]
        for table_name, info in self.tables.items():
            if table_name == entity_table:
                continue
            for fk in info.get("foreign_keys", []):
                if fk["referenced_table"] != entity_table:
                    continue
                t_label = self._natural_label(table_name)
                metrics.append(
                    (f"Total {self._plural_label(table_name)}",
                     f'SELECT COUNT(*) FROM "{table_name}"')
                )
                measure_col = self._pick_measure_column(table_name)
                if measure_col:
                    col_label = self._natural_label(measure_col)
                    if col_label in self._GENERIC_TOTAL_WORDS:
                        label = f"Total {t_label} value"
                    else:
                        label = f"Total {col_label}"
                    metrics.append(
                        (label, f'SELECT COALESCE(SUM("{measure_col}"), 0) '
                                f'FROM "{table_name}"')
                    )
                break
        return metrics

    # ---- _process_metrics_overview (_process_metrics_overview) ----

    def _process_metrics_overview(self, user_goal, entity_table, output_path):
        """Answer an open-ended 'most important metrics for X' goal with a
        deterministic, always-working overview instead of a clarification."""
        entity_label = self._natural_label(entity_table)
        metrics = self._build_metrics_overview(entity_table)
        rows, sql_parts = [], []
        t0 = time.time()
        try:
            with self.engine.connect() as conn:
                for label, sql in metrics:
                    try:
                        value = conn.execute(text(sql)).scalar()
                        rows.append({"metric": label, "value": value})
                        sql_parts.append(sql)
                    except Exception as exc:
                        logging.warning(
                            "Overview metric failed (%s): %s", label, exc
                        )
        except Exception as exc:
            return self._graceful_failure(user_goal, output_path, exc)
        execution_ms = int((time.time() - t0) * 1000)
        if not rows:
            return self._graceful_failure(
                user_goal, output_path,
                RuntimeError("Could not compute an overview for this goal."),
            )

        output = {
            "user_goal": self._active_original_goal or user_goal,
            "kpi_alignment": {
                "kpis": [{"aggregate": None, "description": "metrics overview",
                          "match": None}],
                "dimensions": [entity_table],
            },
            "join_path": [entity_table],
            "sql_used": "; ".join(sql_parts) or None,
            "row_count": len(rows),
            "data": rows,
            "message": f"Here's an overview of the most important metrics "
                       f"for {entity_label}.",
            "preprocessing": {"notes": ["Deterministic overview (no LLM)."]},
            "missing_values_handled": "not applicable (counts/sums)",
            "timestamp": datetime.now().isoformat(),
        }
        output.update(self._build_output_contract(
            user_goal, output["kpi_alignment"], [entity_table],
            sql_parts[0] if sql_parts else None, rows,
            success=True, message=output["message"], status="success",
            warnings=self._typo_warnings, execution_time_ms=execution_ms,
        ))
        output["sql"] = sql_parts
        output["data_selection"]["columns"] = [r["metric"] for r in rows]
        output["analysis_plan"]["measures"] = [r["metric"] for r in rows]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Goal Agent overview done! Saved to %s", output_path)
        return output_path

    # ---- map_goal_to_kpi (map_goal_to_kpi) ----

    def map_goal_to_kpi(self, user_goal):
        """Return detected KPIs + dimensions for a user goal."""
        goal_lower = user_goal.lower()
        kpis = []
        dimensions = set()

        for keyword, (agg, label) in self.kpi_index.items():
            if keyword in goal_lower:
                kpis.append({"aggregate": agg, "description": label, "match": keyword})
        if not kpis:
            kpis = [{"aggregate": "SUM", "description": "general KPI", "match": None}]

        for table_name in self.tables:
            if table_name.lower() in goal_lower:
                dimensions.add(table_name)

        return {"kpis": kpis, "dimensions": sorted(dimensions)}

    # ---- _enrich_kpi_map (_enrich_kpi_map) ----

    def _enrich_kpi_map(self, kpi_map, ir):
        """Bind weak KPI hints to real schema columns (spec §17): the KPI map
        must never rely on a naive universal mapping — every hint that resolves
        to a measure column records the resolved expression/aggregation. The
        'general KPI' fallback binds to the first resolved IR metric so an
        inferred COUNT or a computed measure still reaches the SQL prompt."""
        if not ir:
            return kpi_map
        resolved = []
        for m in ir.get("metrics", []):
            if m.get("resolved_expression") or m.get("resolved_column"):
                resolved.append(m)
        by_concept = {m["concept"]: m for m in resolved}
        for k in kpi_map.get("kpis", []):
            r = by_concept.get(k["description"])
            if not r and resolved and k.get("match") is None:
                r = resolved[0]
            if r:
                k["resolved_column"] = r.get("resolved_expression") or r.get("resolved_column")
                if r.get("aggregation"):
                    k["aggregate"] = r["aggregation"]
                if r.get("computed_measure"):
                    k["computed_measure"] = True
                if r.get("inferred"):
                    k["inferred"] = True
        return kpi_map

    # ---- _detect_aggregation (_detect_aggregation) ----

    def _detect_aggregation(self, goal):
        g = str(goal).lower()
        if re.search(r"\b(total|sum|amount of|sum of)\b", g):
            return "SUM"
        if re.search(r"\b(count|how many|number of|total number)\b", g):
            return "COUNT"
        if re.search(r"\b(average|avg|mean)\b", g):
            return "AVG"
        if re.search(r"\b(max|maximum|largest|biggest)\b", g):
            return "MAX"
        if re.search(r"\b(min|minimum|smallest)\b", g):
            return "MIN"
        return None

    # ---- _detect_ranking (_detect_ranking) ----

    def _detect_ranking(self, goal):
        """Best-effort ranking extraction (top/bottom N, or a one-sided order)."""
        g = str(goal).lower()
        direction = "asc" if any(k in g for k in
                                 ("bottom", "lowest", "least", "worst")) else "desc"
        top_m = re.search(r"\b(top|bottom)\s+(\d+)\b", g)
        if top_m:
            return {"by": None, "direction": direction, "limit": int(top_m.group(2))}
        if any(k in g for k in ("top", "best", "highest", "most", "leader",
                                "worst", "lowest", "least", "bottom")):
            return {"by": None, "direction": direction, "limit": None}
        return None

    # ---- _extract_goal_time (_extract_goal_time) ----

    def _extract_goal_time(self, goal):
        """Time constraints mentioned by the user (year / period granularity)."""
        g = str(goal).lower()
        year = re.search(r"\b(19\d{2}|20\d{2})\b", g)
        if year:
            return {"constraint": "year", "value": int(year.group(1)),
                    "column": None, "granularity": None}
        for period in ("monthly", "quarterly", "yearly", "daily", "weekly"):
            if period in g:
                return {"constraint": "period", "value": None, "column": None,
                        "granularity": period[:-2]}
        return None

    # ---- _extract_goal_filters (_extract_goal_filters) ----

    def _extract_goal_filters(self, goal):
        """Conservative deterministic filter extraction (spec §11). Only
        constraints we can name without inventing data are captured; the rest
        is left to the SQL and mirrored back from it in the contract."""
        filters = []
        year = re.search(r"\b(19\d{2}|20\d{2})\b", goal)
        if year:
            y = int(year.group(1))
            filters.append({"column": None, "operator": ">=", "value": f"{y}-01-01"})
            filters.append({"column": None, "operator": "<=", "value": f"{y}-12-31"})
        return filters

    # ---- _goal_asks_for_comparison (_goal_asks_for_comparison) ----

    def _goal_asks_for_comparison(self, goal):
        g = str(goal).lower()
        return any(k in g for k in ("compare", "comparison", "versus",
                                    "difference between", "among"))

    # ---- _build_goal_ir (_build_goal_ir) ----

    def _build_goal_ir(self, goal):
        """Build a canonical, serializable Goal IR (spec §1/§2) BEFORE any SQL.

        The IR is intentionally deterministic (no LLM): intent, aggregation,
        metric concepts, dimensions (concepts only), filters, time, ranking and
        comparison. `_resolve_semantics` later binds the concepts to real schema
        columns/tables so the LLM is no longer the authority on the plan.
        """
        goal_l = str(goal).lower()
        ir = {
            "original_goal": goal,
            "intent": self._detect_intent(goal),
            "analysis_type": self._detect_analysis_type(goal),
            "operation": None,
            "aggregation": None,
            "metrics": [],
            "measures": [],
            "dimensions": [],
            "filters": [],
            "time": None,
            "ranking": None,
            "comparison": None,
            "ordering": [],
            "required_tables": [],
            "join_plan": [],
            "confidence": 0.5,
            "requires_sql": True,
        }
        agg = self._detect_aggregation(goal)
        if agg:
            ir["aggregation"] = agg
        seen_metrics = set()
        for keyword, (aggregate, description) in self.kpi_index.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", goal_l):
                metric_key = (description, aggregate)
                if metric_key not in seen_metrics:
                    seen_metrics.add(metric_key)
                    ir["metrics"].append({
                        "concept": description,
                        "aggregation": None if keyword == "growth" else aggregate,
                        "expression": None,
                        "resolved_expression": None,
                        "resolved_table": None,
                        "resolved_column": None,
                    })
        if any(k in goal_l for k in ("growth", "trend", "over time", "increase",
                                     "decrease", "change")):
            ir["intent"] = "trend"
            ir["analysis_type"] = "trend_analysis"
        ir["ranking"] = self._detect_ranking(goal)
        ir["time"] = self._extract_goal_time(goal)
        ir["filters"] = self._extract_goal_filters(goal)
        if self._goal_asks_for_comparison(goal):
            ir["comparison"] = {"dimension": None, "segments": []}

        # Spec §3: NEVER assume the metric, but when a ranking/aggregation goal
        # states NO measure at all ("top brands by model year"), a COUNT of the
        # related records is a safe generic default. It must be represented as
        # an INFERRED metric with low-medium confidence, never as certainty.
        # The schema-aware metric resolver may still upgrade it to a real
        # measure column when one clearly exists.
        if not ir["metrics"] and ir["intent"] in (
                "ranking", "aggregation", "summary", "distribution", "trend"):
            ir["metrics"].append({
                "concept": "row count",
                "aggregation": "COUNT",
                "expression": None,
                "resolved_expression": None,
                "resolved_table": None,
                "resolved_column": None,
                "inferred": True,
                "confidence": 0.6,
                "reason": "no explicit measure; ranked/aggregated by count of "
                          "related records",
            })

        # Spec §9: a database-independent operation label for the contract.
        operation = ir["intent"]
        if ir["ranking"]:
            operation = "ranking"
        elif ir["comparison"]:
            operation = "comparison"
        elif ir["aggregation"] == "COUNT":
            operation = "count"
        elif ir["aggregation"]:
            operation = ir["aggregation"].lower()
        elif ir["intent"] == "trend":
            operation = "trend"
        ir["operation"] = operation
        return ir

    # ---- _detect_intent (_detect_intent) ----

    def _detect_intent(self, user_goal):
        """Heuristic intent classification (no LLM call) for the goal contract."""
        g = str(user_goal).lower()
        if re.search(r"\b(top\b|best\b|worst\b|highest\b|lowest\b|most\b|least\b|rank\b|leader\b|bottom\b)", g):
            return "ranking"
        if re.search(r"\b(compare|comparison|versus|difference|among|between)\b", g):
            return "comparison"
        if re.search(r"\b(trend|over\s+time|monthly|daily|weekly|yearly|growth|decline|increase|decrease|change)\b", g):
            return "trend"
        if re.search(r"\b(distribution|breakdown|per\b|by\b|count)\b", g):
            return "distribution"
        return "summary"

    # ---- _detect_analysis_type (_detect_analysis_type) ----

    def _detect_analysis_type(self, user_goal):
        """Heuristic analysis-type classification for the goal contract."""
        g = str(user_goal).lower()
        if any(k in g for k in ("predict", "forecast", "future", "expected",
                                "will ")):
            return "predictive"
        if any(k in g for k in ("why", "cause", "reason", "driver", "because")):
            return "diagnostic"
        if any(k in g for k in ("recommend", "should", "action", "improve",
                                "optimize", "suggest")):
            return "prescriptive"
        return "descriptive"

