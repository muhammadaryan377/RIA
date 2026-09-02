"""Component: schema grounding - relevant tables, join path and semantic resolution of subjects, dimensions and metrics (spec section 6)."""

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


class GroundingMixin:
    # ---- _table_keyword_score (_table_keyword_score) ----

    def _table_keyword_score(self, table_name, info, user_goal_tokens):
        """Score how strongly a table matches the goal words.

        Name matches dominate (synonym/plural-aware). Column matches are only a
        small bonus ON TOP of a name hit; a bare column match (e.g. the generic
        word 'company' hitting `company_name` on every table) must never outrank
        a table whose name actually matches the subject.
        """
        name_parts = set(self._normalize_tokens(table_name))
        name_hit = False
        for tok in user_goal_tokens:
            for part in name_parts:
                if self._schema_token_matches_goal(part, tok):
                    name_hit = True
                    break
            if name_hit:
                break
        score = 0
        if name_hit:
            # Strong: the table's name means something mentioned in the goal.
            # An EXACT name token outranks a partial/synonym match so "orders"
            # picks `orders`, not `order_details` (which only matches because
            # "order" is a prefix of "order_detail").
            for tok in user_goal_tokens:
                matched = False
                for part in name_parts:
                    if tok == part or (tok.endswith("s") and tok[:-1] == part):
                        score += 55
                        matched = True
                        break
                if matched:
                    continue
                for part in name_parts:
                    if self._schema_token_matches_goal(part, tok):
                        score += 40
                        break
        for column_name in info.get("columns", {}).keys():
            col_parts = set(self._normalize_tokens(column_name))
            for tok in user_goal_tokens:
                for part in col_parts:
                    if self._schema_token_matches_goal(part, tok):
                        # Column matches are a bonus only on a name-matched
                        # table; otherwise they are a weak hint that must not
                        # outrank a subject-table name match.
                        score += 8 if name_hit else 3
                        break
        return score

    # ---- _expand_with_neighbors (_expand_with_neighbors) ----

    def _expand_with_neighbors(self, selected, limit=4):
        """Expand the selected tables through FK neighbors, depth-bounded so
        unrelated tables further out in the graph never get pulled in.
        Deterministic: neighbors are ordered by FK connectivity (more central
        tables first) then alphabetically, never by set/dict insertion order.
        Only fills up to `limit` tables so a small relevant set is not drowned
        by the whole graph."""
        expanded = set(selected)
        queue = [(t, 0) for t in selected]
        visited = set()
        while queue and len(expanded) < limit:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if depth >= 2:
                continue
            neighbors = self.relationship_graph.get(current, [])
            neighbors = sorted(
                set(neighbors),
                key=lambda n: (-len(self.tables[n].get("foreign_keys", [])) if n in self.tables else 0, n),
            )
            for neighbor in neighbors:
                if len(expanded) >= limit:
                    break
                if neighbor in self.tables and neighbor not in expanded:
                    expanded.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return sorted(
            list(expanded),
            key=lambda n: (-len(self.tables[n].get("foreign_keys", [])) if n in self.tables else 0, n),
        )

    # ---- _get_relevant_tables (_get_relevant_tables) ----

    def _get_relevant_tables(self, user_goal, max_tables=3):
        goal_tokens = self._normalize_tokens(user_goal)
        if not goal_tokens:
            goal_tokens = {"data", "summary", "report"}

        ranked = []
        for table_name, info in self.tables.items():
            if info.get("empty"):
                # An empty table (0 rows) can never contribute data to an
                # answer; selecting it as relevant forces an unsatisfiable
                # join into the semantic plan and degrades every goal that
                # mentions its name.
                continue
            score = self._table_keyword_score(table_name, info, goal_tokens)
            ranked.append((table_name, score))
        # Deterministic ordering: ties resolved by FK connectivity (more central
        # tables win) then alphabetically, never by dict insertion order.
        ranked.sort(key=lambda x: (-x[1], -len(self.tables[x[0]].get("foreign_keys", [])), x[0]))
        selected = [name for name, _ in ranked if _ > 0][:max_tables]

        if not selected:
            fallback = [name for name, _ in self.tables.items() if not self.tables[name].get("empty")]
            fallback.sort(key=lambda x: (-len(self.tables[x].get("foreign_keys", [])), x))
            selected = fallback[:max_tables] if fallback else []

        if len(selected) < max_tables:
            selected = self._expand_with_neighbors(selected, max_tables)

        return selected[:max_tables]

    # ---- _find_join_hops (_find_join_hops) ----

    def _find_join_hops(self, start, target):
        """BFS join path between two tables using the relationship graph."""
        if start == target:
            return [start]
        queue = [[start]]
        visited = {start}
        while queue:
            path = queue.pop(0)
            for neighbor in self.relationship_graph.get(path[-1], []):
                if neighbor not in self.tables:
                    continue
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    # ---- _determine_join_path (_determine_join_path) ----

    def _determine_join_path(self, relevant_tables):
        """
        Return the ordered join path that connects all relevant tables, using
        the schema's relationship graph. Falls back to the table list itself.
        """
        if not relevant_tables:
            return []
        path = [relevant_tables[0]]
        remaining = relevant_tables[1:]
        while remaining:
            connected = None
            best_local = None
            for table in remaining:
                hop = self._find_join_hops(path[-1], table)
                if hop:
                    best_local = hop
                    connected = table
                    break
            if best_local:
                for node in best_local[1:]:
                    if node not in path:
                        path.append(node)
                remaining = [t for t in remaining if t != connected]
            else:
                # No path found; append remaining tables in order as a fallback.
                path.extend([t for t in remaining if t not in path])
                break
        return path

    # ---- _pick_measure_column (_pick_measure_column) ----

    def _pick_measure_column(self, table_name):
        """First numeric, non-key, non-id, non-FK column (best-guess measure)."""
        info = self.tables[table_name]
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        for col in info["columns"]:
            low = col.lower()
            if col in pk or col in fk_cols or low == "id" \
                    or low.endswith("_id") or low.endswith("id"):
                continue
            if self._classify_column(table_name, col) == "numeric":
                return col
        return None

    # ---- _first_date_column (_first_date_column) ----

    def _first_date_column(self, tables):
        for table in tables:
            if table not in self.tables:
                continue
            for col_name, col_info in self.tables[table].get("columns", {}).items():
                dt = (col_info.get("data_type") or "").lower()
                if any(k in dt for k in ("date", "time", "timestamp")) or "date" in col_name.lower():
                    return (table, col_name)
        return None

    # ---- _is_measure_column (_is_measure_column) ----

    def _is_measure_column(self, table, col):
        """True when a column is a numeric, non-key, non-FK measure candidate."""
        info = self.tables.get(table, {})
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        if col in pk or col in fk_cols:
            return False
        dt = str(info.get("columns", {}).get(col, {}).get("data_type", "")).lower()
        if any(k in dt for k in (
                "int", "decimal", "float", "numeric", "double",
                "money", "real", "bigint", "smallint")):
            return True
        return False

    # ---- _pick_label_column (_pick_label_column) ----

    def _pick_label_column(self, table):
        """Best human-readable grouping/display column of a table (the first
        text, non-key column; otherwise any non-key column)."""
        info = self.tables.get(table, {})
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        text_col = None
        for col in info.get("columns", {}):
            low = col.lower()
            if col in pk or col in fk_cols or low.endswith("_id") or low == "id":
                continue
            if self._classify_column(table, col) == "text" and not text_col:
                text_col = col
            if "name" in low or "label" in low or "title" in low:
                return col
        return text_col

    # ---- _is_dimension_like (_is_dimension_like) ----

    def _is_dimension_like(self, table, col):
        """A column is a breakdown dimension (even when numeric) if its name
        suggests a category/time attribute such as model_year or order_status.
        This keeps quantity/price-style numeric columns as measures while still
        grouping by model_year on any schema."""
        if not self._is_measure_column(table, col):
            return True
        low = col.lower()
        return any(k in low for k in self._DIMENSION_LIKE_NUMERIC)

    # ---- _match_dimension_columns (_match_dimension_columns) ----

    def _match_dimension_columns(self, goal_l, table):
        """Columns of `table` that the goal means AND that are legitimate
        breakdown columns (excludes PK / FK / *_id and pure-measure numeric
        columns). Used for dimension grounding and for choosing the fact table
        of an inferred COUNT.

        Metric/analytical concept words ('sales', 'value', 'average', ...) are
        stripped BEFORE column matching: in 'average order value per customer'
        the word 'value' (or 'sales' -> synonym 'order') must not bind
        order_date / ship_name as a dimension and make the SQL over-group."""
        info = self.tables.get(table, {})
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        pk = set(info.get("primary_key", []) or [])
        goal_tokens = self._normalize_tokens(goal_l)
        # Don't bind metric words, analytical words, or (when the goal carries
        # a metric) bare count/fact nouns like "order"/"sales" — "average ORDER
        # value per customer" must not group by order_date, and "by ORDER STATUS"
        # should bind the status column, not order_date.
        metric_present = bool(goal_tokens & self._METRIC_CONCEPT_WORDS)
        skip = set(self._ANALYTICAL_GOAL_WORDS) | set(self._METRIC_CONCEPT_WORDS)
        if metric_present:
            skip |= self._COUNT_OBJECT_TOKENS
        filtered = " ".join(t for t in goal_tokens if t not in skip)
        out = []
        for score, col in self._match_columns(filtered or goal_l, table):
            low = col.lower()
            if col in fk_cols or col in pk or low == "id" or low.endswith("_id"):
                continue
            if not self._is_dimension_like(table, col):
                continue
            out.append((score, col))
        return out

    # ---- _pick_fact_table (_pick_fact_table) ----

    def _pick_fact_table(self, goal_l, relevant, subject):
        """Best table to COUNT / aggregate when the metric is inferred
        (spec §3 / §10).

        Rules (generic, schema-driven):
          - if the SUBJECT itself carries the requested dimension columns,
            count the subject ("how many customers");
          - else if the goal EXPLICITLY names a child table (word in the goal
            matches the child's table name: "shipping company with most
            ORDERS" -> count orders, not order_details), prefer that child;
          - else if a CHILD table that references the subject carries the
            requested dimension columns, count there ("top brands by model
            year" -> count products, because model_year lives in products);
          - else fall back to the subject.
        """
        if not subject:
            return relevant[0] if relevant else None
        if self._match_dimension_columns(goal_l, subject):
            return subject
        goal_tokens = self._normalize_tokens(goal_l)
        best_child = None
        for t in relevant:
            if t == subject:
                continue
            info = self.tables.get(t, {})
            child_of_subject = any(
                fk.get("referenced_table") == subject
                for fk in info.get("foreign_keys", [])
            )
            if not child_of_subject:
                continue
            name_parts = set(self._normalize_tokens(t))
            named_in_goal = any(
                any(tok == part or (tok.endswith("s") and tok[:-1] == part)
                    for part in name_parts)
                for tok in goal_tokens
            )
            if named_in_goal:
                return t
            if self._match_dimension_columns(goal_l, t):
                best_child = t
                break
        return best_child or subject

    # ---- _find_price_quantity_expression (_find_price_quantity_expression) ----

    def _find_price_quantity_expression(self, relevant, concept_tokens):
        """Spec §3: when no single measure column matches a sales-like concept,
        look for a fact table with BOTH a quantity and a unit-price column and
        synthesize quantity * unit_price. Generic, schema-driven, no DB names."""
        for table in relevant:
            info = self.tables.get(table, {})
            cols = list(info.get("columns", {}))
            qty = next((c for c in cols if self._schema_token_matches_goal("quantity", c)
                        or "qty" in c.lower()), None)
            price = next((c for c in cols if self._schema_token_matches_goal("price", c)
                          or "price" in c.lower() or "amount" in c.lower()), None)
            if qty and price and qty != price:
                if self._is_measure_column(table, qty) and self._is_measure_column(table, price):
                    return {
                        "table": table,
                        "expression": f"{table}.{qty} * {table}.{price}",
                        "quantity": qty,
                        "price": price,
                    }
        return None

    # ---- _is_pure_fact_table (_is_pure_fact_table) ----

    def _is_pure_fact_table(self, table):
        """True for a line-item/fact table: every non-key column is a measure
        (order_details, transaction_items, ...). Such tables are measures, never
        a breakdown subject, and must not contribute label dimensions."""
        info = self.tables.get(table, {})
        cols = [c for c in info.get("columns", {})
                if not (c.lower().endswith("_id") or c.lower() == "id")]
        if not cols:
            return True
        return all(self._is_measure_column(table, c) for c in cols)

    # ---- _pick_subject (_pick_subject) ----

    _COUNT_OBJECT_TOKENS = frozenset({
        "order", "orders", "sale", "sales", "transaction", "transactions",
        "quantity", "qty", "units", "unit", "amount", "price", "cost",
        "revenue", "income", "turnover", "profit", "margin",
    })

    _QUANTITY_COL_TOKENS = frozenset({
        "quantity", "qty", "units", "unit", "count", "num",
    })

    # Column-name preference tiers for measure resolution. When match scores
    # tie, higher-tier (KPI total) columns win so "overall sales" binds the
    # sales.amount total column rather than a per-line quantity/price column.
    _MONETARY_TOTAL_COLS = frozenset({
        "amount", "total", "revenue", "sales",
    })
    _MONETARY_LINE_COLS = frozenset({
        "price", "cost", "value", "margin", "income",
    })

    def _pick_subject(self, matched_tables, goal_tokens, entity_tokens):
        """Choose the subject (entity) table and the ordered list of entity
        (subject) tables from the matched tables.

        Entity tables are those whose NAME matches an entity token
        (non-metric, non-analytical; minus count/fact nouns when a metric is
        present). Pure-fact tables (order_details, ...) are never subjects.
        Falls back to non-fact matched tables, then the strongest name match."""
        def name_matches_entity(table):
            parts = set(self._normalize_tokens(table))
            return any(
                self._schema_token_matches_goal(part, tok)
                for tok in entity_tokens for part in parts
            )
        subject_tables = [
            mt for mt in matched_tables
            if not self._is_pure_fact_table(mt[1]) and name_matches_entity(mt[1])
        ]
        if not subject_tables:
            subject_tables = [
                mt for mt in matched_tables if not self._is_pure_fact_table(mt[1])
            ]
        if not subject_tables:
            subject_tables = matched_tables
        def key(mt):
            t = mt[1]
            return (
                self._table_keyword_score(t, self.tables[t], goal_tokens),
                -len(self.tables[t].get("foreign_keys", [])),
                t,
            )
        best = max(subject_tables, key=key)
        return best[1], subject_tables

    def _explicit_breakdown_tokens(self, goal_l):
        """Goal tokens that name an explicit breakdown dimension via a
        'by X / per X / for each X' phrase. Empty when the goal has no explicit
        breakdown (so only the subject label becomes a dimension)."""
        goal_l = str(goal_l).lower()
        tokens = self._normalize_tokens(goal_l)
        breakdown = set()
        for m in re.finditer(r"\b(?:by|per)\s+([a-z][a-z0-9_\s]+)", goal_l):
            for w in m.group(1).split():
                breakdown.add(w.rstrip("s") if w.endswith("s") else w)
        if re.search(r"\bfor each\b|\bfor every\b", goal_l):
            # tokens after "for each" up to a punctuation/keyword
            for m in re.finditer(r"for each\s+([a-z][a-z0-9_\s]+)", goal_l):
                for w in m.group(1).split():
                    breakdown.add(w.rstrip("s") if w.endswith("s") else w)
        # Intersect with what the goal actually contains so we never invent words.
        return breakdown & tokens

# ---- _resolve_semantics (_resolve_semantics) ----

    def _resolve_semantics(self, goal, ir, join_path):
        """Bind the Goal IR concepts to real schema columns/tables (spec §3).

        Two-layer and database-agnostic:
          1. Generic intent lives in the IR (no schema names).
          2. Schema grounding happens HERE: subjects, dimensions and metrics are
             resolved with synonym-aware matching (customers <-> clients,
             model year <-> model_year, ...) plus an explicit confidence and a
             human reason for every binding (spec §6). Never fabricates a
             mapping; unresolved concepts stay None and confidence drops.
        """
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            relevant = [list(self.tables.keys())[0]] if self.tables else []
        goal_l = str(goal).lower()

        measures = []
        for table in relevant:
            for col_name in self.tables[table].get("columns", {}):
                if self._is_measure_column(table, col_name):
                    measures.append({"table": table, "column": col_name})
        ir["measures"] = measures

        concept_tokens = {
            "aggregate total": {"total", "sum", "amount", "value", "revenue",
                                "sales", "price", "quantity", "qty", "profit",
                                "units", "cost", "volume"},
            "row count": set(),
            "average value": {"average", "avg", "mean", "value", "price", "amount", "total"},
            "maximum value": {"maximum", "max", "value", "price", "amount"},
            "minimum value": {"minimum", "min", "value", "price", "amount"},
            "profit kpi": {"profit", "margin", "net", "income"},
            "revenue kpi": {"revenue", "amount", "total", "sales", "price"},
            "sales kpi": {"sales", "revenue", "amount", "total", "quantity", "qty", "price"},
        }

        # -- 1) SUBJECT grounding (synonym-aware table matching) -------------
        matched_tables = self._match_tables(goal_l)
        goal_tokens = self._normalize_tokens(goal_l)
        metric_present = bool(goal_tokens & self._METRIC_CONCEPT_WORDS)
        entity_tokens = {
            t for t in goal_tokens
            if t not in self._ANALYTICAL_GOAL_WORDS and t not in self._METRIC_CONCEPT_WORDS
        }
        if metric_present:
            # When the goal states a metric ("average order VALUE per customer")
            # bare count/fact nouns ("order", "sales", "quantity") belong to the
            # measure, not the breakdown entity.
            entity_tokens -= self._COUNT_OBJECT_TOKENS
        if matched_tables:
            subject, subject_tables = self._pick_subject(
                matched_tables, goal_tokens, entity_tokens
            )
        else:
            subject, subject_tables = None, []

        # -- 2) DIMENSION grounding -----------------------------------------
        dims = []
        # Label dimensions are bound to the entity (subject) tables the goal
        # actually names — never to pure-fact tables that only appear because a
        # generic word matched their name. This keeps "average order value per
        # customer" from dragging orders.ship_name / order_date into GROUP BY.
        for _score, table in subject_tables:
            label_col = self._pick_label_column(table)
            dims.append({
                "concept": table,
                "resolved_table": table,
                "resolved_column": label_col,
                "confidence": 0.9,
                "reason": "subject table named (or meant) in the question",
            })
        # Explicit breakdown columns ("by X", "per X", "for each X") are bound
        # across the join path from tables whose column names the breakdown
        # tokens resolve to ("by model year" -> products.model_year,
        # "by order status" -> transactions.status). Only from columns that are
        # legitimate dimensions (excludes PK/FK/*_id and pure measures).
        breakdown_tokens = self._explicit_breakdown_tokens(goal_l)
        if breakdown_tokens:
            breakdown_str = " ".join(sorted(breakdown_tokens))
            for table in relevant:
                cols = self._match_dimension_columns(breakdown_str, table)
                if cols:
                    label_col = self._pick_label_column(table)
                    for _score, col in cols:
                        if col != label_col:
                            dims.append({
                                "concept": col,
                                "resolved_table": table,
                                "resolved_column": col,
                                "confidence": 0.85,
                                "reason": "dimension/breakdown column meant in the question",
                            })
        seen_dims = set()
        for d in dims:
            key = (d["resolved_table"], d["resolved_column"])
            if key in seen_dims:
                continue
            seen_dims.add(key)
            ir["dimensions"].append(d)

        # -- 3) METRIC resolution -------------------------------------------
        for metric in ir["metrics"]:
            concept = metric["concept"].lower()
            if metric.get("aggregation") == "COUNT" or concept == "row count":
                fact = self._pick_fact_table(goal_l, relevant, subject)
                metric["resolved_table"] = fact
                metric["resolved_column"] = None
                metric["resolved_expression"] = "COUNT(*)"
                if metric.get("inferred"):
                    metric["confidence"] = 0.6
                    metric["reason"] = (
                        "no explicit measure; count of related records "
                        f"in {fact or 'the fact table'}"
                    )
                else:
                    metric["confidence"] = 0.9
                continue
            tokens = concept_tokens.get(concept, {concept})
            best, best_key = None, (0, -1, "")
            # Aggregate-monetary concepts prefer a monetary total column
            # (amount/total/revenue/sales) over a per-line price or a
            # quantity/units fact column when match scores tie, so "overall
            # sales" binds sales.amount rather than sale_lines.quantity
            # regardless of the order tables appear in the join path.
            monetary_concepts = {
                "sales kpi", "revenue kpi", "profit kpi",
            }
            monetary_bias = concept in monetary_concepts
            for m in measures:
                s = 0
                col_tokens = self._normalize_tokens(m["column"])
                col_name = m["column"].lower()
                for tok in tokens:
                    if tok in col_tokens:
                        s += 2
                    elif tok.endswith("s") and tok[:-1] in col_tokens:
                        s += 1
                # A measure only competes if a concept token actually matched
                # its column; otherwise `best` stays None and the
                # computed-measure path (quantity * unit_price) runs below.
                if s == 0:
                    continue
                tier = 2
                if monetary_bias and col_name in self._MONETARY_TOTAL_COLS:
                    tier = 0
                elif monetary_bias and col_name in self._MONETARY_LINE_COLS:
                    tier = 1
                elif monetary_bias and col_name in self._QUANTITY_COL_TOKENS:
                    tier = 3
                key = (s, -tier, col_name)
                if key > best_key:
                    best_key = key
                    best = m
            # If nothing matched (best is None) the computed-measure path
            # below synthesizes a quantity*unit_price expression.
            if best:
                metric["resolved_table"] = best["table"]
                metric["resolved_column"] = best["column"]
                metric["resolved_expression"] = f"{best['table']}.{best['column']}"
                metric["confidence"] = 0.9
                metric["reason"] = "measure column matched the requested metric"
            else:
                # Spec §3: sales-like concept with no single column -> try a
                # computed quantity * unit_price expression from the schema.
                computed = self._find_price_quantity_expression(relevant, tokens)
                if computed:
                    metric["resolved_table"] = computed["table"]
                    metric["resolved_column"] = None
                    metric["resolved_expression"] = computed["expression"]
                    metric["computed_measure"] = True
                    metric["confidence"] = 0.8
                    metric["reason"] = (
                        f"no single measure column; computed "
                        f"{computed['expression']} from the schema"
                    )

        if ir.get("time"):
            date_col = self._first_date_column(relevant)
            if date_col:
                ir["time"]["column"] = f"{date_col[0]}.{date_col[1]}"
                for f in ir["filters"]:
                    if f["column"] is None:
                        f["column"] = f"{date_col[0]}.{date_col[1]}"

        resolved_any = any(m.get("resolved_table") for m in ir["metrics"]) or bool(ir["dimensions"])
        ir["confidence"] = 0.85 if resolved_any else 0.5
        ir["reasoning"] = {
            "intent_confidence": 0.9 if ir.get("operation") else 0.5,
            "schema_confidence": ir["confidence"],
            "relationship_confidence": 0.7,
        }
        required = set()
        for m in ir["metrics"]:
            if m.get("resolved_table"):
                required.add(m["resolved_table"])
        for d in ir["dimensions"]:
            if d.get("resolved_table"):
                required.add(d["resolved_table"])
        ir["required_tables"] = (sorted(required) if required else list(join_path))
        return ir

