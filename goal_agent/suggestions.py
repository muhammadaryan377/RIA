"""Component: template-driven business-goal suggestions driven purely by the schema."""

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


class SuggestionsMixin:
    # ---- _schema_summary_text (_schema_summary_text) ----

    def _schema_summary_text(self):
        """Compact schema description used for LLM-based suggestion generation."""
        lines = []
        for table_name, info in self.tables.items():
            columns = ", ".join(info.get("columns", {}).keys()) or "no columns"
            lines.append(f"- {table_name} ({columns})")
        return "\n".join(lines)

    # ---- get_suggestions (get_suggestions) ----

    def get_suggestions(self, limit=8, use_llm=True):
        """Suggest searchable goals. Uses Mistral after analyzing the schema; falls back to template suggestions when the LLM is unavailable."""
        if not self.tables:
            return ["What insights can we derive from the connected data?"]

        template = self._template_suggestions(limit)
        if not use_llm:
            return template

        prompt = f"""
You are a senior data analyst writing questions for a business user. After
analysing the database tables below, suggest {limit} analytical questions a
data analyst would actually ask of this data. Prefer question archetypes such
as rankings ("top N by ..."), trends over time, comparisons between segments,
distributions, share of total, and anomalies or outliers. Avoid trivial
counts and generic 'show everything' questions.

Schema:
{self._schema_summary_text()}

Output a JSON array of strings only. No explanation, no markdown.
Write them the way a non-technical business user would type them - full
natural-English questions. Never mention raw table or column identifiers
(e.g. write "customers" as a word, not customers.customer_id).
"""
        try:
            content = self.llm.chat("suggest", messages=[{"role": "user", "content": prompt}],
                                    temperature=0.5, num_predict=300)
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)
            start, end = content.find("["), content.rfind("]")
            if start != -1 and end != -1:
                suggestions = json.loads(content[start:end + 1])
                if isinstance(suggestions, list) and suggestions:
                    return [str(s) for s in suggestions][:limit]
        except Exception as exc:
            logging.warning(f"Suggestion generation failed: {exc}. Using template suggestions.")

        return template

    # ---- _measure_noun (_measure_noun) ----

    def _measure_noun(self, fact_table, col):
        """Natural noun for a measure column when used in analyst questions:
        'orders.total' -> 'order value', 'order_details.unit_price' -> 'unit price'."""
        col_label = self._natural_label(col)
        if col_label in self._GENERIC_TOTAL_WORDS:
            return f"{self._natural_label(fact_table)} value"
        return col_label

    # ---- _template_suggestions (_template_suggestions) ----

    def _template_suggestions(self, limit=8):
        """Deterministic analyst-style questions (ranking / trend / comparison /
        distribution) built from schema structure only. No raw identifiers leak
        into the text, and no trivial 'how many rows' questions are emitted."""
        tables = list(self.tables.keys())
        if not tables:
            return ["Which factors most affect our key business metric?"]

        per_table = []
        for table_name in tables:
            info = self.tables[table_name]
            plural = self._plural_label(table_name)
            singular = self._natural_label(table_name)
            own_measure = self._pick_measure_column(table_name)
            date_col = next(
                (c for c in info["columns"]
                 if self._classify_column(table_name, c) == "date"),
                None,
            )

            fact_refs = []
            for other, other_info in self.tables.items():
                if other == table_name:
                    continue
                for fk in other_info.get("foreign_keys", []):
                    if fk["referenced_table"] == table_name:
                        fact_refs.append(other)
                        break

            ts = []
            if self._is_entity_table(table_name):
                fact = fact_refs[0] if fact_refs else None
                fact_measure = self._pick_measure_column(fact) if fact else None
                fact_date = None
                if fact:
                    fact_date = next(
                        (c for c in self.tables[fact]["columns"]
                         if self._classify_column(fact, c) == "date"),
                        None,
                    )
                if fact_measure:
                    noun = self._measure_noun(fact, fact_measure)
                    ts.append(f"What are the top {plural} by {noun}?")
                    ts.append(f"Which {plural} contribute the most to {noun}?")
                    if fact_date:
                        ts.append(f"How has {noun} changed over time?")
                    ts.append(f"What is the average {noun} per {singular}?")
                elif own_measure:
                    ts.append(
                        f"What are the top {plural} by {self._measure_noun(table_name, own_measure)}?"
                    )
                ts.append(f"What are the most important metrics for {plural}?")
            else:
                if own_measure:
                    noun = self._measure_noun(table_name, own_measure)
                    if date_col:
                        ts.append(f"What is the trend in {noun} over time?")
                        ts.append(f"How is {noun} distributed?")
                    else:
                        ts.append(f"What is the overall {noun}?")
            if ts:
                per_table.append(ts)

        # Round-robin across tables so the first N suggestions are varied,
        # not the first N tables repeated with the same template.
        suggestions = []
        max_depth = max(len(ts) for ts in per_table) if per_table else 0
        for depth in range(max_depth):
            for ts in per_table:
                if depth < len(ts):
                    suggestions.append(ts[depth])

        seen = set()
        unique = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique.append(suggestion)
        return unique[:limit]

