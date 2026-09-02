"""Shared, schema-agnostic constants for the Goal Agent (synonym groups, goal words, SQL keywords, regexes)."""

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


class ConstantsMixin:
    # ---- UNCERTAIN_STATES (UNCERTAIN_STATES) ----

    UNCERTAIN_STATES = {"UNCERTAIN", "REVIEW"}

    # ---- _NUMERIC_TYPES (_NUMERIC_TYPES) ----

    _NUMERIC_TYPES = ("int", "decimal", "numeric", "real", "double", "float",
                      "money", "number")

    # ---- _DATE_TYPES (_DATE_TYPES) ----

    _DATE_TYPES = ("date", "time", "timestamp")

    # ---- _GENERIC_TOTAL_WORDS (_GENERIC_TOTAL_WORDS) ----

    _GENERIC_TOTAL_WORDS = {"total", "amount", "value", "sales", "revenue", "sum"}

    # ---- _OVERVIEW_RE (_OVERVIEW_RE) ----

    _OVERVIEW_RE = re.compile(
        r"\b(?:"
        r"most important (?:metric|metrics|kpi|kpis|things?|aspects?|figures?|numbers?)"
        r"|(?:important|key|main|top|all|primary) (?:metric|metrics|kpi|kpis)"
        r"|what (?:are|is) the (?:most )?important"
        r"|metrics?\b|kpis?\b|overview\b"
        r")\b",
        re.IGNORECASE,
    )

    # ---- _STRIP_STRINGS_RE (_STRIP_STRINGS_RE) ----

    _STRIP_STRINGS_RE = re.compile(r"'([^']|'')*'|\"([^\"]|\"\")*\"|`[^`]*`")

    # ---- _DESTRUCTIVE_RE (_DESTRUCTIVE_RE) ----

    _DESTRUCTIVE_RE = re.compile(
        r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM"
        r"|DROP\s+(?:TABLE|DATABASE|VIEW|INDEX|SCHEMA|TRIGGER|FUNCTION|PROCEDURE)"
        r"|ALTER\s+(?:TABLE|DATABASE|VIEW|SCHEMA)"
        r"|CREATE\s+(?:TABLE|DATABASE|INDEX|VIEW|SCHEMA|TRIGGER|FUNCTION|PROCEDURE)"
        r"|TRUNCATE\s+(?:TABLE\s+)?|GRANT|REVOKE|CALL|EXEC(?:UTE)?\b|MERGE\s+INTO"
        r"|COMMENT\s+ON)\b",
        re.IGNORECASE,
    )

    # ---- SEMANTIC_SYNONYMS (SEMANTIC_SYNONYMS) ----

    SEMANTIC_SYNONYMS = {
        "customer": {"client", "buyer", "consumer", "purchaser", "account"},
        "product": {"item", "article", "merchandise", "sku", "part", "good"},
        "order": {"sale", "transaction", "purchase", "invoice", "booking"},
        "sale": {"order", "transaction", "purchase", "invoice", "deal"},
        "employee": {"staff", "person", "worker", "associate", "user", "member"},
        "store": {"branch", "outlet", "shop", "location", "site", "warehouse"},
        "shipper": {"shipping", "carrier", "courier", "delivery", "freight", "logistics"},
        "brand": {"make", "manufacturer", "vendor", "label", "supplier"},
        "category": {"group", "department", "segment", "class", "type", "division"},
        "quantity": {"qty", "units", "volume", "amount"},
        "price": {"cost", "amount", "value", "rate", "fee"},
        "revenue": {"sales", "income", "turnover", "gross", "earnings"},
        "status": {"state", "stage", "condition", "phase", "position"},
        "year": {"yr", "model_year", "manufacture_year", "production_year", "calendar_year"},
        "month": {"mth", "period", "month_name"},
        "date": {"date", "day", "timestamp", "created", "datetime", "time"},
    }

    # ---- _ANALYTICAL_GOAL_WORDS (_ANALYTICAL_GOAL_WORDS) ----

    _ANALYTICAL_GOAL_WORDS = frozenset({
        "top", "best", "worst", "highest", "lowest", "most", "least", "rank",
        "leader", "bottom", "total", "sum", "count", "how", "many", "average",
        "avg", "mean", "max", "maximum", "min", "minimum", "compare",
        "comparison", "versus", "trend", "over", "time", "monthly", "daily",
        "weekly", "yearly", "growth", "decline", "increase", "decrease",
        "change", "distribution", "breakdown", "percentage", "share", "ratio",
        "per", "by", "show", "list", "what", "which", "is", "are", "was",
        "were", "overall", "number", "sort", "limit", "group", "between",
        "of", "in", "on", "for", "with",
    })

    # ---- _METRIC_CONCEPT_WORDS (_METRIC_CONCEPT_WORDS) ----

    _METRIC_CONCEPT_WORDS = frozenset({
        "sales", "revenue", "profit", "margin", "income", "turnover",
        "quantity", "qty", "units", "volume", "spend", "amount", "value",
        "price", "cost", "total", "sum", "count", "average", "avg", "mean",
    })

    # ---- _OPERATION_WORDS (_OPERATION_WORDS) ----

    _OPERATION_WORDS = frozenset({
        "top", "best", "worst", "highest", "lowest", "most", "least", "bottom",
        "rank", "leader", "how", "many", "sum", "average", "avg", "mean",
        "max", "minimum", "compare", "comparison", "versus", "trend", "growth",
        "decline", "increase", "decrease", "change", "distribution",
        "breakdown", "percentage", "share", "ratio", "monthly", "daily",
        "weekly", "yearly", "month", "year", "over", "by", "per", "overall",
        "group", "between", "among",
    })

    # ---- _PROTECTED_GOAL_WORDS (_PROTECTED_GOAL_WORDS) ----

    _PROTECTED_GOAL_WORDS = frozenset({
        # Articles / prepositions / connectors that are never typos of a schema word.
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "by", "from",
        "for", "per", "with", "between", "over", "under",
        # Imperatives / framing words.
        "show", "me", "list", "top", "all", "each", "how", "many", "what",
        "which", "is", "are", "was", "were", "please", "give",
        # KPI vocabulary (from _build_kpi_index); these carry business meaning
        # and must never be rewritten into a schema word.
        "total", "sum", "count", "number", "average", "avg", "mean", "max",
        "maximum", "min", "minimum", "profit", "revenue", "sales", "growth",
        # Analysis framing.
        "trend", "monthly", "yearly", "daily", "compare", "comparison",
        "ranking", "rank", "forecast", "best", "worst", "highest", "lowest",
        "most", "least", "time", "performance",
    })

    # ---- _DIMENSION_LIKE_NUMERIC (_DIMENSION_LIKE_NUMERIC) ----

    _DIMENSION_LIKE_NUMERIC = (
        "year", "date", "month", "day", "status", "state", "code", "rank",
        "rating", "level", "version", "quarter", "age", "no", "num",
    )

    # ---- _EXTREMES_MAX (_EXTREMES_MAX) ----

    _EXTREMES_MAX = {"highest", "largest", "most", "maximum", "max", "best", "top", "greatest", "biggest"}

    # ---- _EXTREMES_MIN (_EXTREMES_MIN) ----

    _EXTREMES_MIN = {"lowest", "smallest", "least", "minimum", "min", "worst", "bottom", "cheapest"}

    # ---- _SQL_KEYWORDS (_SQL_KEYWORDS) ----

    _SQL_KEYWORDS = {
        "on", "where", "group", "order", "having", "join", "left", "right",
        "inner", "outer", "cross", "full", "as", "using", "limit", "offset",
        "select", "from", "and", "or", "not", "by", "asc", "desc", "nulls",
        "first", "last", "case", "when", "then", "else", "end", "union", "all",
        "distinct", "top", "with", "values",
    }

