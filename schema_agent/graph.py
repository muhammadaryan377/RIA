import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import psycopg2
import psycopg2.extras
from psycopg2 import sql

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    pooling = None

try:
    import llm_provider as _llm_provider
except ImportError:
    _llm_provider = None

try:
    import schema_engine.relationships as _se_relationships
except ImportError:
    _se_relationships = None

from core.config import SCHEMA_DIR

logger = logging.getLogger("aria.schema_agent")

JSON_INDENT = 2
K = 2


# ---- _canonicalize_mapping (1029-1095) ----

def _canonicalize_mapping(mapping):
    """Validation pass: every relationship/edge must reference real objects.

    Runs AFTER all sources (declared FKs, heuristics, LLM) have merged so the
    final payload is guaranteed internally consistent:
      * tables/columns in relationships exist in `mapping["tables"]`
      * relationship_edges mirror declared+inferred relationships (no strays)
      * no duplicate relationship entries
    """
    tables = mapping.get("tables", {})
    known_columns = {
        t: {c["column"] for c in info.get("columns", [])}
        for t, info in tables.items()
    }

    def valid_rel(r, kind):
        if kind == "declared":
            src, src_col = r.get("table_name"), r.get("column_name")
            tgt, tgt_col = r.get("references_table"), r.get("references_column")
        else:
            src, src_col = r.get("table"), r.get("column")
            tgt, tgt_col = r.get("references_table"), r.get("references_column")
        if src not in known_columns or tgt not in known_columns:
            return False
        # Composite relationships carry a LIST of columns.
        cols = src_col if isinstance(src_col, list) else [src_col]
        ref_cols = tgt_col if isinstance(tgt_col, list) else [tgt_col]
        if any(c not in known_columns[src] for c in cols):
            return False
        if any(c not in known_columns[tgt] for c in ref_cols):
            return False
        return True

    mapping["declared_relationships"] = [
        r for r in mapping.get("declared_relationships", []) if valid_rel(r, "declared")
    ]
    mapping["inferred_relationships"] = [
        r for r in mapping.get("inferred_relationships", []) if valid_rel(r, "inferred")
    ]

    # Rebuild edges from the authoritative relationship lists.
    edges = []
    seen_edges = set()
    for r in mapping["declared_relationships"]:
        edges.append({
            "source_table": r["table_name"], "source_column": r["column_name"],
            "target_table": r["references_table"], "target_column": r["references_column"],
            "type": "declared", "confidence": "declared",
        })
    for r in mapping["inferred_relationships"]:
        edges.append({
            "source_table": r["table"], "source_column": r["column"],
            "target_table": r["references_table"], "target_column": r["references_column"],
            "type": "inferred", "confidence": r.get("confidence", "inferred"),
            "cardinality": r.get("cardinality"),
        })
    for e in edges:
        key = (e["source_table"], str(e["source_column"]), e["target_table"], str(e["target_column"]))
        if key not in seen_edges:
            seen_edges.add(key)
    deduped = []
    for e in edges:
        key = (e["source_table"], str(e["source_column"]), e["target_table"], str(e["target_column"]))
        if key in seen_edges:
            deduped.append(e)
            seen_edges.discard(key)
    mapping["relationship_edges"] = deduped

# ---- _build_relationship_graph (1098-1176) ----

def _build_relationship_graph(mapping):
    """Build an explicit per-table relationship graph (txt item 9).

    For every table: its primary key, columns, and separately its OUTGOING
    (this table references others) and INCOMING (others reference this table)
    relationships. Also detects MANY-TO-MANY (txt item 2) via junction tables:
    a table whose composite primary key columns each reference another table is
    a join table; the referenced tables get a many-to-many link through it.
    """
    tables = mapping.get("tables", {})
    edges = mapping.get("relationship_edges", [])
    graph = {}

    # Normalize an edge to (table, column) -> (target, target_column).
    def _out_edges(t):
        out = []
        for e in edges:
            if e.get("source_table") == t:
                out.append(e)
        return out

    def _in_edges(t):
        incoming = []
        for e in edges:
            if e.get("target_table") == t:
                incoming.append(e)
        return incoming

    for t, info in tables.items():
        pks = info.get("primary_key") or info.get("inferred_primary_key") or []
        graph[t] = {
            "primary_key": pks,
            "columns": [c["column"] for c in info.get("columns", [])],
            "outgoing_relationships": [],
            "incoming_relationships": [],
            "many_to_many": [],
        }
        for e in _out_edges(t):
            graph[t]["outgoing_relationships"].append({
                "source_column": e.get("source_column"),
                "target_table": e.get("target_table"),
                "target_column": e.get("target_column"),
                "type": e.get("type"),
                "confidence": e.get("confidence"),
                "cardinality": e.get("cardinality"),
                "ambiguous": e.get("ambiguous"),
            })
        for e in _in_edges(t):
            graph[t]["incoming_relationships"].append({
                "source_table": e.get("source_table"),
                "source_column": e.get("source_column"),
                "target_column": e.get("target_column"),
                "type": e.get("type"),
                "confidence": e.get("confidence"),
                "cardinality": e.get("cardinality"),
                "ambiguous": e.get("ambiguous"),
            })

    # Many-to-many via junction tables (composite PK, each PK column a FK).
    for t, info in tables.items():
        pks = info.get("primary_key") or info.get("inferred_primary_key") or []
        if len(pks) < 2:
            continue
        refs = {}
        for e in _out_edges(t):
            if e.get("source_column") in pks and e.get("target_table") != t:
                refs[e["source_column"]] = e["target_table"]
        if len(refs) >= 2:
            # A junction: every PK column points at a different table.
            involved = sorted(set(refs.values()))
            if len(involved) >= 2:
                junction = {
                    "junction_table": t,
                    "columns": list(refs.keys()),
                    "tables": involved,
                }
                for parent in involved:
                    graph[parent]["many_to_many"].append(junction)
    return graph
