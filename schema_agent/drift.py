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


# ---- detect_schema_drift (1313-1423) ----

def detect_schema_drift(previous, current):
    """Compare two schema mappings (txt item 5) and report what changed.

    Detects new/removed tables, added/removed columns, changed data types,
    changed PK/FK structure, and newly inferred relationships. Used when the
    agent runs repeatedly against a live database whose schema evolves.
    """
    prev_tables = previous.get("tables", {}) if previous else {}
    curr_tables = current.get("tables", {}) if current else {}

    def _edge_set(mapping, declared=True):
        out = set()
        for rel in mapping.get("declared_relationships", []):
            if declared:
                out.add((rel["table_name"], str(rel["column_name"]),
                         rel["references_table"], str(rel["references_column"])))
        return out

    def _inferred_set(mapping):
        out = set()
        for rel in mapping.get("inferred_relationships", []):
            col = rel.get("column")
            if isinstance(col, list):
                col = ",".join(col)
            ref = rel.get("references_column")
            if isinstance(ref, list):
                ref = ",".join(ref)
            out.add((rel.get("table"), str(col),
                     rel.get("references_table"), str(ref)))
        return out

    drift = {"has_changes": False, "changed_tables": []}

    new_tables = sorted(set(curr_tables) - set(prev_tables))
    removed_tables = sorted(set(prev_tables) - set(curr_tables))
    if new_tables:
        drift["new_tables"] = new_tables
        drift["has_changes"] = True
    if removed_tables:
        drift["removed_tables"] = removed_tables
        drift["has_changes"] = True

    added_columns = {}
    removed_columns = {}
    datatype_changed = {}
    pk_changed = {}
    for table in sorted(set(curr_tables) & set(prev_tables)):
        prev_cols = {c["column"]: c for c in prev_tables[table].get("columns", [])}
        curr_cols = {c["column"]: c for c in curr_tables[table].get("columns", [])}
        new_cols = sorted(set(curr_cols) - set(prev_cols))
        gone_cols = sorted(set(prev_cols) - set(curr_cols))
        type_changes = {}
        for col in sorted(set(curr_cols) & set(prev_cols)):
            if str(prev_cols[col].get("data_type", "")).lower() != str(curr_cols[col].get("data_type", "")).lower():
                type_changes[col] = {
                    "from": prev_cols[col].get("data_type"),
                    "to": curr_cols[col].get("data_type"),
                }
        prev_pk = sorted(prev_tables[table].get("primary_key", []) or [])
        curr_pk = sorted(curr_tables[table].get("primary_key", []) or [])
        table_changed = new_cols or gone_cols or type_changes or (prev_pk != curr_pk)
        if table_changed:
            added_columns[table] = new_cols
            removed_columns[table] = gone_cols
            datatype_changed[table] = type_changes
            pk_changed[table] = {"from": prev_pk, "to": curr_pk}
            drift["changed_tables"].append(table)
            drift["has_changes"] = True

    if added_columns:
        drift["columns_added"] = added_columns
    if removed_columns:
        drift["columns_removed"] = removed_columns
    if datatype_changed:
        drift["datatypes_changed"] = datatype_changed
    if pk_changed:
        drift["primary_keys_changed"] = pk_changed

    # FK structure: declared constraints added/removed.
    prev_declared = _edge_set(previous)
    curr_declared = _edge_set(current)
    declared_added = sorted(curr_declared - prev_declared, key=str)
    declared_removed = sorted(prev_declared - curr_declared, key=str)
    if declared_added:
        drift["declared_fk_added"] = declared_added
        drift["has_changes"] = True
    if declared_removed:
        drift["declared_fk_removed"] = declared_removed
        drift["has_changes"] = True

    # Inferred relationships added/removed.
    prev_inferred = _inferred_set(previous)
    curr_inferred = _inferred_set(current)
    inf_added = sorted(curr_inferred - prev_inferred, key=str)
    inf_removed = sorted(prev_inferred - curr_inferred, key=str)
    if inf_added:
        drift["relationships_inferred_added"] = inf_added
        drift["has_changes"] = True
    if inf_removed:
        drift["relationships_inferred_removed"] = inf_removed
        drift["has_changes"] = True

    return drift
