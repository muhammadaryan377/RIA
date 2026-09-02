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
from .connection import _qname

logger = logging.getLogger("aria.schema_agent")

JSON_INDENT = 2
K = 2


# ---- normalize_identifier (365-367) ----

def normalize_identifier(value):
    """Normalize names to compare database identifiers across styles."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())

# ---- _singularize (370-376) ----

def _singularize(name):
    """Best-effort singular of an English identifier (for PK/name matching)."""
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        return name[:-1]
    return name

# ---- _ID_TOKENS (380-380) ----

_ID_TOKENS = {"id", "no", "num", "number", "code", "key", "fk", "ref", "uuid", "sk", "nk"}

# ---- _tokenize_words (383-391) ----

def _tokenize_words(name):
    """Split an identifier into lowercase word tokens (snake_case, camelCase, acronyms).

    "MediaTypeId" -> ["media", "type", "id"]; "support_rep_id" -> ["support", "rep", "id"].
    This is what lets name matching work on ANY naming convention, not just X_id.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if w]

# ---- _is_reference_to_other_table (394-434) ----

def _is_reference_to_other_table(col_name, table_name, tables):
    """True when `col_name` (X_id style) names ANOTHER table (X / Xs / Xes / Xies).

    A column like `order_id` in a table that also has an `orders` table is
    almost certainly a foreign key, NOT the table's own primary key. Promoting
    it to a PK turns a reference into a target: it then wins overlap
    tiebreakers over the real referenced table (e.g. olist payments.order_id
    vs orders.order_id) and hides genuine relationships. This guard keeps
    FK-like columns out of PK inference.

    Matching is token-based (txt item 1: name-match evidence), so it works for
    ANY naming convention — plain `orders`, prefixed `olist_orders_dataset`,
    suffixed `tbl_order_items`, snake/camel — without per-database patches.
    """
    base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
    if not base or base in _ID_TOKENS:
        return False
    base = _singularize(base)
    forms = {base, base + "s", base + "es"}
    if base.endswith("y") and len(base) > 1:
        forms.add(base[:-1] + "ies")
    forms.add(_singularize(base))

    def _matches(t):
        """Does any word-token of table `t` (or its whole name) singularize to base?"""
        if normalize_identifier(t) in forms:
            return True
        return any(_singularize(tok) in forms for tok in _tokenize_words(t))

    # Find the MOST SPECIFIC table the base could name (shortest name that
    # still contains the base as a token). `orders` names `olist_orders_dataset`
    # but `order` inside `olist_order_payments_dataset` is just a qualifier —
    # the shorter/simpler name wins. If that most-specific match is the table's
    # OWN table, the column is its own identity; otherwise it is a reference.
    best = None
    for t in tables:
        if not _matches(t):
            continue
        if best is None or len(t) < len(best):
            best = t
    return best is not None and best != table_name

# ---- infer_primary_keys (437-480) ----

def infer_primary_keys(tables, declared_primary_keys):
    """Return likely PKs when the database does not declare them explicitly."""
    inferred = {}

    for table_name, columns in tables.items():
        if declared_primary_keys.get(table_name):
            inferred[table_name] = declared_primary_keys[table_name]
            continue

        table_norm = normalize_identifier(table_name)
        table_singular = _singularize(table_norm)
        candidates = []
        for col in columns:
            col_name = col["column"]
            normalized = normalize_identifier(col_name)
            if (normalized == "id"
                    or normalized == table_norm + "id"
                    or normalized == table_singular + "id"):
                candidates.append(col_name)

        if len(candidates) == 1:
            inferred[table_name] = candidates
            continue

        non_null_columns = [col["column"] for col in columns if not col["nullable"]]
        if len(non_null_columns) == 1:
            # Never trust a single non-null column as PK if it looks like a
            # foreign key (e.g. a table whose only NOT NULL column is
            # `customer_id` almost certainly points elsewhere). Only accept it
            # when it does not look like a reference to another table.
            candidate = non_null_columns[0]
            normalized = normalize_identifier(candidate)
            is_id_like = normalized.endswith("id") and normalized not in ("id", normalize_identifier(table_name) + "id")
            if not is_id_like:
                inferred[table_name] = non_null_columns
            continue

        id_like = [col["column"] for col in columns if col["column"].lower().endswith("_id") or col["column"].lower() == "id"]
        if len(id_like) == 1:
            candidate = id_like[0]
            if not _is_reference_to_other_table(candidate, table_name, tables):
                inferred[table_name] = id_like

    return inferred

# ---- _table_rows (490-516) ----

def _table_rows(conn, table_name, schema, db_type):
    """Best-effort row-count for the table (None when unknown).

    PostgreSQL uses pg_class.reltuples (a planner estimate, cheap and stable).
    MySQL uses a real COUNT(*) — information_schema.TABLES.TABLE_ROWS is an
    InnoDB estimate that can be 0 right after a bulk load or stale, which would
    wrongly gate PK verification. The session statement_timeout bounds the cost.
    """
    try:
        if db_type == "postgresql":
            cur = conn.cursor()
            cur.execute(
                "SELECT c.reltuples::bigint FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s",
                (schema, table_name),
            )
            row = cur.fetchone()
            return row[0] if row else None
        cur = conn.cursor()
        table_ref = _qname(db_type, table_name)
        cur.execute(f"SELECT COUNT(*) FROM {table_ref}")
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.debug("row count failed for %s.%s: %s", schema, table_name, exc)
        return None

# ---- _column_is_unique_key (519-538) ----

def _column_is_unique_key(conn, table_name, column, schema, db_type):
    """Return True when the column is fully populated and every value is unique."""
    try:
        cur = conn.cursor()
        table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                     if db_type == "postgresql" else _qname(db_type, table_name))
        col_ref = _qname(db_type, column)
        cur.execute(
            f"SELECT count(*) AS total, count({col_ref}) AS non_null, "
            f"count(DISTINCT {col_ref}) AS distinct_values FROM {table_ref}"
        )
        row = cur.fetchone()
        if not row:
            return False
        total, non_null, distinct_values = row[0], row[1], row[2]
        return (total is not None and total > 0
                and non_null == total and distinct_values == total)
    except Exception as exc:
        logger.debug("uniqueness check failed for %s.%s: %s", table_name, column, exc)
        return False

# ---- _column_is_empty (541-569) ----

def _column_is_empty(conn, table_name, column, schema, db_type):
    """Return True when a column holds no populated values (all NULL or blank).

    A primary key or foreign key built entirely on such a column carries no
    information, so it is dropped from the mapping instead of being reported as
    a real key.
    """
    try:
        with conn.cursor() as cur:
            table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                         if db_type == "postgresql" else _qname(db_type, table_name))
            col_ref = _qname(db_type, column)
            if db_type == "mysql":
                cur.execute(
                    f"SELECT COUNT({col_ref}) AS non_null, "
                    f"COUNT(NULLIF(TRIM(CAST({col_ref} AS CHAR)), '')) AS non_blank "
                    f"FROM {table_ref}"
                )
            else:
                cur.execute(
                    f"SELECT COUNT({col_ref}), COUNT(NULLIF(TRIM({col_ref}::text), '')) "
                    f"FROM {table_ref}"
                )
            row = cur.fetchone()
            return row is not None and (row[0] or 0) == 0 and (row[1] or 0) == 0
    except Exception as exc:
        logger.warning("empty-column check failed for %s.%s: %s", table_name, column, exc)
        return False

# ---- infer_primary_keys_from_data (572-619) ----

def infer_primary_keys_from_data(conn, tables, schema, db_type, existing_pks, max_rows=500000, exclude_columns=None):
    """Data-backed PK inference for tables the name heuristics could not resolve.

    When a table declares no PK and no column *name* points to one, look at the
    actual data: an id-like column that is fully populated and whose every value
    is unique (distinct count == row count) is the de-facto key. This handles
    flat/denormalized tables whose natural key is a prefixed text id
    (e.g. 'T0000001'), which naming conventions cannot detect, and never guesses
    for a column that is merely shared with other tables (customer_id, ...).

    `exclude_columns` ((table, column) pairs) lets confirmed foreign-key columns
    be ruled out as key candidates — a column that provably references another
    table is a reference, not the table's own identity.
    """
    exclude_columns = exclude_columns or set()
    inferred = {}
    for table_name, columns in tables.items():
        if existing_pks.get(table_name):
            continue
        id_like = [c["column"] for c in columns
                   if str(c["column"]).lower().endswith("_id") or str(c["column"]).lower() == "id"]
        id_like = [c for c in id_like
                   if (table_name, c) not in exclude_columns
                   # A column whose name matches ANOTHER table (order_id -> orders)
                   # is a disguised FK, not this table's identity. Never promote
                   # it to a primary key from data alone.
                   and not _is_reference_to_other_table(c, table_name, tables)]
        if not id_like:
            continue
        rows = _table_rows(conn, table_name, schema, db_type)
        if rows is not None and rows > max_rows:
            # Too large to verify with a scan; don't guess on partial evidence.
            continue
        unique = [col for col in id_like
                  if _column_is_unique_key(conn, table_name, col, schema, db_type)]
        if not unique:
            continue
        if len(unique) == 1:
            inferred[table_name] = unique
            continue
        # Several unique id columns: only commit when exactly one carries the
        # primary-looking name (id / <table>id / <table_singular>id).
        table_norm = normalize_identifier(table_name)
        primary_named = [c for c in unique if normalize_identifier(c) in
                         ("id", table_norm + "id", _singularize(table_norm) + "id")]
        if len(primary_named) == 1:
            inferred[table_name] = primary_named
    return inferred

# ---- infer_relationships (622-635) ----

def infer_relationships(tables, primary_keys, declared_fks, conn=None, schema="public", db_type="postgresql", null_stats=None):
    """Infer relationships for columns with no declared FK constraint.

    Phase C: delegates to the schema_engine pipeline (candidates -> evidence
    -> classifier -> acceptance policy -> registry), the single acceptance
    authority (RULE 6). Returns (inferred, ambiguous) legacy-shaped dicts.
    """
    from schema_engine.relationships import infer_relationships as _pipeline

    inferred, ambiguous, _registry = _pipeline(
        tables, primary_keys, declared_fks,
        conn=conn, schema=schema, db_type=db_type, null_stats=null_stats,
    )
    return inferred, ambiguous
