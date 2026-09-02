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
from .connection import _dict_cursor, _quote_ident, _lower_keys

logger = logging.getLogger("aria.schema_agent")

JSON_INDENT = 2
K = 2


# ---- get_tables_and_columns (171-199) ----

def get_tables_and_columns(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [{column, data_type, nullable}, ...]}

    Only BASE TABLEs are profiled; views are skipped (their PK/uniqueness
    checks can be slow or meaningless against a complex view query).
    """
    query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name IN (
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
          )
        ORDER BY table_name, ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema, schema))
        rows = cur.fetchall()

    tables = {}
    for row in rows:
        row = _lower_keys(row)
        tables.setdefault(row["table_name"], []).append({
            "column": row["column_name"],
            "data_type": row["data_type"],
            "nullable": row["is_nullable"] == "YES",
        })
    return tables

# ---- get_primary_keys (202-226) ----

def get_primary_keys(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [pk_column, ...]}"""
    # NOTE: the join must include tc.table_name. In MySQL every primary key
    # constraint is literally named 'PRIMARY', so joining on constraint_name
    # alone cross-joins every table's PK columns to every other table.
    query = """
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    pks = {}
    for row in rows:
        row = _lower_keys(row)
        pks.setdefault(row["table_name"], []).append(row["column_name"])
    return pks

# ---- get_unique_keys (229-258) ----

def get_unique_keys(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [[col, ...], ...]} — one entry per UNIQUE constraint.

    Declared UNIQUE constraints are authoritative: a column covered by one is
    already known to be unique, so it should never be re-inferred or treated as
    a duplicate/weak key downstream.
    """
    query = """
        SELECT tc.table_name, kcu.column_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'UNIQUE'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    uniques = {}
    for row in rows:
        row = _lower_keys(row)
        uniques.setdefault(row["table_name"], {})
        uniques[row["table_name"]].setdefault(row["constraint_name"], []).append(
            row["column_name"]
        )
    return {t: list(cols.values()) for t, cols in uniques.items()}

# ---- get_declared_foreign_keys (261-301) ----

def get_declared_foreign_keys(conn, schema="public", db_type="postgresql"):
    """Return list of {table_name, column_name, references_table, references_column}"""
    if db_type == "mysql":
        # MySQL exposes the referenced table/column directly on key_column_usage.
        query = """
            SELECT
                kcu.table_name AS table_name,
                kcu.column_name AS column_name,
                kcu.referenced_table_name AS references_table,
                kcu.referenced_column_name AS references_column
            FROM information_schema.key_column_usage kcu
            WHERE kcu.referenced_table_name IS NOT NULL
              AND kcu.table_schema = %s
            ORDER BY kcu.table_name, kcu.ordinal_position;
        """
    else:
        # PostgreSQL: join pg_constraint directly and align conkey/confkey by
        # ordinal position. Joining information_schema.key_column_usage to
        # constraint_column_usage on the constraint name alone would cross-produce
        # composite foreign keys (a 2-column FK would yield 4 wrong pairs).
        query = """
            SELECT
                refcon.relname AS table_name,
                refatt.attname AS column_name,
                conrel.relname AS references_table,
                conatt.attname AS references_column
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            JOIN pg_class refcon ON refcon.oid = c.conrelid
            JOIN pg_class conrel ON conrel.oid = c.confrelid
            JOIN unnest(c.conkey) WITH ORDINALITY AS srckeys(attnum, ord) ON TRUE
            JOIN unnest(c.confkey) WITH ORDINALITY AS tgtkeys(attnum, ord) ON srckeys.ord = tgtkeys.ord
            JOIN pg_attribute refatt ON refatt.attrelid = c.conrelid AND refatt.attnum = srckeys.attnum
            JOIN pg_attribute conatt ON conatt.attrelid = c.confrelid AND conatt.attnum = tgtkeys.attnum
            WHERE c.contype = 'f'
              AND n.nspname = %s
            ORDER BY refcon.relname, srckeys.ord;
        """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        return [dict(row) for row in cur.fetchall()]

# ---- get_null_stats (304-358) ----

def get_null_stats(conn, tables, schema="public", db_type="postgresql"):
    """Return {table_name: {column_name: {"total_rows": n, "nulls": n, "null_pct": p}}}

    ONE aggregate query per table computes every column's null count (PART 9):
    the old per-column COUNT pattern is an N+1 anti-pattern that issued
    len(columns)+1 round trips per table. A table that cannot be counted
    (permission error, exotic type, huge cost) reports UNKNOWN for every column
    rather than zeros and never kills the run (missing != negative).
    """
    stats = {}
    with conn.cursor() as cur:
        for table_name, columns in tables.items():
            if not columns:
                continue
            table_stats = {}
            try:
                col_names = [c["column"] for c in columns]
                if db_type == "mysql":
                    qualified = f"{_quote_ident(db_type, schema)}.{_quote_ident(db_type, table_name)}"
                    select_parts = ", ".join(
                        f"COUNT({_quote_ident(db_type, c)}) AS c{i}" for i, c in enumerate(col_names)
                    )
                    cur.execute(f"SELECT COUNT(*) AS total, {select_parts} FROM {qualified}")
                    row = cur.fetchone()
                else:
                    select_parts = ", ".join(
                        f"COUNT({_quote_ident(db_type, c)}) AS c{i}" for i, c in enumerate(col_names)
                    )
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) AS total, {} FROM {}.{}").format(
                            sql.SQL(select_parts),
                            sql.Identifier(schema),
                            sql.Identifier(table_name),
                        )
                    )
                    row = cur.fetchone()
                if row:
                    total = row[0]
                    for i, col_name in enumerate(col_names):
                        non_null = row[i + 1]
                        nulls = total - (non_null or 0)
                        null_pct = round((nulls / total) * 100, 2) if total else 0.0
                        table_stats[col_name] = {
                            "total_rows": total,
                            "nulls": nulls,
                            "null_pct": null_pct,
                        }
            except Exception:
                # A single unreadable table must not poison the whole mapping;
                # every column reports UNKNOWN (missing evidence, not zero).
                logger.debug("null stats failed for table %s", table_name, exc_info=True)
                stats[table_name] = {}
                continue
            stats[table_name] = table_stats
    return stats
