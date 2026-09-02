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

from .snapshots import _prune_snapshots, get_output_paths
from .introspection import (
    get_tables_and_columns, get_primary_keys, get_unique_keys,
    get_declared_foreign_keys, get_null_stats,
)
from .inference import (
    infer_primary_keys, infer_primary_keys_from_data, _column_is_empty,
)
from .graph import _canonicalize_mapping, _build_relationship_graph
from .llm import enrich_with_llm
from .drift import detect_schema_drift
from .connection import _dict_cursor


# ---- schema_has_tables (1183-1197) ----

def schema_has_tables(conn, schema):
    """Return True when the given Postgres schema contains at least one base table."""
    if not schema:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
            )
            """,
            (schema,),
        )
        return bool(cur.fetchone()[0])

# ---- list_schemas_with_tables (1200-1214) ----

def list_schemas_with_tables(conn):
    """Return non-system schemas that contain at least one base table (Postgres)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_schema NOT LIKE 'pg_%'
              AND table_type = 'BASE TABLE'
            GROUP BY table_schema
            ORDER BY count(*) DESC, table_schema
            """
        )
        return [row[0] for row in cur.fetchall()]

# ---- detect_schema_mode (1217-1230) ----

def detect_schema_mode(conn):
    """Classify a database as single-schema or multi-schema.

    Returns {"mode": "single"|"multi", "schemas": [...], "count": n}.
    Databases like northwind use only 'public' (single); AdventureWorks-style
    databases spread tables across several schemas (multi). The caller picks
    the matching extraction path so the single-schema flow is untouched.
    """
    schemas = list_schemas_with_tables(conn)
    return {
        "mode": "multi" if len(schemas) > 1 else "single",
        "schemas": schemas,
        "count": len(schemas),
    }

# ---- find_schema_with_tables (1233-1253) ----

def find_schema_with_tables(conn):
    """Return the largest non-system Postgres schema that contains base tables.

    Prefers the schema with the most tables (best signal for relationship
    inference). Falls back to 'public' when only it has tables.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_schema NOT LIKE 'pg_%'
              AND table_type = 'BASE TABLE'
            GROUP BY table_schema
            ORDER BY count(*) DESC, table_schema
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else "public"

# ---- _drop_empty_key_columns (1256-1310) ----

def _drop_empty_key_columns(mapping, conn, schema, db_type):
    """Remove any primary/foreign key whose source column is fully empty.

    A key built on a column with no populated values (all NULL or blank) cannot
    identify or reference anything, so it is dropped from the mapping. The
    removals are recorded under mapping["empty_key_columns_removed"] so the
    decision is transparent. On any query failure the column is kept (safe).
    """
    tables = mapping.setdefault("tables", {})
    removed_pks, removed_fks = [], []

    for table_name, info in tables.items():
        col_names = {c["column"] for c in info.get("columns", [])}
        for key_field in ("primary_key", "inferred_primary_key"):
            kept = []
            for pk in info.get(key_field, []):
                if pk in col_names and _column_is_empty(conn, table_name, pk, schema, db_type):
                    removed_pks.append((table_name, pk))
                else:
                    kept.append(pk)
            info[key_field] = kept
        if not info.get("primary_key"):
            info["primary_key"] = []

    def source_pair(rel, declared):
        if declared:
            return rel.get("table_name"), rel.get("column_name")
        return rel.get("table"), rel.get("column")

    removed_fk_pairs = set()
    for rel_list, declared in ((mapping.get("declared_relationships", []), True),
                               (mapping.get("inferred_relationships", []), False)):
        kept = []
        for rel in rel_list:
            table_name, col_name = source_pair(rel, declared)
            col_names = {c["column"] for c in tables.get(table_name, {}).get("columns", [])}
            if isinstance(col_name, str) and (table_name in tables and col_name in col_names
                    and _column_is_empty(conn, table_name, col_name, schema, db_type)):
                removed_fk_pairs.add((table_name, col_name))
                removed_fks.append((table_name, col_name, declared))
            else:
                kept.append(rel)
        rel_list[:] = kept

    if removed_fk_pairs:
        mapping["relationship_edges"] = [
            e for e in mapping.get("relationship_edges", [])
            if (e.get("source_table"), e.get("source_column")) not in removed_fk_pairs
        ]

    mapping["empty_key_columns_removed"] = {
        "primary_keys": sorted(set(removed_pks)),
        "foreign_keys": sorted(set(removed_fks)),
    }
    return mapping

# ---- build_schema_mapping (1426-1638) ----

def build_schema_mapping(conn, schema="public", db_type="postgresql", llm=None, database_name=None,
                         previous_mapping=None):
    tables = get_tables_and_columns(conn, schema, db_type)
    primary_keys = get_primary_keys(conn, schema, db_type)
    unique_keys = get_unique_keys(conn, schema, db_type)
    inferred_primary_keys = infer_primary_keys(tables, primary_keys)
    inferred_primary_keys.update(infer_primary_keys_from_data(
        conn, tables, schema, db_type,
        {**primary_keys, **inferred_primary_keys},
    ))
    declared_fks = get_declared_foreign_keys(conn, schema, db_type)
    null_stats = get_null_stats(conn, tables, schema, db_type)

    from schema_engine.relationships import infer_relationships as _pipeline

    merged_primary_keys = {**primary_keys, **inferred_primary_keys}
    inferred_fks, ambiguous_fks, registry = _pipeline(
        tables, merged_primary_keys, declared_fks,
        conn=conn, schema=schema, db_type=db_type,
        null_stats=null_stats,
    )

    # Refinement: a column confirmed (by data) to reference another table is a
    # foreign key, not the table's own identity. Ruling those out can resolve
    # ambiguous key candidates (e.g. a warranty table where both claim_id and
    # sale_id are unique — sale_id provably points at sales, so claim_id wins).
    fk_sources = {(r["table"], r["column"]) for r in inferred_fks
                  if r.get("confidence") in ("data-confirmed", "data-inferred")}
    if fk_sources:
        refined = infer_primary_keys_from_data(
            conn, tables, schema, db_type,
            {**primary_keys, **inferred_primary_keys},
            exclude_columns=fk_sources,
        )
        if refined:
            inferred_primary_keys.update(refined)
            merged_primary_keys = {**primary_keys, **inferred_primary_keys}
            inferred_fks, ambiguous_fks, registry = _pipeline(
                tables, merged_primary_keys, declared_fks,
                conn=conn, schema=schema, db_type=db_type,
            )

    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "database": database_name or os.getenv("DB_NAME") or "unknown_database",
        "schema": schema,
        "tables": {},
        "unique_keys": unique_keys,
        "declared_relationships": declared_fks,
        "inferred_relationships": inferred_fks,
        "ambiguous_relationships": ambiguous_fks,
        "relationship_edges": [],
    }

    for rel in declared_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table_name"],
            "source_column": rel["column_name"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "declared",
        })

    for rel in inferred_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table"],
            "source_column": rel["column"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "inferred",
            "confidence": rel.get("confidence", "heuristic"),
            "confidence_score": rel.get("confidence_score"),
            "review_status": rel.get("review_status"),
            "ambiguous": rel.get("ambiguous"),
            "self_referencing": rel.get("self_referencing"),
            "cardinality": rel.get("cardinality"),
        })

    for table_name, columns in tables.items():
        table_primary_key = primary_keys.get(table_name, inferred_primary_keys.get(table_name, []))
        mapping["tables"][table_name] = {
            "columns": columns,
            "primary_key": table_primary_key,
            "inferred_primary_key": inferred_primary_keys.get(table_name, []),
            "unique_keys": unique_keys.get(table_name, []),
            "null_stats": null_stats.get(table_name, {}),
        }

    # Track all tables discovered in the DB (before dropping empties).
    all_db_tables = set(tables.keys())

    # Empty tables (0 rows) stay in the mapping with their structure intact so
    # the schema is fully described, but they are excluded from relationship
    # inference (a key built on no data cannot be confirmed).
    empty_tables = {
        t for t, col_stats in null_stats.items()
        if col_stats and all(s.get("total_rows", 1) == 0 for s in col_stats.values())
    }
    if empty_tables:
        for t in empty_tables:
            if t in mapping["tables"]:
                mapping["tables"][t]["empty"] = True
                mapping["tables"][t]["row_count"] = 0
        mapping["declared_relationships"] = [
            r for r in mapping["declared_relationships"]
            if r["table_name"] not in empty_tables and r["references_table"] not in empty_tables
        ]
        mapping["inferred_relationships"] = [
            r for r in mapping["inferred_relationships"]
            if r.get("table") not in empty_tables and r.get("references_table") not in empty_tables
        ]
        mapping["relationship_edges"] = [
            e for e in mapping["relationship_edges"]
            if e["source_table"] not in empty_tables and e["target_table"] not in empty_tables
        ]

    # Build summary metadata for the API response.
    mapping["summary"] = {
        "total_db_tables": len(all_db_tables),
        "tables_used": len(mapping["tables"]),
        "empty_tables": sorted(empty_tables),
        "declared_fk_count": len(mapping["declared_relationships"]),
        "inferred_fk_count": len(mapping["inferred_relationships"]),
        "edge_count": len(mapping["relationship_edges"]),
        "tables_with_pk": sorted(
            t for t in mapping["tables"]
            if mapping["tables"][t].get("primary_key")
        ),
        "tables_without_pk": sorted(
            t for t in mapping["tables"]
            if not mapping["tables"][t].get("primary_key")
        ),
    }

    if llm is not None:
        mapping = enrich_with_llm(mapping, llm)

    if mapping.get("tables"):
        _drop_empty_key_columns(mapping, conn, schema, db_type)

    _canonicalize_mapping(mapping)

    # Item 9: build the explicit relationship graph (per-table outgoing and
    # incoming relationships) ARIA can consume directly for SQL generation /
    # RAG / query planning. Also detects many-to-many via junction tables
    # (item 2).
    mapping["relationship_graph"] = _build_relationship_graph(mapping)

    # Rich health report (drives the UI dashboard / QA checks).
    # Strong vs weak uses the STRUCTURED state (PART 12); the legacy confidence
    # strings remain only as provenance on each record.
    inferred = mapping.get("inferred_relationships", [])
    strong = [r for r in inferred
              if r.get("relationship_state") in ("CONFIRMED", "PROBABLE")]
    weak = [r for r in inferred
            if r.get("relationship_state") not in ("CONFIRMED", "PROBABLE")]
    # Item 10: human-review list for uncertain relationships.
    review_list = [
        {
            "table": r.get("table"),
            "column": r.get("column"),
            "references_table": r.get("references_table"),
            "references_column": r.get("references_column"),
            "confidence_score": r.get("confidence_score"),
            "confidence": r.get("confidence"),
            "confidence_level": r.get("confidence_level"),
            "relationship_state": r.get("relationship_state"),
            "evidence": r.get("evidence", []),
        }
        for r in inferred if r.get("review_status") in ("flagged", "review")
    ]
    # Ambiguous candidates (data cannot distinguish the real target) also go to
    # review so a human can arbitrate, but they never enter the accepted set.
    for r in mapping.get("ambiguous_relationships", []):
        review_list.append({
            "table": r.get("table"),
            "column": r.get("column"),
            "references_table": r.get("references_table"),
            "references_column": r.get("references_column"),
            "confidence_score": r.get("confidence_score"),
            "confidence": r.get("confidence"),
            "confidence_level": r.get("confidence_level"),
            "relationship_state": r.get("relationship_state"),
            "evidence": r.get("evidence", []),
        })
    review_list.sort(key=lambda x: (x.get("table") or "", x.get("column") or ""))

    # Truthful profiling budget (PART 8) - never hardcoded.
    budget_summary = registry.summary() if registry is not None else {}
    health = {
        "strong_inferred_fk_count": len(strong),
        "weak_inferred_fk_count": len(weak),
        "empty_table_count": len(empty_tables),
        "profile_budget_exhausted": budget_summary.get("profile_budget_exhausted", False),
        "profiling_status": budget_summary.get("profiling_status", "unknown"),
        "queries_used": budget_summary.get("queries_used"),
        "queries_remaining": budget_summary.get("queries_remaining"),
        "warning": None,
    }
    mapping["summary"]["health"] = health
    mapping["summary"]["review_list"] = review_list
    mapping["summary"]["review_count"] = len(review_list)
    # PART 14: rejected candidates are retained internally for debugging and
    # benchmarking even though they never enter the relationship lists.
    from schema_engine.serializer import serialize_rejected
    mapping["rejected_candidates"] = serialize_rejected(registry) if registry is not None else []

    # Item 5: schema drift vs the previous run of this database (when supplied).
    if previous_mapping:
        mapping["drift"] = detect_schema_drift(previous_mapping, mapping)

    return mapping

# ---- _all_schemas_foreign_keys (1641-1684) ----

def _all_schemas_foreign_keys(conn, schemas):
    """Declared FKs across every mapped schema, with schema-qualified names.

    Runs once for the merged multi-schema mapping so cross-schema foreign keys
    (e.g. purchasing.productvendor -> production.product) survive qualification
    and join paths can span schemas.
    """
    query = """
        SELECT
            refn.nspname AS table_schema,
            refcon.relname AS table_name,
            refatt.attname AS column_name,
            conn.nspname AS references_schema,
            conrel.relname AS references_table,
            conatt.attname AS references_column
        FROM pg_constraint c
        JOIN pg_namespace refn ON refn.oid = c.connamespace
        JOIN pg_class refcon ON refcon.oid = c.conrelid
        JOIN pg_namespace conn ON conn.oid = (SELECT relnamespace FROM pg_class WHERE oid = c.confrelid)
        JOIN pg_class conrel ON conrel.oid = c.confrelid
        JOIN unnest(c.conkey) WITH ORDINALITY AS srckeys(attnum, ord) ON TRUE
        JOIN unnest(c.confkey) WITH ORDINALITY AS tgtkeys(attnum, ord) ON srckeys.ord = tgtkeys.ord
        JOIN pg_attribute refatt ON refatt.attrelid = c.conrelid AND refatt.attnum = srckeys.attnum
        JOIN pg_attribute conatt ON conatt.attrelid = c.confrelid AND conatt.attnum = tgtkeys.attnum
        WHERE c.contype = 'f'
          AND refn.nspname = ANY(%s)
        ORDER BY refn.nspname, refcon.relname, srckeys.ord;
    """
    with _dict_cursor(conn) as cur:
        cur.execute(query, (list(schemas),))
        rows = cur.fetchall()
        rels = []
        for r in rows:
            rel = {
                "table_name": f"{r['table_schema']}.{r['table_name']}",
                "column_name": r["column_name"],
                "references_table": f"{r['references_schema']}.{r['references_table']}",
                "references_column": r["references_column"],
                "schema": r["table_schema"],
                "references_schema": r["references_schema"],
            }
            rel["same_schema"] = bool(r["table_schema"] == r["references_schema"])
            rels.append(rel)
        return rels

# ---- _infer_cross_schema_relationships (1687-1788) ----

def _infer_cross_schema_relationships(mapping, conn, db_type="postgresql"):
    """Global relationship inference across ALL schemas after the per-schema merge.

    Each schema's own pass already found intra-schema relationships. This pass
    searches for cross-schema edges (e.g. purchasing.productvendor.productid ->
    production.product.productid) that no single-schema pass could see, using the
    SAME generic evidence as normal inference (datatype compatibility, value
    overlap, uniqueness/PK-likeness, cardinality, null patterns, name and
    semantic similarity) on fully-qualified `schema.table.column` names.

    Only edges whose source and target schemas DIFFER are added, so intra-schema
    results are not duplicated. Edges already declared or already inferred are
    never re-emitted. Requires the mapping's tables to be schema-qualified and
    the connection to be available (no-op otherwise).
    """
    if conn is None:
        return
    tables = {
        qualified: info.get("columns", [])
        for qualified, info in mapping.get("tables", {}).items()
        if not info.get("empty")
    }
    if not tables:
        return
    schemas = {t.split(".", 1)[0] for t in tables if "." in t}
    if len(schemas) < 2:
        return  # single schema -> nothing cross-schema to find

    primary_keys = {
        qualified: list(info.get("primary_key", []) or [])
        for qualified, info in mapping.get("tables", {}).items()
    }
    declared = mapping.get("declared_relationships", [])
    from schema_engine.relationships import infer_relationships as _xs_pipeline
    cross, cross_ambiguous, cross_registry = _xs_pipeline(
        tables, primary_keys, declared,
        conn=conn, schema="", db_type=db_type,
    )

    existing = {
        (r.get("table"), r.get("column"), r.get("references_table"))
        for r in mapping.get("inferred_relationships", [])
    }
    existing |= {
        (r.get("table_name"), r.get("column_name"), r.get("references_table"))
        for r in declared
    }

    # Cross-schema candidates use the SAME pipeline and the SAME acceptance
    # state as intra-schema ones (PART 15): there is no separate cross-schema
    # acceptance rule. Only the structured relationship_state decides whether
    # an edge is emitted; everything else is surfaced for human review.
    added = []
    for r in cross:
        src_schema = r.get("schema")
        tgt_schema = r.get("references_schema")
        if src_schema and tgt_schema and src_schema == tgt_schema:
            continue  # intra-schema; already found by the per-schema pass
        key = (r.get("table"), r.get("column"), r.get("references_table"))
        if key in existing:
            continue
        if r.get("relationship_state") not in ("CONFIRMED", "PROBABLE"):
            r["relationship_state"] = "UNCERTAIN"
            r["review_status"] = "review"
            mapping.setdefault("_cross_schema_review", []).append(r)
            continue
        r["same_schema"] = False
        r["note"] = (r.get("note") or "") + (
            " Detected by the global cross-schema inference pass."
        )
        added.append(r)

    if added:
        mapping["inferred_relationships"].extend(added)
        for rel in added:
            mapping["relationship_edges"].append({
                "source_table": rel["table"],
                "source_column": rel["column"],
                "target_table": rel["references_table"],
                "target_column": rel["references_column"],
                "type": "inferred",
                "confidence": rel.get("confidence", "heuristic"),
                "confidence_score": rel.get("confidence_score"),
                "confidence_band": rel.get("confidence_band"),
                "relationship_state": rel.get("relationship_state"),
                "review_status": rel.get("review_status"),
                "ambiguous": rel.get("ambiguous"),
                "self_referencing": rel.get("self_referencing"),
                "cardinality": rel.get("cardinality"),
            })

    # Ambiguous cross-schema candidates: keep them visible for human review
    # even though they are never accepted as edges (same pipeline, same rule).
    for r in cross_ambiguous:
        r["relationship_state"] = "UNCERTAIN"
        r["review_status"] = "review"
        mapping.setdefault("_cross_schema_review", []).append(r)
    # Rejected cross-schema candidates are also retained (PART 14) rather than
    # disappearing, so benchmarks/debugging can explain every non-edge.
    if cross_registry is not None and cross_registry.rejected:
        mapping.setdefault("_cross_schema_rejected", []).extend(
            cross_registry.rejected)

# ---- build_schema_mapping_all (1791-2008) ----

def build_schema_mapping_all(conn, db_type="postgresql", llm=None, database_name=None,
                             previous_mapping=None):
    """Build a merged mapping across every populated schema in a database.

    Used only when `detect_schema_mode` reports more than one schema (e.g.
    AdventureWorks). Each schema is extracted with the existing single-schema
    pipeline, then merged with schema-qualified table names
    (`sales.salesorderheader`) so the Goal Agent can answer questions that span
    schemas. Single-schema databases never hit this path.
    """
    schemas = list_schemas_with_tables(conn)
    schema_health = []
    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "database": database_name or os.getenv("DB_NAME") or "unknown_database",
        "schema": "*",
        "schemas": schemas,
        "tables": {},
        "unique_keys": {},
        "declared_relationships": [],
        "inferred_relationships": [],
        "relationship_edges": [],
    }

    for schema in schemas:
        sub = build_schema_mapping(
            conn, schema, db_type=db_type, llm=None,
            database_name=database_name, previous_mapping=None,
        )
        # Aggregate truthful profiling budget per schema (PART 8).
        _h = (sub.get("summary") or {}).get("health") or {}
        schema_health.append(_h)
        for table_name, info in sub.get("tables", {}).items():
            qualified = f"{schema}.{table_name}"
            mapping["tables"][qualified] = info
            mapping["unique_keys"].update(
                {f"{schema}.{t}": v for t, v in sub.get("unique_keys", {}).items()}
            )
        for rel in sub.get("declared_relationships", []):
            mapping["declared_relationships"].append({
                "table_name": f"{schema}.{rel['table_name']}",
                "column_name": rel["column_name"],
                "references_table": f"{schema}.{rel['references_table']}",
                "references_column": rel["references_column"],
            })
        for rel in sub.get("inferred_relationships", []):
            mapping["inferred_relationships"].append({
                "table": f"{schema}.{rel['table']}",
                "column": rel["column"],
                "references_table": f"{schema}.{rel['references_table']}",
                "references_column": rel["references_column"],
                "confidence": rel.get("confidence", "heuristic"),
                "confidence_score": rel.get("confidence_score"),
                "confidence_band": rel.get("confidence_band"),
                "relationship_state": rel.get("relationship_state"),
                "review_status": rel.get("review_status"),
                "ambiguous": rel.get("ambiguous"),
                "self_referencing": rel.get("self_referencing"),
                "cardinality": rel.get("cardinality"),
                "evidence": rel.get("evidence", []),
                "evidence_detail": rel.get("evidence_detail"),
                "note": rel.get("note", ""),
                "schema": schema,
                "references_schema": schema,
                "same_schema": True,
            })
        for rel in sub.get("ambiguous_relationships", []):
            mapping.setdefault("_cross_schema_review", []).append({
                "table": f"{schema}.{rel['table']}",
                "column": rel["column"],
                "references_table": f"{schema}.{rel['references_table']}",
                "references_column": rel["references_column"],
                "confidence": rel.get("confidence", "heuristic"),
                "confidence_score": rel.get("confidence_score"),
                "confidence_band": rel.get("confidence_band"),
                "relationship_state": "uncertain",
                "review_status": "review",
                "ambiguous": rel.get("ambiguous"),
                "evidence": rel.get("evidence", []),
                "note": "Ambiguous candidate - data cannot distinguish best target.",
                "schema": schema,
                "references_schema": schema,
            })

    # Cross-schema declared FKs replace the per-schema (same-schema-only) list,
    # so join paths can span schemas and qualification is consistent.
    mapping["declared_relationships"] = _all_schemas_foreign_keys(conn, schemas)

    for rel in mapping["declared_relationships"]:
        mapping["relationship_edges"].append({
            "source_table": rel["table_name"],
            "source_column": rel["column_name"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "declared",
        })
    for rel in mapping["inferred_relationships"]:
        mapping["relationship_edges"].append({
            "source_table": rel["table"],
            "source_column": rel["column"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "inferred",
            "confidence": rel.get("confidence", "heuristic"),
            "confidence_score": rel.get("confidence_score"),
            "confidence_band": rel.get("confidence_band"),
            "relationship_state": rel.get("relationship_state"),
            "review_status": rel.get("review_status"),
            "ambiguous": rel.get("ambiguous"),
            "self_referencing": rel.get("self_referencing"),
            "cardinality": rel.get("cardinality"),
        })

    empty_tables = {
        t for t, info in mapping["tables"].items()
        if info.get("empty")
    }
    for t in empty_tables:
        if t in mapping["tables"]:
            mapping["tables"][t]["empty"] = True
            mapping["tables"][t]["row_count"] = 0
    mapping["declared_relationships"] = [
        r for r in mapping["declared_relationships"]
        if r["table_name"] not in empty_tables and r["references_table"] not in empty_tables
    ]
    mapping["inferred_relationships"] = [
        r for r in mapping["inferred_relationships"]
        if r.get("table") not in empty_tables and r.get("references_table") not in empty_tables
    ]
    mapping["relationship_edges"] = [
        e for e in mapping["relationship_edges"]
        if e["source_table"] not in empty_tables and e["target_table"] not in empty_tables
    ]

    # Global cross-schema inference (txt): after per-schema extraction and
    # merge, search across ALL schemas for edges no single-schema pass can see
    # (e.g. purchasing.productvendor -> production.product). Uses the same
    # generic evidence as normal inference on schema-qualified names.
    _infer_cross_schema_relationships(mapping, conn, db_type)

    _canonicalize_mapping(mapping)
    mapping["relationship_graph"] = _build_relationship_graph(mapping)

    inferred = mapping.get("inferred_relationships", [])
    strong = [r for r in inferred if r.get("confidence") in
              ("data-confirmed", "llm-confirmed", "composite-data-confirmed")]
    weak = [r for r in inferred if r.get("confidence") not in
            ("data-confirmed", "llm-confirmed", "composite-data-confirmed")]
    review_list = [
        {
            "table": r.get("table"),
            "column": r.get("column"),
            "references_table": r.get("references_table"),
            "references_column": r.get("references_column"),
            "confidence_score": r.get("confidence_score"),
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence", []),
        }
        for r in inferred if r.get("review_status") in ("flagged", "review")
    ]
    # Cross-schema candidates that lacked strong evidence are surfaced as
    # review items too, so a human can confirm genuine cross-schema FKs without
    # the engine auto-declaring weak value-overlap edges.
    for r in mapping.get("_cross_schema_review", []):
        review_list.append({
            "table": r.get("table"),
            "column": r.get("column"),
            "references_table": r.get("references_table"),
            "references_column": r.get("references_column"),
            "confidence_score": r.get("confidence_score"),
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence", []),
            "note": r.get("note"),
        })
    review_list.sort(key=lambda x: (x.get("table") or "", x.get("column") or ""))
    mapping["summary"] = {
        "total_db_tables": len(mapping["tables"]),
        "tables_used": len(mapping["tables"]),
        "empty_tables": sorted(empty_tables),
        "declared_fk_count": len(mapping["declared_relationships"]),
        "inferred_fk_count": len(inferred),
        "edge_count": len(mapping["relationship_edges"]),
        "tables_with_pk": sorted(
            t for t in mapping["tables"]
            if mapping["tables"][t].get("primary_key")
        ),
        "tables_without_pk": sorted(
            t for t in mapping["tables"]
            if not mapping["tables"][t].get("primary_key")
        ),
        "health": {
            "strong_inferred_fk_count": len(strong),
            "weak_inferred_fk_count": len(weak),
            "empty_table_count": len(empty_tables),
            "profile_budget_exhausted": bool(
                any(h.get("profile_budget_exhausted") for h in schema_health)
            ),
            "profiling_status": (
                "exhausted" if any(h.get("profile_budget_exhausted") for h in schema_health)
                else "active" if schema_health else "unknown"
            ),
            "queries_used": sum(h.get("queries_used") or 0 for h in schema_health),
            "queries_remaining": sum(h.get("queries_remaining") or 0 for h in schema_health),
            "warning": None,
        },
        "review_list": review_list,
        "review_count": len(review_list),
    }

    if llm is not None:
        mapping = enrich_with_llm(mapping, llm)

    if previous_mapping:
        mapping["drift"] = detect_schema_drift(previous_mapping, mapping)

    mapping.pop("_cross_schema_review", None)
    return mapping
