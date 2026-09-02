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

from .connection import (
    _apply_session_timeout, get_connection, _dict_cursor,
    _lower_keys, _quote_ident, _qname,
)
from .introspection import (
    get_tables_and_columns, get_primary_keys, get_unique_keys,
    get_declared_foreign_keys, get_null_stats,
)
from .inference import (
    normalize_identifier, _singularize, _ID_TOKENS, _tokenize_words,
    _is_reference_to_other_table, infer_primary_keys, _table_rows,
    _column_is_unique_key, _column_is_empty, infer_primary_keys_from_data,
    infer_relationships,
)
from .llm import (
    _SCHEMA_REASONING_ROLE, _SCHEMA_REASONING_PROMPT,
    _SCHEMA_REASONING_PROMPT_TAIL, _NUMERIC_TYPE_HINTS,
    _is_numeric_type, _column_data_type, _build_reasoning_brief,
    _parse_schema_reasoning, _apply_schema_reasoning, enrich_with_llm,
)
from .graph import _canonicalize_mapping, _build_relationship_graph
from .drift import detect_schema_drift
from .build import (
    schema_has_tables, list_schemas_with_tables, detect_schema_mode,
    find_schema_with_tables, _drop_empty_key_columns,
    build_schema_mapping, _all_schemas_foreign_keys,
    _infer_cross_schema_relationships, build_schema_mapping_all,
)
from .snapshots import SNAPSHOT_KEEP, _prune_snapshots, get_output_paths

QUERY_TIMEOUT_MS = 30_000


def main():
    """CLI entry point (python -m schema_agent)."""
    import argparse
    from llm_provider import create_provider

    parser = argparse.ArgumentParser(
        description="ARIA Schema Agent - relational database schema understanding"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--db-type", choices=["postgresql", "mysql"], default="postgresql")
    parser.add_argument("--output", default=str(SCHEMA_DIR / "schema_mapping.json"))
    parser.add_argument(
        "--provider", choices=["local", "cloud"], default=None,
        help="Enable LLM-assisted reasoning (local=Ollama, cloud=Groq).",
    )
    args = parser.parse_args()

    schema = args.schema or (args.db if args.db_type == "mysql" else "public")
    llm = create_provider(provider=args.provider) if args.provider else None
    conn = get_connection(args)
    try:
        multi_schema = False
        if not args.schema and args.db_type != "mysql":
            mode = detect_schema_mode(conn)
            multi_schema = mode["mode"] == "multi"
            if multi_schema:
                schema = "*"
            else:
                schema = mode["schemas"][0] if mode["schemas"] else "public"
            schema_names = ', '.join(mode['schemas']) or 'none'
            print(f"Detected schema mode '{mode['mode']}' "
                  f"({len(mode['schemas'])} schema(s): {schema_names}).")
        print(f"Connected ({args.db_type}). Extracting schema for '{schema}' schema...")

        previous_mapping = None
        try:
            from pathlib import Path as _P
            prev_base = _P(args.output)
            prev_stem = prev_base.stem
            if args.db:
                import re as _re
                safe_db = _re.sub(r"[^a-zA-Z0-9_-]+", "_", str(args.db))[:64] or "db"
                prev_stem = f"{prev_stem}_{safe_db}"
            prev_latest = prev_base.with_name(f"{prev_stem}_latest{prev_base.suffix or '.json'}")
            if prev_latest.exists():
                with open(prev_latest, encoding="utf-8") as fh:
                    previous_mapping = json.load(fh)
        except Exception as exc:
            logger.debug("could not load previous schema for drift detection: %s", exc)

        if multi_schema:
            mapping = build_schema_mapping_all(conn, db_type=args.db_type, llm=llm,
                                               database_name=args.db,
                                               previous_mapping=previous_mapping)
        else:
            mapping = build_schema_mapping(conn, schema, db_type=args.db_type, llm=llm,
                                           database_name=args.db,
                                           previous_mapping=previous_mapping)
        if not mapping.get("tables"):
            sys.exit(
                f"ERROR: no tables found in schema '{schema}' ({args.db_type}). "
                "The schema is empty or does not exist; nothing was mapped."
            )
        db_name = mapping.get("database", args.db or "unknown")
    finally:
        conn.close()

    output_path, timestamped_path, latest_path, snapshot_id = get_output_paths(
        args.output, db_name or None, snapshot_id=mapping.get("snapshot_id")
    )
    mapping["snapshot_id"] = snapshot_id

    for path in (output_path, timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    n_tables = len(mapping["tables"])
    n_declared = len(mapping["declared_relationships"])
    n_inferred = len(mapping["inferred_relationships"])
    print(f"Done. {n_tables} tables, {n_declared} declared relationships, "
          f"{n_inferred} inferred relationships.")
    if mapping.get("llm_reasoning"):
        print(f"LLM-assisted reasoning applied via {mapping['llm_reasoning'].get('provider')} "
              f"(model {mapping['llm_reasoning'].get('model')}).")
    drift = mapping.get("drift")
    if drift:
        if drift.get("has_changes"):
            print(f"Schema drift detected: {len(drift.get('new_tables', []))} new table(s), "
                  f"{len(drift.get('removed_tables', []))} removed, "
                  f"{len(drift.get('columns_added', {}))} table(s) with column changes, "
                  f"{len(drift.get('relationships_inferred_added', []))} new inferred relationship(s).")
        else:
            print("Schema drift: no changes since the previous run.")
    print(f"Written to {output_path}")
    print(f"Saved fresh snapshot to {timestamped_path}")
    print(f"Updated latest copy at {latest_path}")


if __name__ == "__main__":
    main()
