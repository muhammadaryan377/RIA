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


# ---- SNAPSHOT_KEEP (2011-2011) ----

SNAPSHOT_KEEP = 20  # max snapshot files retained per (user, database) file set

# ---- _prune_snapshots (2014-2033) ----

def _prune_snapshots(directory, stem, suffix, keep=SNAPSHOT_KEEP):
    """Delete the oldest snapshot files for a file set, keeping the newest `keep`.

    Never touches the base file ({stem}{suffix}) or the {stem}_latest alias.
    """
    try:
        snapshots = [
            p for p in Path(directory).glob(f"{stem}_*.json")
            if p.name != f"{stem}_latest{suffix}"
        ]
        if len(snapshots) <= keep:
            return
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in snapshots[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass

# ---- get_output_paths (2036-2075) ----

def get_output_paths(output_path, db_name=None, snapshot_id=None):
    """Return the base output, a unique snapshot, the latest alias, and the snapshot id.

    Every database (when `db_name` is given) gets its own file set:

        schema_mapping_<db>.json          - canonical base (overwritten each run)
        schema_mapping_<db>_latest.json   - stable pointer for the active session
        schema_mapping_<db>_<id>.json     - unique snapshot, never overwritten

    The snapshot file name embeds the same `snapshot_id` that the mapping JSON
    carries, so the JSON and its file are always correlated. Old snapshots beyond
    SNAPSHOT_KEEP are pruned. Returns (base, snapshot, latest, snapshot_id).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    stem, suffix = output.stem, output.suffix or ".json"

    if db_name:
        safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(db_name))[:64] or "db"
        stem = f"{stem}_{safe_db}"

    if not snapshot_id:
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # Guarantee a filename that does not already exist, so a snapshot is never
    # overwritten (unique within and across users, DBs, and processes).
    candidate = f"{stem}_{snapshot_id}{suffix}"
    counter = 1
    while (output.parent / candidate).exists():
        candidate = f"{stem}_{snapshot_id}_{counter}{suffix}"
        counter += 1

    base = output.with_name(f"{stem}{suffix}")
    snapshot = output.with_name(candidate)
    latest = output.with_name(f"{stem}_latest{suffix}")

    _prune_snapshots(output.parent, stem, suffix)
    used_id = candidate[len(stem) + 1:-len(suffix)]
    return base, snapshot, latest, used_id
