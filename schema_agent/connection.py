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
QUERY_TIMEOUT_MS = 30_000


# ---- _apply_session_timeout (54-70) ----

def _apply_session_timeout(conn, db_type="postgresql"):
    """Set a session-level statement timeout for read-only introspection.

    PostgreSQL: `statement_timeout` applies to every statement.
    MySQL: `MAX_EXECUTION_TIME` applies to SELECTs (MySQL 5.7.8+). If the
    server is older and rejects it, the timeout is skipped (query still runs,
    just unbounded — better than failing the connection).
    """
    try:
        with conn.cursor() as cur:
            if db_type == "mysql":
                cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {QUERY_TIMEOUT_MS}")
            else:
                cur.execute(f"SET statement_timeout = {QUERY_TIMEOUT_MS}")
    except Exception:
        pass

# ---- get_connection (77-137) ----

def get_connection(args):
    """Build a database connection (PostgreSQL or MySQL) from CLI args / env vars."""
    db_type = getattr(args, "db_type", None) or "postgresql"
    host = args.host or os.getenv("DB_HOST", "localhost")
    # The port must never leak across engine types: DB_PORT in .env targets
    # PostgreSQL, so MySQL must not inherit it. Only --port (or MYSQL_PORT) can
    # override MySQL's default.
    if db_type == "mysql":
        port = args.port or os.getenv("MYSQL_PORT", "3306")
    else:
        port = args.port or os.getenv("DB_PORT", "5432")
    dbname = args.db or os.getenv("DB_NAME")
    user = args.user or os.getenv("DB_USER")
    password = args.password or os.getenv("DB_PASSWORD")

    missing = [name for name, val in
               [("database", dbname), ("user", user), ("password", password)]
               if not val]
    if missing:
        sys.exit(
            f"ERROR: missing required connection info: {', '.join(missing)}.\n"
            f"Set them via .env or pass --db --user --password on the command line."
        )

    if db_type == "mysql":
        if not MYSQL_AVAILABLE:
            sys.exit("ERROR: pymysql is not installed. Run: pip install pymysql")
        try:
            conn = pymysql.connect(
                host=host, port=int(port), user=user, password=password,
                database=dbname, charset="utf8mb4", connect_timeout=10,
            )
        except pymysql.MySQLError as e:
            sys.exit(
                f"ERROR: could not connect to MySQL at {host}:{port}/{dbname}: {e}\n"
                "Check: is the MySQL server running on that host/port? Are the "
                "database name, user, and password correct, and does the server "
                "accept TCP connections?"
            )
        _apply_session_timeout(conn, "mysql")
        return conn

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
            connect_timeout=10,
        )
        # Read-only introspection: autocommit prevents one failed query from
        # leaving the connection in the "current transaction is aborted" state,
        # which would silently fail every later query on the same connection.
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        sys.exit(
            f"ERROR: could not connect to PostgreSQL at {host}:{port}/{dbname}: {e}\n"
            "Check: is the PostgreSQL service running on that host/port? Are the "
            "database name, user, and password correct? Does pg_hba.conf allow "
            "TCP connections from this host?"
        )
    _apply_session_timeout(conn, "postgresql")

    return conn

# ---- _dict_cursor (140-144) ----

def _dict_cursor(conn, db_type="postgresql"):
    """Return a dictionary-row cursor for the connected database type."""
    if db_type == "mysql":
        return conn.cursor(pymysql.cursors.DictCursor)
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ---- _lower_keys (147-157) ----

def _lower_keys(row):
    """Normalize a dict row's keys to lowercase.

    PostgreSQL's information_schema returns lowercase column names, but MySQL
    returns UPPERCASE ones (TABLE_NAME, COLUMN_NAME, ...). All downstream code
    reads lowercase keys, so MySQL rows must be normalized or every
    information_schema query fails with KeyError.
    """
    if not row:
        return row
    return {str(k).lower(): v for k, v in row.items()}

# ---- _quote_ident (160-164) ----

def _quote_ident(db_type, name):
    """Return a SQL-safe quoted identifier (double quotes for PG, backticks for MySQL)."""
    if db_type == "mysql":
        return "`" + str(name).replace("`", "") + "`"
    return '"' + str(name).replace('"', "") + '"'

# ---- _qname (alias for _quote_ident, kept for backward compatibility) ----

_qname = _quote_ident
