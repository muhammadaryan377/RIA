"""PostgreSQL adapter -> normalized model.

SQL is equivalent to the legacy `schema_agent.py` extraction so Phase B
preserves behavior; two deliberate fixes are included now because they are
pure metadata correctness (PART 10):

* the declared-FK query already aligns `conkey`/`confkey` by ordinal (no
  accidental ordering reliance) and now also carries the constraint identity;
* null statistics are computed with ONE aggregate query per table instead of
  one query per column (PART 9).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import psycopg2.extras
from psycopg2 import sql

from schema_engine.adapters.base import DatabaseAdapter
from schema_engine.models import (
    Column, ColumnStatistics, Database, ForeignKey, Schema, Table,
    UniqueConstraint,
)

logger = logging.getLogger("aria.schema_engine.postgres")


class PostgreSQLAdapter(DatabaseAdapter):
    db_type = "postgresql"

    def _dict_cursor(self):
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _quote_ident(self, name: str) -> str:
        return '"' + str(name).replace('"', "") + '"'

    # ------------------------------------------------------------------ #

    def list_schemas(self) -> List[str]:
        with self._dict_cursor() as cur:
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
            return [row["table_schema"] for row in cur.fetchall()]

    # ------------------------------------------------------------------ #

    def _base_tables(self, schemas: List[str]) -> List[Tuple[str, str]]:
        """(schema, table) pairs for base tables in the given schemas."""
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
                """,
                (list(schemas),),
            )
            return [(r["table_schema"], r["table_name"]) for r in cur.fetchall()]

    def _column_rows(self, schemas: List[str]) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = ANY(%s)
                ORDER BY table_schema, table_name, ordinal_position
                """,
                (list(schemas),),
            )
            return [dict(r) for r in cur.fetchall()]

    def _pk_rows(self, schemas: List[str]) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT tc.table_schema, tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = ANY(%s)
                ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
                """,
                (list(schemas),),
            )
            return [dict(r) for r in cur.fetchall()]

    def _unique_rows(self, schemas: List[str]) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT tc.table_schema, tc.table_name, tc.constraint_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'UNIQUE'
                  AND tc.table_schema = ANY(%s)
                ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
                """,
                (list(schemas),),
            )
            return [dict(r) for r in cur.fetchall()]

    def _fk_rows(self, schemas: List[str]) -> List[dict]:
        """Declared FKs via pg_constraint with conkey/confkey ordinal alignment.

        Alignment is by explicit ordinal position (never accidental result
        ordering), and the constraint identity is preserved for composite FKs.
        """
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.conname AS constraint_name,
                    refn.nspname AS table_schema,
                    refcon.relname AS table_name,
                    refatt.attname AS column_name,
                    conn.nspname AS references_schema,
                    conrel.relname AS references_table,
                    conatt.attname AS references_column,
                    srckeys.ord AS ordinal_position
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
                ORDER BY refn.nspname, refcon.relname, srckeys.ord
                """,
                (list(schemas),),
            )
            return [dict(r) for r in cur.fetchall()]

    def _row_counts(self, tables: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[int]]:
        """pg_class.reltuples estimates (cheap; matches legacy behavior)."""
        out: Dict[Tuple[str, str], Optional[int]] = {}
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT n.nspname AS schema, c.relname AS table, c.reltuples::bigint AS rows
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = ANY(%s) AND c.reltuples IS NOT NULL AND c.relkind = 'r'
                    """,
                    (list({s for s, _ in tables}),),
                )
                for row in cur.fetchall():
                    out[(row[0], row[1])] = row[2]
        except Exception as exc:
            logger.debug("row-count query failed: %s", exc)
        return out

    def _null_stats(self, table: Table) -> Dict[str, ColumnStatistics]:
        """One aggregate query per table computes every column's null count.

        Batched (PART 9) but value-identical to the legacy per-column COUNT
        queries. On failure the whole table reports UNKNOWN rather than lying
        with zeros (PART 7/8).
        """
        stats: Dict[str, ColumnStatistics] = {}
        if not table.columns:
            return stats
        try:
            col_refs = [self._quote_ident(c.name) for c in table.columns]
            select_parts = ", ".join(
                f"COUNT({ref}) AS c{i}" for i, ref in enumerate(col_refs)
            )
            q = sql.SQL("SELECT COUNT(*) AS total, {} FROM {}.{}").format(
                sql.SQL(select_parts),
                sql.Identifier(table.schema),
                sql.Identifier(table.name),
            )
            with self.conn.cursor() as cur:
                cur.execute(q)
                row = cur.fetchone()
                if row:
                    total = row[0]
                    for i, col in enumerate(table.columns):
                        non_null = row[i + 1]
                        nulls = total - (non_null or 0)
                        null_pct = round((nulls / total) * 100, 2) if total else 0.0
                        stats[col.name] = ColumnStatistics(
                            status="MEASURED",
                            total_rows=total,
                            nulls=nulls,
                            null_pct=null_pct,
                        )
        except Exception as exc:
            logger.debug("null stats failed for %s: %s", table.qualified_name, exc)
            for col in table.columns:
                stats[col.name] = ColumnStatistics(
                    status="UNKNOWN",
                    reason="query_failed",
                )
        return stats

    # ------------------------------------------------------------------ #

    def extract(self, database_name: Optional[str] = None,
                schemas: Optional[List[str]] = None) -> Database:
        schemas = schemas or self.list_schemas()
        if not schemas:
            return Database(database=database_name or "unknown", db_type=self.db_type)

        db = Database(database=database_name or "unknown", db_type=self.db_type)
        for schema_name in schemas:
            db.schemas[schema_name] = Schema(name=schema_name)

        tables_index = self._base_tables(schemas)
        row_counts = self._row_counts(tables_index)

        # Populate tables + raw columns.
        table_map: Dict[Tuple[str, str], Table] = {}
        for sch, tbl in tables_index:
            t = Table(schema=sch, name=tbl, row_count=row_counts.get((sch, tbl)))
            table_map[(sch, tbl)] = t
            db.schemas[sch].tables[tbl] = t

        for r in self._column_rows(schemas):
            t = table_map.get((r["table_schema"], r["table_name"]))
            if t is None:
                continue
            t.columns.append(Column(
                name=r["column_name"],
                data_type=r["data_type"],
                normalized_type=self._normalize_type(r["data_type"]),
                nullable=r["is_nullable"] == "YES",
                ordinal_position=r["ordinal_position"],
            ))

        # Primary keys.
        pk_order: Dict[Tuple[str, str], int] = {}
        for r in self._pk_rows(schemas):
            t = table_map.get((r["table_schema"], r["table_name"]))
            if t is None:
                continue
            t.primary_key.append(r["column_name"])
            pk_order[(r["table_schema"], r["table_name"], r["column_name"])] = len(t.primary_key)
        for (sch, tbl, col), _ in pk_order.items():
            c = table_map[(sch, tbl)].column(col)
            if c is not None:
                c.is_primary_key = True

        # Unique constraints.
        for r in self._unique_rows(schemas):
            t = table_map.get((r["table_schema"], r["table_name"]))
            if t is None:
                continue
            u = next((u for u in t.unique_constraints
                      if u.constraint_name == r["constraint_name"]), None)
            if u is None:
                u = UniqueConstraint(constraint_name=r["constraint_name"], table_name=t.name)
                t.unique_constraints.append(u)
            u.columns.append(r["column_name"])
        for t in table_map.values():
            for u in t.unique_constraints:
                for col_name in u.columns:
                    c = t.column(col_name)
                    if c is not None:
                        c.is_part_of_unique_constraint = True

        # Composite-key membership (PK with >1 column).
        for t in table_map.values():
            if len(t.primary_key) > 1:
                for col_name in t.primary_key:
                    c = t.column(col_name)
                    if c is not None:
                        c.is_part_of_composite_key = True

        # Declared FKs (composite-safe, constraint identity preserved).
        for r in self._fk_rows(schemas):
            t = table_map.get((r["table_schema"], r["table_name"]))
            if t is None:
                continue
            fk = ForeignKey(
                constraint_name=r["constraint_name"],
                table_schema=r["table_schema"],
                table_name=r["table_name"],
                column_name=r["column_name"],
                references_schema=r["references_schema"],
                references_table=r["references_table"],
                references_column=r["references_column"],
                ordinal_position=r["ordinal_position"],
            )
            t.foreign_keys.append(fk)
            c = t.column(r["column_name"])
            if c is not None:
                c.declared_foreign_key = fk

        # Column statistics (one batched query per table).
        for t in table_map.values():
            t.statistics = self._null_stats(t)

        return db

    @staticmethod
    def _normalize_type(data_type: str) -> str:
        """Canonical type family shared with the MySQL adapter."""
        t = (data_type or "").lower()
        if t in ("integer", "int4"):
            return "integer"
        if t in ("bigint", "int8"):
            return "bigint"
        if t in ("smallint", "int2"):
            return "smallint"
        if t in ("numeric", "decimal"):
            return "numeric"
        if t in ("real", "float4"):
            return "real"
        if t in ("double precision", "float8"):
            return "float"
        if "money" in t:
            return "money"
        if t in ("text", "character varying", "varchar", "character", "char", "name"):
            return "text"
        if "uuid" in t:
            return "uuid"
        if "timestamp" in t or "date" in t:
            return "datetime" if "time" in t else "date"
        if t in ("boolean", "bool"):
            return "boolean"
        if "bytea" in t or t in ("bytea",):
            return "binary"
        return t or "unknown"