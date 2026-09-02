"""MySQL adapter -> normalized model.

The MySQL "schema" concept is the connected database name. Declared FKs are
mapped from `information_schema.key_column_usage` with an explicit
`constraint_name + ordinal_position` ordering so composite FKs are grouped by
constraint identity, never by accidental result order (PART 10).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pymysql.cursors

from schema_engine.adapters.base import DatabaseAdapter
from schema_engine.models import (
    Column, ColumnStatistics, Database, ForeignKey, Schema, Table,
    UniqueConstraint,
)

logger = logging.getLogger("aria.schema_engine.mysql")


class MySQLAdapter(DatabaseAdapter):
    db_type = "mysql"

    def _dict_cursor(self):
        return self.conn.cursor(pymysql.cursors.DictCursor)

    def _quote_ident(self, name: str) -> str:
        return "`" + str(name).replace("`", "") + "`"

    def _database_name(self) -> str:
        with self._dict_cursor() as cur:
            cur.execute("SELECT DATABASE() AS db")
            row = cur.fetchone()
            return (row.get("db") if row else None) or "unknown"

    # ------------------------------------------------------------------ #

    def list_schemas(self) -> List[str]:
        db = self._database_name()
        return [db] if db and db != "unknown" else []

    # ------------------------------------------------------------------ #

    def _base_tables(self, db: str) -> List[str]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                (db,),
            )
            return [r["table_name"] for r in cur.fetchall()]

    def _column_rows(self, db: str) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
                """,
                (db,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _pk_rows(self, db: str) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = %s
                ORDER BY tc.table_name, kcu.ordinal_position
                """,
                (db,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _unique_rows(self, db: str) -> List[dict]:
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT tc.table_name, tc.constraint_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'UNIQUE'
                  AND tc.table_schema = %s
                ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position
                """,
                (db,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _fk_rows(self, db: str) -> List[dict]:
        """Declared FKs with explicit constraint identity + ordinal ordering.

        The result is ordered by (constraint_name, ordinal_position) and the
        composite mapping is reconstructed from that identity in `extract`,
        so a reordered metadata result set cannot corrupt the mapping.
        """
        with self._dict_cursor() as cur:
            cur.execute(
                """
                SELECT
                    kcu.constraint_name,
                    kcu.table_name,
                    kcu.column_name,
                    kcu.referenced_table_name,
                    kcu.referenced_column_name,
                    kcu.ordinal_position
                FROM information_schema.key_column_usage kcu
                WHERE kcu.referenced_table_name IS NOT NULL
                  AND kcu.table_schema = %s
                ORDER BY kcu.constraint_name, kcu.ordinal_position
                """,
                (db,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _row_counts(self, db: str, tables: List[str]) -> Dict[str, Optional[int]]:
        out: Dict[str, Optional[int]] = {}
        for t in tables:
            try:
                with self._dict_cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) AS n FROM {self._quote_ident(db)}.{self._quote_ident(t)}")
                    row = cur.fetchone()
                    out[t] = row["n"] if row else None
            except Exception as exc:
                logger.debug("row count failed for %s: %s", t, exc)
                out[t] = None
        return out

    def _null_stats(self, db: str, table: Table) -> Dict[str, ColumnStatistics]:
        stats: Dict[str, ColumnStatistics] = {}
        if not table.columns:
            return stats
        try:
            col_refs = [self._quote_ident(c.name) for c in table.columns]
            select_parts = ", ".join(
                f"COUNT({ref}) AS c{i}" for i, ref in enumerate(col_refs)
            )
            q = (
                f"SELECT COUNT(*) AS total, {select_parts} "
                f"FROM {self._quote_ident(db)}.{self._quote_ident(table.name)}"
            )
            with self._dict_cursor() as cur:
                cur.execute(q)
                row = cur.fetchone()
            if row:
                total = row["total"]
                for i, col in enumerate(table.columns):
                    non_null = row[f"c{i}"]
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
                stats[col.name] = ColumnStatistics(status="UNKNOWN", reason="query_failed")
        return stats

    # ------------------------------------------------------------------ #

    def extract(self, database_name: Optional[str] = None,
                schemas: Optional[List[str]] = None) -> Database:
        db_name = database_name or self._database_name()
        if schemas:
            db_name = schemas[0]  # MySQL exposes the database as the schema
        db = Database(database=db_name, db_type=self.db_type)
        schema = Schema(name=db_name)
        db.schemas[db_name] = schema

        tables = self._base_tables(db_name)
        if not tables:
            return db
        row_counts = self._row_counts(db_name, tables)

        table_map: Dict[str, Table] = {}
        for tbl in tables:
            t = Table(schema=db_name, name=tbl, row_count=row_counts.get(tbl))
            table_map[tbl] = t
            schema.tables[tbl] = t

        for r in self._column_rows(db_name):
            t = table_map.get(r["table_name"])
            if t is None:
                continue
            t.columns.append(Column(
                name=r["column_name"],
                data_type=r["data_type"],
                normalized_type=self._normalize_type(r["data_type"]),
                nullable=r["is_nullable"] == "YES",
                ordinal_position=r["ordinal_position"],
            ))

        for r in self._pk_rows(db_name):
            t = table_map.get(r["table_name"])
            if t is None:
                continue
            t.primary_key.append(r["column_name"])
            c = t.column(r["column_name"])
            if c is not None:
                c.is_primary_key = True

        for r in self._unique_rows(db_name):
            t = table_map.get(r["table_name"])
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

        for t in table_map.values():
            if len(t.primary_key) > 1:
                for col_name in t.primary_key:
                    c = t.column(col_name)
                    if c is not None:
                        c.is_part_of_composite_key = True

        # Reconstruct composite FKs by constraint identity (never result order).
        fk_rows = self._fk_rows(db_name)
        by_constraint: Dict[str, List[dict]] = {}
        for r in fk_rows:
            by_constraint.setdefault(r["constraint_name"], []).append(r)
        for rows in by_constraint.values():
            rows.sort(key=lambda r: r["ordinal_position"])
            first = rows[0]
            t = table_map.get(first["table_name"])
            if t is None:
                continue
            for r in rows:
                fk = ForeignKey(
                    constraint_name=r["constraint_name"],
                    table_schema=db_name,
                    table_name=r["table_name"],
                    column_name=r["column_name"],
                    references_schema=db_name,
                    references_table=r["referenced_table_name"],
                    references_column=r["referenced_column_name"],
                    ordinal_position=r["ordinal_position"],
                )
                t.foreign_keys.append(fk)
                c = t.column(r["column_name"])
                if c is not None:
                    c.declared_foreign_key = fk

        for t in table_map.values():
            t.statistics = self._null_stats(db_name, t)

        return db

    @staticmethod
    def _normalize_type(data_type: str) -> str:
        """Canonical type family shared with the PostgreSQL adapter."""
        t = (data_type or "").lower()
        if t in ("int", "integer"):
            return "integer"
        if t in ("bigint",):
            return "bigint"
        if t in ("smallint", "tinyint"):
            return "smallint"
        if t in ("decimal", "numeric"):
            return "numeric"
        if t in ("float", "double", "real", "double precision"):
            return "float"
        if t in ("char", "varchar", "text", "tinytext", "mediumtext", "longtext"):
            return "text"
        if t in ("char", "character"):
            return "text"
        if t in ("date",):
            return "date"
        if t in ("datetime", "timestamp"):
            return "datetime"
        if t in ("bool", "boolean"):
            return "boolean"
        if t in ("binary", "varbinary", "blob", "tinyblob", "mediumblob", "longblob"):
            return "binary"
        return t or "unknown"