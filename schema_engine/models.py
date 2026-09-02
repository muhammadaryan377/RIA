"""Normalized relational-schema model (PART 2 of the refactor spec).

The inference engine must not depend directly on PostgreSQL-specific or
MySQL-specific structures.  Every database adapter produces these objects so
the downstream pipeline (candidates -> evidence -> classifier -> policy ->
registry -> serializer) is engine-agnostic.

Schema identity is `schema.table.column` (PART 18): two identically-named
tables in different schemas are distinct objects and never collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ForeignKey:
    """A DECLARED foreign-key constraint (authoritative metadata, PART 3).

    A composite FK keeps its constraint identity plus the ordinal mapping
    between source and referenced columns, so it is never accidentally split
    into unrelated single-column relationships (PART 10).
    """

    constraint_name: str
    table_schema: str
    table_name: str
    column_name: str
    references_schema: str
    references_table: str
    references_column: str
    ordinal_position: int  # 1-based position within the composite constraint

    @property
    def qualified_source(self) -> str:
        return f"{self.table_schema}.{self.table_name}"

    @property
    def qualified_target(self) -> str:
        return f"{self.references_schema}.{self.references_table}"

    def source_key(self) -> Tuple[str, str]:
        return (self.qualified_source, self.column_name)

    def as_declared_dict(self) -> dict:
        """Legacy shape consumed by the current serializer/goal agent."""
        return {
            "table_name": self.qualified_source,
            "column_name": self.column_name,
            "references_table": self.qualified_target,
            "references_column": self.references_column,
            "schema": self.table_schema,
            "references_schema": self.references_schema,
            "same_schema": bool(self.table_schema == self.references_schema),
        }


@dataclass
class UniqueConstraint:
    constraint_name: str
    table_name: str
    columns: List[str] = field(default_factory=list)


@dataclass
class ColumnStatistics:
    """Profiling statistics for a column.

    Missing data is represented as `status == "UNKNOWN"`, never as a zero/
    false value (PART 7 and PART 8): unavailable evidence is not negative
    evidence.
    """

    status: str = "UNKNOWN"  # "MEASURED" | "UNKNOWN"
    total_rows: Optional[int] = None
    nulls: Optional[int] = None
    null_pct: Optional[float] = None
    distinct_count: Optional[int] = None
    reason: Optional[str] = None  # e.g. "profiling_budget_exhausted"


@dataclass
class Column:
    name: str
    data_type: str
    normalized_type: str
    nullable: bool
    ordinal_position: int
    is_primary_key: bool = False
    is_part_of_unique_constraint: bool = False
    is_part_of_composite_key: bool = False
    declared_foreign_key: Optional[ForeignKey] = None
    statistics: Optional[ColumnStatistics] = None

    @property
    def is_unique(self) -> bool:
        """A column is a valid FK target when it is a sole PK/UNIQUE column
        or a member of any declared uniqueness constraint."""
        return self.is_primary_key or self.is_part_of_unique_constraint


@dataclass
class Table:
    schema: str
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    unique_constraints: List[UniqueConstraint] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    row_count: Optional[int] = None
    statistics: Dict[str, ColumnStatistics] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str) -> Optional[Column]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def columns_by_name(self) -> Dict[str, Column]:
        return {col.name: col for col in self.columns}

    def is_empty(self) -> bool:
        return self.row_count is not None and self.row_count == 0


@dataclass
class Schema:
    name: str
    tables: Dict[str, Table] = field(default_factory=dict)


@dataclass
class Database:
    database: str
    db_type: str
    schemas: Dict[str, Schema] = field(default_factory=dict)

    def all_tables(self) -> Dict[str, Table]:
        """Qualified table name -> Table, across every schema."""
        out: Dict[str, Table] = {}
        for sch in self.schemas.values():
            for table in sch.tables.values():
                out[table.qualified_name] = table
        return out

    def table(self, qualified_name: str) -> Optional[Table]:
        return self.all_tables().get(qualified_name)

    # -- legacy dict shapes (parity with current schema_agent extraction) ----

    def tables_dict(self) -> Dict[str, list]:
        """{qualified_table: [{column, data_type, nullable}, ...]}."""
        return {
            t.qualified_name: [
                {"column": c.name, "data_type": c.data_type, "nullable": c.nullable}
                for c in t.columns
            ]
            for t in self.all_tables().values()
        }

    def primary_keys_dict(self) -> Dict[str, List[str]]:
        return {
            t.qualified_name: list(t.primary_key)
            for t in self.all_tables().values()
            if t.primary_key
        }

    def unique_keys_dict(self) -> Dict[str, List[List[str]]]:
        return {
            t.qualified_name: [list(u.columns) for u in t.unique_constraints]
            for t in self.all_tables().values()
            if t.unique_constraints
        }

    def declared_fks_dict(self) -> List[dict]:
        return [fk.as_declared_dict() for t in self.all_tables().values()
                for fk in t.foreign_keys]