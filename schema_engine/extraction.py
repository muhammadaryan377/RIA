"""Extraction orchestration: database connection -> normalized model."""

from __future__ import annotations

from typing import List, Optional

from schema_engine.adapters import ADAPTERS
from schema_engine.models import Database


def extract_database(conn, db_type: str = "postgresql",
                     database_name: Optional[str] = None,
                     schemas: Optional[List[str]] = None) -> Database:
    """Extract the normalized model for a connected database.

    `schemas` optionally restricts which schemas are mapped (PostgreSQL).
    For MySQL the connected database name is used as the schema.
    """
    adapter_cls = ADAPTERS.get(db_type)
    if adapter_cls is None:
        raise ValueError(f"Unsupported db_type {db_type!r}; choose from {sorted(ADAPTERS)}")
    return adapter_cls(conn).extract(database_name=database_name, schemas=schemas)