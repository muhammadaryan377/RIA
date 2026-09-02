"""Database adapter boundary (PART 2).

The inference engine depends only on this abstract interface; the concrete
PostgreSQL and MySQL adapters translate their engine's `information_schema` /
catalog structures into the normalized :mod:`schema_engine.models` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from schema_engine.models import Database


class DatabaseAdapter(ABC):
    db_type: str = "base"

    def __init__(self, conn):
        self.conn = conn

    @abstractmethod
    def list_schemas(self) -> List[str]:
        """Schemas/databases that contain at least one base table, best-first."""

    @abstractmethod
    def extract(self, database_name: Optional[str] = None,
                schemas: Optional[List[str]] = None) -> Database:
        """Build the normalized model for the given (or all) schemas."""