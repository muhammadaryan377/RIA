"""Database adapters for the schema_engine package."""

from schema_engine.adapters.base import DatabaseAdapter
from schema_engine.adapters.mysql import MySQLAdapter
from schema_engine.adapters.postgres import PostgreSQLAdapter

ADAPTERS = {
    "postgresql": PostgreSQLAdapter,
    "mysql": MySQLAdapter,
}

__all__ = ["DatabaseAdapter", "PostgreSQLAdapter", "MySQLAdapter", "ADAPTERS"]