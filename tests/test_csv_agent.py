"""CSV orchestration tests: schema mapping format, SQLite loading, and a
Goal Agent integration test on the loaded CSV (dialect=sqlite).

Hermetic: no live LLM or relational database required.
"""

import json
import os
import sqlite3
import tempfile
import unittest

from csv_orchestrator import build_csv_schema_mapping, load_csv_to_sqlite
from goal_agent import GoalAgent

SALES_CSV = (
    "order_id,sku,amount,active,order_date\n"
    "1,00123,10.50,true,2026-08-01\n"
    "2,00456,20.00,false,2026-08-02\n"
    "3,00789,,true,2026-08-03\n"
)


def _write(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


class BuildCsvSchemaMapping(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.csv = _write(self._tmp.name, "sales.csv", SALES_CSV)
        self.mapping = build_csv_schema_mapping(self.csv)

    def test_goal_agent_compatible_shape(self):
        self.assertEqual(self.mapping["schema"], "file")
        self.assertEqual(self.mapping["declared_relationships"], [])
        self.assertEqual(self.mapping["inferred_relationships"], [])
        self.assertIn("sales", self.mapping["tables"])

    def test_columns_typed_and_nullable(self):
        table = self.mapping["tables"]["sales"]
        cols = {c["column"]: c for c in table["columns"]}
        self.assertEqual(cols["order_id"]["data_type"], "INTEGER")
        self.assertEqual(cols["sku"]["data_type"], "TEXT")           # leading zeros -> string
        self.assertEqual(cols["amount"]["data_type"], "REAL")
        self.assertEqual(cols["active"]["data_type"], "INTEGER")     # boolean
        self.assertEqual(cols["order_date"]["data_type"], "TEXT")    # ISO date
        self.assertFalse(cols["order_id"]["nullable"])
        self.assertTrue(cols["amount"]["nullable"])

    def test_candidate_key_becomes_inferred_pk(self):
        table = self.mapping["tables"]["sales"]
        self.assertEqual(table["inferred_primary_key"], ["order_id"])
        self.assertEqual(table["primary_key"], [])
        self.assertEqual(table["row_count"], 3)
        self.assertFalse(table["empty"])

    def test_quality_and_source_included(self):
        table = self.mapping["tables"]["sales"]
        self.assertIn("csv_profile", table)
        self.assertEqual(table["csv_profile"]["delimiter"], ",")
        self.assertIn("quality", table["csv_profile"])
        self.assertEqual(table["source"], self.csv)


class LoadCsvToSqlite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.csv = _write(self._tmp.name, "sales.csv", SALES_CSV)
        self.db = os.path.join(self._tmp.name, "sales.db")
        self.uri, self.table, self.rows = load_csv_to_sqlite(self.csv, self.db)

    def test_uri_table_rows(self):
        self.assertEqual(self.uri, f"sqlite:///{self.db}")
        self.assertEqual(self.table, "sales")
        self.assertEqual(self.rows, 3)

    def test_values_typed_and_null_preserving(self):
        conn = sqlite3.connect(self.db)
        try:
            row1 = conn.execute("SELECT order_id, sku, amount, active FROM sales WHERE order_id = 1").fetchone()
            self.assertEqual(row1, (1, "00123", 10.5, 1))
            row2 = conn.execute("SELECT order_id, amount FROM sales WHERE order_id = 2").fetchone()
            self.assertEqual(row2, (2, 20.0))
            nulls = conn.execute("SELECT amount FROM sales WHERE order_id = 3").fetchone()
            self.assertIsNone(nulls[0])
        finally:
            conn.close()

    def test_ddl_matches_mapping(self):
        conn = sqlite3.connect(self.db)
        try:
            ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sales'").fetchone()[0]
        finally:
            conn.close()
        self.assertIn('"order_id" INTEGER', ddl)
        self.assertIn('"amount" REAL', ddl)


class CsvGoalAgentIntegration(unittest.TestCase):
    """End-to-end: CSV -> mapping + SQLite -> Goal Agent answers a goal."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.csv = _write(self.dir, "sales.csv", SALES_CSV)
        self.mapping = build_csv_schema_mapping(self.csv)
        self.schema_file = os.path.join(self.dir, "schema.json")
        with open(self.schema_file, "w", encoding="utf-8") as fh:
            json.dump(self.mapping, fh)
        self.uri, self.table, _ = load_csv_to_sqlite(
            self.csv, os.path.join(self.dir, "sales.db")
        )

    def _sql_llm(self):
        class FakeSQLGenerator:
            def complete(self, role, prompt, **kw):
                return "SELECT SUM(amount) AS total FROM sales;"
        return FakeSQLGenerator()

    def test_goal_agent_answers_on_csv_sqlite(self):
        agent = GoalAgent(
            schema_json_path=self.schema_file, db_uri=self.uri,
            provider=self._sql_llm(), dialect="sqlite",
        )
        try:
            out = os.path.join(self.dir, "out.json")
            agent.process_goal("total sales", output_path=out)
        finally:
            agent.engine.dispose()
        with open(out, encoding="utf-8") as fh:
            result = json.load(fh)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["data"][0]["total"], 30.5)
        self.assertEqual(result["sql_used"], "SELECT SUM(amount) AS total FROM sales;")

    def test_goal_agent_normalizes_csv_mapping(self):
        agent = GoalAgent(
            schema_json_path=self.schema_file, db_uri=self.uri,
            provider=object(), dialect="sqlite",
        )
        try:
            self.assertIn("sales", agent.tables)
            self.assertEqual(agent.tables["sales"]["primary_key"], ["order_id"])
            counts = agent._row_counts()
        finally:
            agent.engine.dispose()
        self.assertEqual(counts["sales"], 3)


if __name__ == "__main__":
    unittest.main()