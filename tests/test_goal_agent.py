"""Goal Agent contract + guard tests (spec: goal.txt).

Hermetic: no live LLM or database is required. The Goal Agent is exercised with
a FakeLLM and a temp schema_mapping.json; the pure-logic layers under test are
the spec's deterministic guards and the output JSON contract:

  * clarification gate       (spec §3  -> status=needs_clarification)
  * read-only SQL guard      (spec §9  -> no destructive SQL)
  * UNCERTAIN relationship   (spec §5  -> warnings, never silent)
  * JSON contract            (spec §12 -> status / execution_time_ms / warnings)
  * intent / analysis type   (spec §2)
  * plan parsing             (spec §6)
"""

import json
import os
import tempfile
import unittest

from goal_agent import GoalAgent

DUMMY_URI = "postgresql://u:p@localhost:5432/nonexistent"


def write_mapping(mapping, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    return path


def base_mapping():
    """Two-table schema, one CONFIRMED relationship (orders.customer_id)."""
    return {
        "schema": "public",
        "tables": {
            "customers": {"columns": ["id", "name", "city"], "primary_key": ["id"]},
            "orders": {"columns": ["order_id", "customer_id", "total"], "primary_key": ["order_id"]},
        },
        "relationship_edges": [],
        "declared_relationships": [],
        "inferred_relationships": [{
            "table": "orders", "column": "customer_id",
            "references_table": "customers", "references_column": "id",
            "relationship_state": "CONFIRMED", "confidence_score": 90,
        }],
        "ambiguous_relationships": [],
    }


def make_agent(mapping, tmpdir):
    schema_file = write_mapping(mapping, os.path.join(tmpdir, "schema.json"))
    return GoalAgent(
        schema_json_path=schema_file, db_uri=DUMMY_URI,
        provider=object(), dialect="postgresql",
    )


def typed_mapping():
    """customers + orders with real data types and a declared FK, so measure /
    date detection works."""
    return {
        "schema": "public",
        "tables": {
            "customers": {
                "columns": [
                    {"column": "id", "data_type": "integer", "nullable": False},
                    {"column": "name", "data_type": "text", "nullable": False},
                    {"column": "city", "data_type": "text", "nullable": True},
                ],
                "primary_key": ["id"],
            },
            "orders": {
                "columns": [
                    {"column": "order_id", "data_type": "integer", "nullable": False},
                    {"column": "customer_id", "data_type": "integer", "nullable": False},
                    {"column": "total", "data_type": "real", "nullable": True},
                    {"column": "order_date", "data_type": "timestamp", "nullable": True},
                ],
                "primary_key": ["order_id"],
            },
        },
        "relationship_edges": [],
        "declared_relationships": [{
            "table_name": "orders", "column_name": "customer_id",
            "references_table": "customers", "references_column": "id",
        }],
        "inferred_relationships": [],
        "ambiguous_relationships": [],
    }


class FakeLLM:
    """Minimal provider shape; the tested paths never call it."""
    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called in this test path")

    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called in this test path")


class GoalAgentSetup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(base_mapping(), self.dir)

    def test_loads_schema_tables_and_relationships(self):
        self.assertEqual(set(self.agent.tables), {"customers", "orders"})
        self.assertEqual(self.agent.tables["orders"]["foreign_keys"][0]["referenced_table"], "customers")

    def test_confirmed_edge_is_not_uncertain(self):
        self.assertEqual(self.agent.uncertain_edge_keys, set())


class ClarificationGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(base_mapping(), self.dir)

    def test_ambiguous_goal_needs_clarification(self):
        needs, question = self.agent._needs_clarification("explain the weather forecast")
        self.assertTrue(needs)
        self.assertIsInstance(question, str)

    def test_schema_matching_goal_does_not_ask(self):
        needs, _ = self.agent._needs_clarification("show total sales by customer")
        self.assertFalse(needs)
        needs, _ = self.agent._needs_clarification("how many orders")
        self.assertFalse(needs)

    def test_analytical_concept_goal_does_not_ask(self):
        # Spec §7: a metric concept + operation must not be pre-blocked.
        needs, _ = self.agent._needs_clarification("overall sales")
        self.assertFalse(needs)

    def test_process_goal_returns_needs_clarification_contract(self):
        out = os.path.join(self.dir, "processed.json")
        path = self.agent.process_goal("explain the weather forecast", output_path=out)
        with open(path, encoding="utf-8") as fh:
            contract = json.load(fh)
        self.assertEqual(contract["status"], "needs_clarification")
        self.assertEqual(contract["sql"], None)
        self.assertEqual(contract["data"], [])
        self.assertEqual(contract["execution"]["success"], False)
        self.assertIn("goal", contract)
        self.assertIn("question", contract)


class ReadOnlyGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.agent = make_agent(base_mapping(), self._tmp.name)

    def test_accepts_select_and_cte(self):
        self.assertTrue(self.agent._is_read_only_sql("SELECT * FROM orders;"))
        self.assertTrue(self.agent._is_read_only_sql(
            "WITH t AS (SELECT * FROM orders) SELECT count(*) FROM t;"))

    def test_rejects_destructive_statements(self):
        for sql in (
            "INSERT INTO orders (order_id) VALUES (1);",
            "UPDATE orders SET total = 0 WHERE order_id = 1;",
            "DELETE FROM orders;",
            "DROP TABLE orders;",
            "ALTER TABLE orders ADD COLUMN x int;",
            "CREATE TABLE z (id int);",
            "TRUNCATE TABLE orders;",
            "GRANT ALL ON orders TO public;",
            "CALL some_proc();",
            "MERGE INTO orders USING x ON true WHEN MATCHED THEN DELETE;",
        ):
            self.assertFalse(self.agent._is_read_only_sql(sql), sql)

    def test_string_literal_containing_keyword_is_safe(self):
        # A destructive keyword inside a string literal is data, not SQL.
        self.assertTrue(self.agent._is_read_only_sql(
            "SELECT * FROM customers WHERE name = 'update me now';"))


class UncertainRelationshipWarnings(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        mapping = base_mapping()
        mapping["ambiguous_relationships"] = [{
            "table": "orders", "column": "customer_id",
            "references_table": "customers", "references_column": "id",
        }]
        self.agent = make_agent(mapping, self.dir)

    def test_uncertain_edge_collected(self):
        self.assertIn("orders.customer_id->customers.id", self.agent.uncertain_edge_keys)

    def test_warning_emitted_when_uncertain_edge_used(self):
        sql = "SELECT o.order_id FROM orders o JOIN customers c ON o.customer_id = c.id"
        warnings = self.agent._warnings_for_sql(sql)
        self.assertEqual(len(warnings), 1)
        self.assertIn("UNCERTAIN", warnings[0])

    def test_no_warning_when_uncertain_edge_not_in_query(self):
        warnings = self.agent._warnings_for_sql("SELECT * FROM customers;")
        self.assertEqual(warnings, [])


class OutputContract(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(base_mapping(), self.dir)

    def test_contract_has_spec_fields(self):
        contract = self.agent._build_output_contract(
            "total sales by customer",
            {"kpis": [{"aggregate": "SUM", "description": "sales"}], "dimensions": ["customers"]},
            ["orders", "customers"], "SELECT 1;",
            [{"order_id": 1}], success=True, message=None,
            status="success", warnings=["w"], execution_time_ms=12,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["execution"]["row_count"], 1)
        self.assertEqual(contract["execution"]["execution_time_ms"], 12)
        self.assertEqual(contract["warnings"], ["w"])
        for key in ("goal", "data_selection", "analysis_plan", "sql", "data",
                    "suggested_questions", "message"):
            self.assertIn(key, contract)
        self.assertIn("original_question", contract["goal"])
        self.assertIn("intent", contract["goal"])
        self.assertIn("analysis_type", contract["goal"])
        self.assertEqual(contract["data_selection"]["tables"], ["orders", "customers"])
        self.assertIn("order_by", contract["analysis_plan"])
        self.assertIn("limit", contract["analysis_plan"])

    def test_query_failed_status(self):
        contract = self.agent._build_output_contract(
            "x", {"kpis": [], "dimensions": []}, [], None, [],
            success=False, message="boom", status="query_failed",
        )
        self.assertEqual(contract["status"], "query_failed")
        self.assertEqual(contract["execution"]["success"], False)


class SuspiciousPkJoinDetection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        mapping = base_mapping()
        mapping["tables"]["products"] = {
            "columns": ["product_id", "product_name"], "primary_key": ["product_id"],
        }
        mapping["tables"]["order_items"] = {
            "columns": ["order_id", "product_id", "quantity"],
            "primary_key": ["order_id", "product_id"],
        }
        mapping["declared_relationships"].append({
            "table_name": "order_items", "column_name": "product_id",
            "references_table": "products", "references_column": "product_id",
        })
        mapping["declared_relationships"].append({
            "table_name": "order_items", "column_name": "order_id",
            "references_table": "orders", "references_column": "order_id",
        })
        self.agent = make_agent(mapping, self.dir)

    def test_composite_pk_fk_member_is_not_suspicious(self):
        # order_items PK is (order_id, product_id); product_id is an FK to
        # products, so joining on it is legitimate, not a PK=PK correlation.
        sql = "SELECT oi.quantity FROM order_items oi JOIN products p ON oi.product_id = p.product_id"
        self.assertEqual(self.agent._find_suspicious_pk_joins(sql), [])

    def test_true_pk_pk_join_still_flagged(self):
        # products.product_id = orders.order_id has no FK edge -> flagged.
        sql = ("SELECT oi.quantity FROM order_items oi "
               "JOIN orders o ON oi.order_id = o.order_id "
               "JOIN products p ON p.product_id = o.order_id")
        suspects = self.agent._find_suspicious_pk_joins(sql)
        self.assertTrue(
            any("product_id = orders.order_id" in s for s in suspects),
            suspects,
        )


class IntentAndPlanParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.agent = make_agent(base_mapping(), self._tmp.name)

    def test_detect_intent(self):
        self.assertEqual(self.agent._detect_intent("top 5 products"), "ranking")
        self.assertEqual(self.agent._detect_intent("compare regions"), "comparison")
        self.assertEqual(self.agent._detect_intent("sales trend over time"), "trend")
        self.assertEqual(self.agent._detect_intent("count of orders"), "distribution")

    def test_detect_analysis_type(self):
        self.assertEqual(self.agent._detect_analysis_type("forecast sales"), "predictive")
        self.assertEqual(self.agent._detect_analysis_type("why did sales drop"), "diagnostic")
        self.assertEqual(self.agent._detect_analysis_type("recommend which products to stock"), "prescriptive")
        self.assertEqual(self.agent._detect_analysis_type("sales by month"), "descriptive")

    def test_parse_sql_plan(self):
        sql = "SELECT c.name, SUM(o.total) FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.name ORDER BY SUM(o.total) DESC LIMIT 5"
        order_by, limit = self.agent._parse_sql_plan(sql)
        self.assertEqual(order_by, ["SUM(o.total) DESC"])
        self.assertEqual(limit, 5)

    def test_clarify_contract_shape(self):
        out = os.path.join(self._tmp.name, "clarify.json")
        self.agent._clarify("Show me sales performance", "which metric?", out)
        with open(out, encoding="utf-8") as fh:
            c = json.load(fh)
        self.assertEqual(c["status"], "needs_clarification")
        self.assertEqual(c["analysis_plan"]["measures"], [])


class TypoCorrection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(base_mapping(), self.dir)

    def test_correct_goal_typo(self):
        corrected, corrections = self.agent._correct_goal("show orders by cutomer")
        self.assertEqual(corrected, "show orders by customer")
        self.assertIn(("cutomer", "customer"), corrections)

    def test_correct_goal_no_change_when_clean(self):
        corrected, corrections = self.agent._correct_goal("show orders by customer")
        self.assertEqual(corrected, "show orders by customer")
        self.assertEqual(corrections, [])

    def test_correct_goal_plural_is_exact_not_typo(self):
        corrected, corrections = self.agent._correct_goal("customers list")
        self.assertEqual(corrected, "customers list")
        self.assertEqual(corrections, [])

    def test_correct_goal_weak_match_untouched(self):
        corrected, corrections = self.agent._correct_goal("sales performance analysis")
        self.assertEqual(corrected, "sales performance analysis")
        self.assertEqual(corrections, [])


class TypoProcessGoalOnSqlite(unittest.TestCase):
    def setUp(self):
        from sqlalchemy import create_engine, text
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.db = os.path.join(self.dir, "test.db")
        self._engine = create_engine(f"sqlite:///{self.db}")
        with self._engine.begin() as conn:
            conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT)"))
            conn.execute(text("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL)"))
            conn.execute(text("INSERT INTO customers (id, name, city) VALUES (1, 'a', 'x')"))
            conn.execute(text("INSERT INTO orders (order_id, customer_id, total) VALUES (10, 1, 5.0)"))

    def _sql_llm(self):
        class FakeSQLGenerator:
            def complete(self, role, prompt, **kw):
                return "SELECT COUNT(*) AS cnt FROM customers;"
        return FakeSQLGenerator()

    def test_process_goal_auto_corrects_and_warns(self):
        schema_file = write_mapping(base_mapping(), os.path.join(self.dir, "schema.json"))
        agent = GoalAgent(
            schema_json_path=schema_file, db_uri=f"sqlite:///{self.db}",
            provider=self._sql_llm(), dialect="sqlite",
        )
        out = os.path.join(self.dir, "out.json")
        path = agent.process_goal("how many cutomers", output_path=out)
        agent.engine.dispose()
        self._engine.dispose()
        with open(path, encoding="utf-8") as fh:
            c = json.load(fh)
        self.assertEqual(c["status"], "success")
        self.assertEqual(c["goal"]["original_question"], "how many cutomers")
        self.assertEqual(c["row_count"], 1)
        self.assertTrue(any("Did you mean 'customers' for 'cutomers'" in w for w in c["warnings"]), c["warnings"])


class NaturalLanguageSuggestions(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        mapping = typed_mapping()
        self.agent = make_agent(mapping, self.dir)

    def test_suggestions_are_natural_english(self):
        suggestions = self.agent._template_suggestions(limit=10)
        self.assertTrue(suggestions)
        for s in suggestions:
            self.assertNotIn(".", s, s)  # no table.column identifiers
        self.assertIn("What are the top customers by order value?", suggestions)
        self.assertIn("Which customers contribute the most to order value?", suggestions)
        self.assertIn("What is the average order value per customer?", suggestions)
        self.assertIn("What are the most important metrics for customers?", suggestions)
        self.assertIn("What is the trend in order value over time?", suggestions)

    def test_suggestions_cap_respected(self):
        suggestions = self.agent._template_suggestions(limit=3)
        self.assertEqual(len(suggestions), 3)


class MetricsOverview(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(typed_mapping(), self.dir)

    def test_detects_open_ended_metric_goals(self):
        self.assertEqual(
            self.agent._detect_metrics_overview(
                "what are the most important metrics for customers"
            ),
            "customers",
        )
        self.assertEqual(
            self.agent._detect_metrics_overview("key metrics for orders"),
            "orders",
        )

    def test_concrete_goals_are_not_overviews(self):
        for goal in ("total sales by customer", "how many orders",
                     "count of orders", "orders from germany"):
            self.assertIsNone(self.agent._detect_metrics_overview(goal), goal)

    def test_build_metrics_overview(self):
        metrics = self.agent._build_metrics_overview("customers")
        labels = [label for label, _ in metrics]
        self.assertEqual(
            labels,
            ["Total customers", "Total orders", "Total order value"],
        )

    def test_process_goal_overview_on_sqlite_no_llm(self):
        from sqlalchemy import create_engine, text
        db = os.path.join(self.dir, "test.db")
        engine = create_engine(f"sqlite:///{db}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT)"))
            conn.execute(text("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL)"))
            conn.execute(text("INSERT INTO customers (id, name, city) VALUES (1, 'a', 'x')"))
            conn.execute(text("INSERT INTO orders (order_id, customer_id, total) VALUES (10, 1, 5.0)"))

        schema_file = write_mapping(typed_mapping(), os.path.join(self.dir, "schema.json"))
        agent = GoalAgent(
            schema_json_path=schema_file, db_uri=f"sqlite:///{db}",
            provider=FakeLLM(), dialect="sqlite",
        )
        out = os.path.join(self.dir, "out.json")
        path = agent.process_goal("most important metrics for customer", output_path=out)
        agent.engine.dispose()
        engine.dispose()
        with open(path, encoding="utf-8") as fh:
            c = json.load(fh)
        self.assertEqual(c["status"], "success")
        self.assertEqual(c["row_count"], 3)
        self.assertEqual(c["goal"]["original_question"], "most important metrics for customer")
        self.assertIsInstance(c["sql_used"], str)
        self.assertEqual(c["data"][2]["metric"], "Total order value")
        self.assertEqual(c["data"][2]["value"], 5.0)
        self.assertIn("overview", c["message"].lower())


class GoalIR(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(typed_mapping(), self.dir)

    def test_ir_extracts_aggregation_ranking_time(self):
        ir = self.agent._build_goal_ir("top 5 customers by total sales in 2024")
        self.assertEqual(ir["intent"], "ranking")
        self.assertEqual(ir["aggregation"], "SUM")
        self.assertEqual(ir["ranking"]["limit"], 5)
        self.assertEqual(ir["ranking"]["direction"], "desc")
        self.assertEqual(ir["time"]["value"], 2024)
        self.assertTrue(any(m["concept"] == "sales KPI" for m in ir["metrics"]))
        self.assertTrue(any(m["aggregation"] == "SUM" for m in ir["metrics"]))

    def test_ir_growth_is_trend_not_avg(self):
        ir = self.agent._build_goal_ir("monthly sales growth over time")
        self.assertEqual(ir["intent"], "trend")
        self.assertEqual(ir["analysis_type"], "trend_analysis")
        growth = [m for m in ir["metrics"] if m["concept"] == "growth KPI"]
        self.assertTrue(growth)
        self.assertIsNone(growth[0]["aggregation"])

    def test_ir_serializable(self):
        ir = self.agent._build_goal_ir("compare total revenue among regions")
        json.dumps(ir)
        self.assertEqual(ir["comparison"]["dimension"], None)

    def test_semantic_resolution_binds_metric_to_measure(self):
        ir = self.agent._build_goal_ir("total sales by customer")
        join_path = self.agent._determine_join_path(
            self.agent._get_relevant_tables("total sales by customer"))
        ir = self.agent._resolve_semantics("total sales by customer", ir, join_path)
        sales = [m for m in ir["metrics"] if m["concept"] == "sales KPI"]
        self.assertEqual(sales[0]["resolved_table"], "orders")
        self.assertEqual(sales[0]["resolved_column"], "total")
        self.assertEqual(sales[0]["resolved_expression"], "orders.total")
        self.assertTrue(ir["measures"])

    def test_semantic_resolution_binds_time_column(self):
        ir = self.agent._build_goal_ir("total sales in 2024")
        join_path = ["orders", "customers"]
        ir = self.agent._resolve_semantics("total sales in 2024", ir, join_path)
        self.assertEqual(ir["time"]["column"], "orders.order_date")
        year_filters = [f for f in ir["filters"] if f["value"].startswith("2024")]
        self.assertEqual(len(year_filters), 2)
        self.assertEqual(year_filters[0]["column"], "orders.order_date")

    def test_semantic_resolution_skips_fk_and_pk_measures(self):
        ir = self.agent._build_goal_ir("count orders")
        join_path = ["orders"]
        ir = self.agent._resolve_semantics("count orders", ir, join_path)
        cols = [m["column"] for m in ir["measures"]]
        self.assertNotIn("customer_id", cols)
        self.assertNotIn("order_id", cols)
        self.assertIn("total", cols)

    def test_ir_has_required_tables_and_join_plan_fields(self):
        ir = self.agent._build_goal_ir("total sales by customers in 2024")
        self.assertEqual(ir["required_tables"], [])
        self.assertEqual(ir["join_plan"], [])
        join_path = self.agent._determine_join_path(
            self.agent._get_relevant_tables("total sales by customers in 2024"))
        ir = self.agent._resolve_semantics("total sales by customers in 2024", ir, join_path)
        plan = self.agent._plan_joins(self.agent._required_tables_from_ir(ir, join_path))
        ir["join_plan"] = plan["edges"]
        self.assertTrue(ir["required_tables"])
        self.assertTrue(all(t in ir["required_tables"] for t in ("orders", "customers")))
        self.assertTrue(ir["join_plan"])
        for e in ir["join_plan"]:
            self.assertIn("cardinality", e)


class JoinPlan(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        mapping = {
            "schema": "public",
            "tables": {
                "customers": {"columns": ["id"], "primary_key": ["id"]},
                "orders": {"columns": ["order_id", "customer_id"], "primary_key": ["order_id"]},
                "payments": {"columns": ["payment_id", "order_id", "amount"], "primary_key": ["payment_id"]},
                "unrelated": {"columns": ["x"], "primary_key": ["x"]},
            },
            "relationship_edges": [],
            "declared_relationships": [
                {"table_name": "orders", "column_name": "customer_id",
                 "references_table": "customers", "references_column": "id"},
                {"table_name": "payments", "column_name": "order_id",
                 "references_table": "orders", "references_column": "order_id"},
            ],
            "inferred_relationships": [],
            "ambiguous_relationships": [],
        }
        self.agent = make_agent(mapping, self.dir)

    def test_plan_covers_required_tables_with_declared_edges(self):
        plan = self.agent._plan_joins(["customers", "orders", "payments"])
        self.assertEqual(plan["required_tables"], ["customers", "orders", "payments"])
        endpoints = set()
        for e in plan["edges"]:
            self.assertEqual(e["trust"], "DECLARED")
            endpoints.add(e["left_table"])
            endpoints.add(e["right_table"])
        self.assertTrue({"customers", "orders", "payments"} <= endpoints)
        self.assertNotIn("unrelated", endpoints)

    def test_plan_nodes_form_a_connected_path(self):
        plan = self.agent._plan_joins(["customers", "payments"])
        self.assertEqual(plan["nodes"], ["customers", "orders", "payments"])
        self.assertEqual(len(plan["edges"]), 2)

    def test_plan_edges_carry_cardinality(self):
        plan = self.agent._plan_joins(["customers", "orders"])
        self.assertEqual(len(plan["edges"]), 1)
        e = plan["edges"][0]
        self.assertEqual(e["cardinality"], "many_to_one")
        self.assertEqual(e["left_table"], "orders")
        self.assertEqual(e["right_table"], "customers")

    def test_plan_falls_back_without_edges_for_disconnected(self):
        plan = self.agent._plan_joins(["customers", "unrelated"])
        self.assertEqual(plan["edges"], [])


class SqlTruthExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(typed_mapping(), self.dir)

    def test_extract_filters_from_where(self):
        sql = ("SELECT c.city, SUM(o.total) AS s FROM orders o "
               "JOIN customers c ON o.customer_id = c.id "
               "WHERE o.order_date >= '2024-01-01' AND o.order_date <= '2024-12-31' "
               "GROUP BY c.city")
        filters = self.agent._extract_sql_filters(sql)
        self.assertEqual(len(filters), 2)
        self.assertEqual(filters[0]["column"], "o.order_date")
        self.assertEqual(filters[0]["operator"], ">=")
        self.assertEqual(filters[0]["value"], "'2024-01-01'")

    def test_extract_group_by(self):
        sql = ("SELECT c.city, COUNT(*) AS n FROM orders o "
               "JOIN customers c ON o.customer_id = c.id GROUP BY c.city ORDER BY n DESC")
        self.assertEqual(self.agent._extract_sql_group_by(sql), ["c.city"])

    def test_extract_joins_from_on(self):
        sql = ("SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id")
        joins = self.agent._extract_sql_joins(sql)
        self.assertEqual(len(joins), 1)
        self.assertEqual(joins[0]["left_table"], "orders")
        self.assertEqual(joins[0]["left_column"], "customer_id")
        self.assertEqual(joins[0]["right_table"], "customers")
        self.assertEqual(joins[0]["right_column"], "id")

    def test_contract_reflects_sql_filters_not_empty(self):
        contract = self.agent._build_output_contract(
            "total sales in 2024",
            {"kpis": [{"aggregate": "SUM", "description": "sales"}], "dimensions": []},
            ["orders"], "SELECT SUM(total) FROM orders WHERE order_date >= '2024-01-01'",
            [{"sum": 5}], success=True, message=None, status="success",
            warnings=[], execution_time_ms=1,
        )
        self.assertEqual(len(contract["data_selection"]["filters"]), 1)
        self.assertEqual(contract["validation"]["structural"], True)
        self.assertIn("validation", contract)
        self.assertIn("semantic", contract["validation"])
        self.assertIn("relationship", contract["validation"])

    def test_extract_measures_from_select(self):
        sql = ("SELECT c.city, COUNT(*) AS n, SUM(o.total) AS s FROM orders o "
               "JOIN customers c ON o.customer_id = c.id GROUP BY c.city")
        measures = self.agent._extract_sql_measures(sql)
        self.assertIn("COUNT(*) AS n", measures)
        self.assertIn("SUM(o.total) AS s", measures)
        self.assertNotIn("c.city", measures)

    def test_contract_uses_sql_measures(self):
        contract = self.agent._build_output_contract(
            "total sales in 2024",
            {"kpis": [{"aggregate": "SUM", "description": "sales"}], "dimensions": []},
            ["orders"],
            "SELECT SUM(total) AS s FROM orders WHERE order_date >= '2024-01-01'",
            [{"sum": 5}], success=True, message=None, status="success",
            warnings=[], execution_time_ms=1,
        )
        self.assertIn("SUM", contract["analysis_plan"]["aggregation"])
        self.assertIn("SUM(total) AS s", contract["analysis_plan"]["measures"])


class SemanticValidation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(typed_mapping(), self.dir)
        self.ir = self.agent._build_goal_ir("total sales in 2024")
        join_path = ["orders", "customers"]
        self.ir = self.agent._resolve_semantics("total sales in 2024", self.ir, join_path)
        self.plan = self.agent._plan_joins(["orders", "customers"])
        self.sql = (
            "SELECT SUM(o.total) AS s FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "WHERE o.order_date >= '2024-01-01' AND o.order_date <= '2024-12-31'"
        )

    def test_passes_when_sql_implements_plan(self):
        issues, warnings = self.agent._semantic_validate_sql(
            self.ir, self.sql, ["orders", "customers"], self.plan
        )
        self.assertEqual(issues, [])
        self.assertEqual(warnings, [])

    def test_flags_missing_metric_aggregation(self):
        issues, _ = self.agent._semantic_validate_sql(
            self.ir, "SELECT o.total FROM orders o", ["orders", "customers"], self.plan
        )
        self.assertTrue(any("SUM" in i for i in issues), issues)

    def test_flags_missing_measure_column(self):
        issues, _ = self.agent._semantic_validate_sql(
            self.ir, "SELECT SUM(o.customer_id) AS s FROM orders o", ["orders", "customers"], self.plan
        )
        self.assertTrue(any("total" in i for i in issues), issues)

    def test_flags_missing_filter_on_date_column(self):
        issues, _ = self.agent._semantic_validate_sql(
            self.ir, "SELECT SUM(o.total) AS s FROM orders o", ["orders", "customers"], self.plan
        )
        self.assertTrue(any("order_date" in i for i in issues), issues)

    def test_flags_missing_planned_join(self):
        issues, _ = self.agent._semantic_validate_sql(
            self.ir, "SELECT SUM(total) AS s FROM orders", ["orders", "customers"], self.plan
        )
        self.assertTrue(any("join" in i.lower() for i in issues), issues)

    def test_ranking_requires_order_by_and_limit(self):
        ir = self.agent._build_goal_ir("top 5 customers by total sales")
        join_path = ["orders", "customers"]
        ir = self.agent._resolve_semantics(ir["original_goal"], ir, join_path)
        plan = self.agent._plan_joins(["orders", "customers"])
        issues, _ = self.agent._semantic_validate_sql(
            ir, "SELECT c.name, SUM(o.total) AS s FROM orders o "
                "JOIN customers c ON o.customer_id = c.id GROUP BY c.name",
            join_path, plan
        )
        self.assertTrue(any("LIMIT" in i for i in issues), issues)
        self.assertTrue(any("ORDER BY" in i for i in issues), issues)

    def test_extra_table_is_warning_not_failure(self):
        sql = ("SELECT SUM(o.total) AS s FROM orders o "
               "JOIN customers c ON o.customer_id = c.id "
               "JOIN unrelated u ON u.id = o.customer_id "
               "WHERE o.order_date >= '2024-01-01' AND o.order_date <= '2024-12-31'")
        issues, warnings = self.agent._semantic_validate_sql(
            self.ir, sql, ["orders", "customers"], self.plan
        )
        self.assertEqual(issues, [])
        self.assertTrue(any("outside" in w for w in warnings), warnings)

    def test_kpi_enrichment_resolves_to_real_column(self):
        kpi_map = {"kpis": [{"aggregate": "SUM", "description": "sales KPI",
                             "match": "sales"}], "dimensions": []}
        enriched = self.agent._enrich_kpi_map(kpi_map, self.ir)
        self.assertEqual(enriched["kpis"][0]["resolved_column"], "orders.total")

    def test_fan_out_warns_when_metric_table_is_ancestor_of_grain(self):
        ir = self.agent._build_goal_ir("total sales by customer in 2024")
        join_path = ["orders", "customers"]
        ir = self.agent._resolve_semantics(ir["original_goal"], ir, join_path)
        plan = self.agent._plan_joins(["orders", "customers"])
        ir["metrics"] = [
            {"concept": "sales KPI", "aggregation": "SUM",
             "resolved_table": "customers", "resolved_column": "id",
             "resolved_expression": "customers.id"}
        ]
        ir["dimensions"] = [
            {"concept": "orders", "resolved_table": "orders", "resolved_column": None}
        ]
        sql = ("SELECT o.order_id, SUM(c.id) AS s FROM orders o "
               "JOIN customers c ON o.customer_id = c.id GROUP BY o.order_id")
        _, warnings = self.agent._semantic_validate_sql(ir, sql, join_path, plan)
        self.assertTrue(any("fan-out" in w for w in warnings), warnings)

    def test_fan_out_silent_for_same_table(self):
        ir = self.agent._build_goal_ir("total sales in 2024")
        join_path = ["orders", "customers"]
        ir = self.agent._resolve_semantics(ir["original_goal"], ir, join_path)
        plan = self.agent._plan_joins(["orders", "customers"])
        sql = ("SELECT SUM(o.total) AS s FROM orders o "
               "JOIN customers c ON o.customer_id = c.id "
               "WHERE o.order_date >= '2024-01-01' AND o.order_date <= '2024-12-31'")
        _, warnings = self.agent._semantic_validate_sql(ir, sql, join_path, plan)
        self.assertFalse(any("fan-out" in w for w in warnings), warnings)

    def test_over_grouping_flagged_for_scalar_aggregate(self):
        # "average order value" is a scalar overall aggregate: a GROUP BY in the
        # outer SELECT splits the single AOV into per-group rows (over-grouping),
        # the bug Fix 3 closes.
        ir = {
            "aggregation": "AVG", "intent": "summary", "ranking": None,
            "comparison": None, "dimensions": [],
            "metrics": [{"concept": "average value", "aggregation": "AVG",
                         "resolved_table": "orders", "resolved_column": "orders.total",
                         "resolved_expression": "orders.total"}],
            "filters": [], "time": None,
        }
        plan = self.agent._plan_joins(["orders"])
        sql = ("SELECT AVG(orders.total) AS v FROM orders "
               "GROUP BY orders.customer_id")
        issues, _ = self.agent._semantic_validate_sql(ir, sql, ["orders"], plan)
        self.assertTrue(any("over-grouping" in i for i in issues), issues)

    def test_over_grouping_not_flagged_for_per_entity_breakdown(self):
        # "average order value per customer" is a grouped breakdown: a GROUP BY
        # here is correct, not over-grouping.
        ir = {
            "aggregation": "AVG", "intent": "distribution", "ranking": None,
            "comparison": None,
            "dimensions": [{"concept": "customer", "resolved_table": "customers",
                            "resolved_column": "customers.name"}],
            "metrics": [{"concept": "average value", "aggregation": "AVG",
                         "resolved_table": "orders", "resolved_column": "orders.total",
                         "resolved_expression": "orders.total"}],
            "filters": [], "time": None,
        }
        plan = self.agent._plan_joins(["orders", "customers"])
        sql = ("SELECT customers.name, AVG(orders.total) AS v FROM orders "
               "JOIN customers ON orders.customer_id = customers.id "
               "GROUP BY customers.name")
        issues, _ = self.agent._semantic_validate_sql(ir, sql, ["orders", "customers"], plan)
        self.assertFalse(any("over-grouping" in i for i in issues), issues)

    def test_over_grouping_not_flagged_for_subquery(self):
        # A correct overall-AOV subquery groups INSIDE the subquery; the outer
        # SELECT must not be flagged (no depth-0 GROUP BY).
        ir = {
            "aggregation": "AVG", "intent": "summary", "ranking": None,
            "comparison": None, "dimensions": [],
            "metrics": [{"concept": "average value", "aggregation": "AVG",
                         "resolved_table": "orders", "resolved_column": "orders.total",
                         "resolved_expression": "orders.total"}],
            "filters": [], "time": None,
        }
        plan = self.agent._plan_joins(["orders"])
        sql = ("SELECT AVG(sub.v) AS aov FROM ("
               "SELECT SUM(orders.total) AS v FROM orders GROUP BY orders.order_id"
               ") sub")
        issues, _ = self.agent._semantic_validate_sql(ir, sql, ["orders"], plan)
        self.assertFalse(any("over-grouping" in i for i in issues), issues)


class NullPreservation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.agent = make_agent(typed_mapping(), self.dir)

    def test_nulls_preserved_not_filled(self):
        records = [{"total": 10.0, "city": "Lahore"},
                   {"total": None, "city": "Karachi"}]
        cleaned = self.agent._clean_data(records)
        self.assertEqual(cleaned[1]["total"], None)
        self.assertEqual(cleaned[1]["city"], "Karachi")
        self.assertEqual(self.agent.preprocessing_report["nulls_preserved"], 1)
        self.assertNotIn("nulls_filled", self.agent.preprocessing_report)

    def test_numeric_null_never_becomes_zero(self):
        cleaned = self.agent._clean_data(
            [{"total": 1.0, "city": "x"}, {"total": None, "city": "y"}]
        )
        self.assertEqual(cleaned[1]["total"], None)
        self.assertEqual(cleaned[1]["city"], "y")

    def test_real_values_untouched(self):
        cleaned = self.agent._clean_data([{"total": 5.5, "city": "Karachi"}])
        self.assertEqual(cleaned[0]["total"], 5.5)
        self.assertEqual(cleaned[0]["city"], "Karachi")


class DatabaseGeneralization(unittest.TestCase):
    """Spec §18: three intentionally different schemas (A/B/C) with renamed
    tables and columns. The SAME questions must ground to the analogous
    concepts on every schema, with ZERO Goal Agent code changes between them.
    """

    SCHEMAS = {
        "A": {
            "schema": "public",
            "tables": {
                "customers": {"columns": [
                    {"column": "id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["id"]},
                "orders": {"columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "customer_id", "data_type": "integer"},
                    {"column": "order_status", "data_type": "text"},
                    {"column": "total", "data_type": "numeric"},
                ], "primary_key": ["order_id"]},
                "order_items": {"columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "product_id", "data_type": "integer"},
                    {"column": "quantity", "data_type": "integer"},
                    {"column": "unit_price", "data_type": "numeric"},
                ], "primary_key": ["order_id", "product_id"]},
                "products": {"columns": [
                    {"column": "product_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                    {"column": "brand_id", "data_type": "integer"},
                    {"column": "model_year", "data_type": "integer"},
                ], "primary_key": ["product_id"]},
                "categories": {"columns": [
                    {"column": "category_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["category_id"]},
            },
            "relationship_edges": [],
            "declared_relationships": [
                {"table_name": "orders", "column_name": "customer_id",
                 "references_table": "customers", "references_column": "id"},
                {"table_name": "order_items", "column_name": "order_id",
                 "references_table": "orders", "references_column": "order_id"},
                {"table_name": "order_items", "column_name": "product_id",
                 "references_table": "products", "references_column": "product_id"},
                {"table_name": "products", "column_name": "brand_id",
                 "references_table": "categories", "references_column": "category_id"},
            ],
            "inferred_relationships": [],
            "ambiguous_relationships": [],
        },
        "B": {
            "schema": "public",
            "tables": {
                "clients": {"columns": [
                    {"column": "id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["id"]},
                "transactions": {"columns": [
                    {"column": "transaction_id", "data_type": "integer"},
                    {"column": "client_id", "data_type": "integer"},
                    {"column": "status", "data_type": "text"},
                    {"column": "amount", "data_type": "numeric"},
                ], "primary_key": ["transaction_id"]},
                "transaction_items": {"columns": [
                    {"column": "transaction_id", "data_type": "integer"},
                    {"column": "item_id", "data_type": "integer"},
                    {"column": "quantity", "data_type": "integer"},
                    {"column": "unit_price", "data_type": "numeric"},
                ], "primary_key": ["transaction_id", "item_id"]},
                "items": {"columns": [
                    {"column": "item_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                    {"column": "brand_id", "data_type": "integer"},
                    {"column": "manufacture_year", "data_type": "integer"},
                ], "primary_key": ["item_id"]},
                "departments": {"columns": [
                    {"column": "department_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["department_id"]},
            },
            "relationship_edges": [],
            "declared_relationships": [
                {"table_name": "transactions", "column_name": "client_id",
                 "references_table": "clients", "references_column": "id"},
                {"table_name": "transaction_items", "column_name": "transaction_id",
                 "references_table": "transactions", "references_column": "transaction_id"},
                {"table_name": "transaction_items", "column_name": "item_id",
                 "references_table": "items", "references_column": "item_id"},
                {"table_name": "items", "column_name": "brand_id",
                 "references_table": "departments", "references_column": "department_id"},
            ],
            "inferred_relationships": [],
            "ambiguous_relationships": [],
        },
        "C": {
            "schema": "public",
            "tables": {
                "buyers": {"columns": [
                    {"column": "id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["id"]},
                "sales": {"columns": [
                    {"column": "sale_id", "data_type": "integer"},
                    {"column": "buyer_id", "data_type": "integer"},
                    {"column": "status", "data_type": "text"},
                    {"column": "amount", "data_type": "numeric"},
                ], "primary_key": ["sale_id"]},
                "sale_lines": {"columns": [
                    {"column": "sale_id", "data_type": "integer"},
                    {"column": "product_id", "data_type": "integer"},
                    {"column": "quantity", "data_type": "integer"},
                    {"column": "price", "data_type": "numeric"},
                ], "primary_key": ["sale_id", "product_id"]},
                "catalog": {"columns": [
                    {"column": "product_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                    {"column": "brand_id", "data_type": "integer"},
                    {"column": "model_year", "data_type": "integer"},
                ], "primary_key": ["product_id"]},
                "groups": {"columns": [
                    {"column": "group_id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ], "primary_key": ["group_id"]},
            },
            "relationship_edges": [],
            "declared_relationships": [
                {"table_name": "sales", "column_name": "buyer_id",
                 "references_table": "buyers", "references_column": "id"},
                {"table_name": "sale_lines", "column_name": "sale_id",
                 "references_table": "sales", "references_column": "sale_id"},
                {"table_name": "sale_lines", "column_name": "product_id",
                 "references_table": "catalog", "references_column": "product_id"},
                {"table_name": "catalog", "column_name": "brand_id",
                 "references_table": "groups", "references_column": "group_id"},
            ],
            "inferred_relationships": [],
            "ambiguous_relationships": [],
        },
    }

    # question -> the analogous fact table + status column on each schema
    CASES = [
        ("What are the top customers by order status?",
         {"A": "orders", "B": "transactions", "C": "sales"}),
        ("How many customers?",
         {"A": "customers", "B": "clients", "C": "buyers"}),
        ("What is the overall quantity?",
         {"A": "order_items", "B": "transaction_items", "C": "sale_lines"}),
    ]

    def _agent(self, name):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return make_agent(self.SCHEMAS[name], self._tmp.name)

    def test_same_questions_ground_to_analogous_tables(self):
        for question, expected in self.CASES:
            for name, fact in expected.items():
                ag = self._agent(name)
                ir = ag._build_goal_ir(question)
                jp = ag._determine_join_path(ag._get_relevant_tables(question))
                ir = ag._resolve_semantics(question, ir, jp)
                metric_tables = [m["resolved_table"] for m in ir["metrics"]]
                self.assertIn(
                    fact, metric_tables,
                    f"{question!r} on schema {name} resolved metrics to "
                    f"{metric_tables}, expected the analogous fact {fact}",
                )

    def test_top_by_model_year_dimension_grounds_everywhere(self):
        for name, year_col in (("A", "model_year"), ("B", "manufacture_year"),
                               ("C", "model_year")):
            ag = self._agent(name)
            q = "What are the top brands by model year?"
            ir = ag._build_goal_ir(q)
            jp = ag._determine_join_path(ag._get_relevant_tables(q))
            ir = ag._resolve_semantics(q, ir, jp)
            dim_cols = [d["resolved_column"] for d in ir["dimensions"]]
            self.assertIn(year_col, dim_cols,
                          f"schema {name} did not ground model_year: {dim_cols}")

    def test_product_quantity_metric_grounds_everywhere(self):
        for name, fact in (("A", "order_items"), ("B", "transaction_items"),
                           ("C", "sale_lines")):
            ag = self._agent(name)
            q = "What are the top products by quantity?"
            ir = ag._build_goal_ir(q)
            jp = ag._determine_join_path(ag._get_relevant_tables(q))
            ir = ag._resolve_semantics(q, ir, jp)
            metric_tables = [m["resolved_table"] for m in ir["metrics"]]
            self.assertIn(fact, metric_tables,
                          f"schema {name} resolved {metric_tables}")


class SemanticGrounding(unittest.TestCase):
    """Spec §5/§6/§19: synonym matching, singular/plural, renamed columns,
    indirect relationships, inferred metrics with confidence, and graceful
    ambiguity."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def test_synonym_customers_to_clients(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["B"], self.dir)
        q = "show top customers by order status"
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        dim_tables = [d["resolved_table"] for d in ir["dimensions"]]
        self.assertIn("clients", dim_tables)
        self.assertNotIn("customers", dim_tables)

    def test_singular_grounds_to_plural_table(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["A"], self.dir)
        tables = [t for _s, t in ag._match_tables("customer")]
        self.assertIn("customers", tables)

    def test_renamed_column_model_year(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["B"], self.dir)
        hits = [c for _s, c in ag._match_columns("model year", "items")]
        self.assertIn("manufacture_year", hits)

    def test_indirect_relationship_customer_to_product_quantity(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["A"], self.dir)
        q = "top customers by product quantity"
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        metric = [m for m in ir["metrics"] if m.get("resolved_table")][0]
        self.assertEqual(metric["resolved_table"], "order_items")
        plan = ag._plan_joins(ag._required_tables_from_ir(ir, jp))
        plan_nodes = set(plan["nodes"])
        self.assertTrue({"customers", "orders", "order_items"} <= plan_nodes)

    def test_missing_metric_is_inferred_with_confidence(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["A"], self.dir)
        ir = ag._build_goal_ir("top customers")
        count = [m for m in ir["metrics"] if m["aggregation"] == "COUNT"]
        self.assertTrue(count)
        self.assertTrue(count[0].get("inferred"))
        self.assertLess(count[0].get("confidence", 1.0), 1.0)

    def test_unsupported_words_still_clarify(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["A"], self.dir)
        needs, _ = ag._needs_clarification("explain the dark matter forecast")
        self.assertTrue(needs)

    def test_computed_measure_quantity_times_price(self):
        # Schema C has an explicit amount column -> the natural revenue measure.
        ag = make_agent(DatabaseGeneralization.SCHEMAS["C"], self.dir)
        q = "what is the overall sales?"
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        sales = [m for m in ir["metrics"] if m["resolved_column"] == "amount"]
        self.assertTrue(sales, "sales with an amount column should resolve to it")

        # When there is NO amount-like column, compose quantity * price (§3).
        bare = {
            "schema": "public",
            "tables": {
                "orders": {"columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "customer_id", "data_type": "integer"},
                ], "primary_key": ["order_id"]},
                "order_items": {"columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "product_id", "data_type": "integer"},
                    {"column": "quantity", "data_type": "integer"},
                    {"column": "unit_price", "data_type": "numeric"},
                ], "primary_key": ["order_id", "product_id"]},
            },
            "relationship_edges": [],
            "declared_relationships": [
                {"table_name": "order_items", "column_name": "order_id",
                 "references_table": "orders", "references_column": "order_id"},
            ],
            "inferred_relationships": [],
            "ambiguous_relationships": [],
        }
        ag2 = make_agent(bare, self.dir)
        q2 = "what is the total profit?"
        ir2 = ag2._build_goal_ir(q2)
        jp2 = ag2._determine_join_path(ag2._get_relevant_tables(q2))
        ir2 = ag2._resolve_semantics(q2, ir2, jp2)
        computed = [m for m in ir2["metrics"] if m.get("computed_measure")]
        self.assertTrue(computed, "profit with no matching column must be computed")
        self.assertIn("order_items.quantity", computed[0]["resolved_expression"])
        self.assertIn("order_items.unit_price", computed[0]["resolved_expression"])

    def test_multiple_dimensions_ground(self):
        ag = make_agent(DatabaseGeneralization.SCHEMAS["A"], self.dir)
        q = "top products by category and model year"
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        dims = {(d["resolved_table"], d["resolved_column"]) for d in ir["dimensions"]}
        self.assertTrue(any(t == "products" and c == "model_year" for t, c in dims), dims)
        self.assertTrue(any(t == "categories" for t, c in dims), dims)


if __name__ == "__main__":
    unittest.main()