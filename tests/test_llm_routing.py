"""LLM reasoning routing tests (RULE 5 / PART 16).

The LLM is never an acceptance authority. Every suggestion it returns must be
validated against the real schema and routed through the deterministic
classifier + single AcceptancePolicy. These tests drive the real
`_apply_schema_reasoning` merge logic with fabricated LLM responses.
"""

import unittest

import schema_agent
from schema_agent import _apply_schema_reasoning, _build_reasoning_brief


def _mapping():
    return {
        "database": "x",
        "schema": "public",
        "tables": {
            "orders": {
                "columns": [
                    {"column": "customer_id", "data_type": "integer", "nullable": False},
                    {"column": "region_id", "data_type": "integer", "nullable": True},
                    {"column": "total", "data_type": "numeric", "nullable": True},
                    {"column": "description", "data_type": "text", "nullable": True},
                    {"column": "status", "data_type": "text", "nullable": True},
                ],
                "primary_key": ["order_id"],
                "inferred_primary_key": [],
                "null_stats": {"customer_id": {"null_pct": 0.0}},
            },
            "customers": {
                "columns": [
                    {"column": "id", "data_type": "integer", "nullable": False},
                    {"column": "phone", "data_type": "text", "nullable": True},
                ],
                "primary_key": ["id"],
                "inferred_primary_key": [],
                "null_stats": {},
            },
        },
        "declared_relationships": [],
        "inferred_relationships": [
            {"table": "orders", "column": "customer_id",
             "references_table": "customers", "references_column": "id",
             "confidence": "data-confirmed", "confidence_score": 80,
             "relationship_state": "CONFIRMED", "review_status": "auto-accept"},
        ],
        "ambiguous_relationships": [],
        "relationship_edges": [
            {"source_table": "orders", "source_column": "customer_id",
             "target_table": "customers", "target_column": "id",
             "type": "inferred", "confidence": "data-confirmed",
             "confidence_score": 80, "relationship_state": "CONFIRMED",
             "review_status": "auto-accept"},
        ],
    }


class LlmCannotHallucinate(unittest.TestCase):
    def test_ghost_table_ignored(self):
        m = _mapping()
        parsed = {"relationships": [
            {"table": "ghost", "column": "x", "references_table": "customers",
             "references_column": "id", "kind": "add"},
        ]}
        _apply_schema_reasoning(m, parsed, schema="public")
        self.assertEqual(len(m["inferred_relationships"]), 1)

    def test_ghost_column_ignored(self):
        m = _mapping()
        parsed = {"relationships": [
            {"table": "orders", "column": "not_a_column",
             "references_table": "customers", "references_column": "id",
             "kind": "add"},
        ]}
        _apply_schema_reasoning(m, parsed, schema="public")
        self.assertEqual(len(m["inferred_relationships"]), 1)

    def test_non_reference_column_skipped(self):
        # a plain text description is never a FK, even when the LLM says so
        m = _mapping()
        parsed = {"relationships": [
            {"table": "orders", "column": "description",
             "references_table": "customers", "references_column": "id",
             "kind": "confirm"},
        ]}
        _apply_schema_reasoning(m, parsed, schema="public")
        self.assertEqual(len(m["inferred_relationships"]), 1)


class LlmRoutesThroughPolicy(unittest.TestCase):
    def test_confirm_corroborates_via_policy(self):
        m = _mapping()
        parsed = {"relationships": [
            {"table": "orders", "column": "customer_id",
             "references_table": "customers", "references_column": "id",
             "kind": "confirm"},
        ]}
        _apply_schema_reasoning(m, parsed, schema="public")
        rel = m["inferred_relationships"][0]
        self.assertEqual(rel["confidence"], "llm-confirmed")
        # corroboration re-applied the policy; the state recomputed
        self.assertIn("LLM reasoning corroborates", " ".join(rel["evidence"]))
        self.assertEqual(rel["relationship_state"], "CONFIRMED")

    def test_add_without_corroboration_goes_to_review(self):
        # RULE 5: a bare LLM "add" with no structural/data corroboration cannot
        # become an accepted FK; the policy sends it to the review stash.
        m = _mapping()
        m["inferred_relationships"] = []  # no existing edge
        m["relationship_edges"] = []
        parsed = {"relationships": [
            {"table": "orders", "column": "region_id",
             "references_table": "customers", "references_column": "phone",
             "kind": "add"},
        ]}
        _apply_schema_reasoning(m, parsed, schema="public")
        self.assertEqual(m["inferred_relationships"], [])
        stash = m.get("_llm_review", [])
        self.assertEqual(len(stash), 1)
        self.assertEqual(stash[0]["relationship_state"], "UNCERTAIN")

    def test_table_enrichment_preserved(self):
        m = _mapping()
        parsed = {"tables": {
            "orders": {"description": "sales orders", "measures": ["total"],
                       "dimensions": ["status"]},
        }}
        _apply_schema_reasoning(m, parsed, schema="public")
        orders = m["tables"]["orders"]
        self.assertEqual(orders["description"], "sales orders")
        self.assertEqual(orders["semantic_tags"]["measures"], ["total"])
        self.assertIn("status", orders["semantic_tags"]["dimensions"])


class BriefIsFocusedNotLossyWholeSchema(unittest.TestCase):
    def test_brief_is_bounded_and_candidate_centric(self):
        m = _mapping()
        # a large table that is NOT under review must not be column-dumped
        m["tables"]["audit_log"] = {
            "columns": [{"column": f"col_{i}", "data_type": "text", "nullable": True}
                        for i in range(200)],
            "primary_key": ["col_0"], "inferred_primary_key": [],
            "null_stats": {},
        }
        brief = _build_reasoning_brief(m)
        self.assertIn("TABLES:", brief)
        self.assertIn("audit_log (PK: col_0)", brief)
        self.assertIn("CANDIDATES FOR SEMANTIC REVIEW (1):", brief)
        self.assertIn("orders.customer_id -> customers.id", brief)
        # the 200-column audit log is listed by name only, not dumped
        self.assertNotIn("col_199", brief)
        # the whole schema is never passed with an arbitrary character cap
        self.assertLess(len(brief), 4000)


if __name__ == "__main__":
    unittest.main()