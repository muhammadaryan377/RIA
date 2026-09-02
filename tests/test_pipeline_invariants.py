"""Pipeline invariants (PART 20, "Pipeline invariants" group).

These are the eight non-negotiable structural properties of the refactored
architecture:
  * candidate generation never accepts
  * evidence collection never accepts
  * LLM never directly accepts                (covered in test_llm_routing.py)
  * cross-schema inference uses the shared pipeline
  * rejected candidates are retained
  * missing evidence != negative evidence
  * declared FK != inferred FK
  * composite FK mapping is preserved
  * multi-schema identity is preserved        (PART 18)
  * shared-key ambiguity -> UNCERTAIN         (PART 11)
"""

import unittest

from schema_engine.candidates import CandidateGenerator
from schema_engine.evidence import EvidenceCollector
from schema_engine.models import Database, ForeignKey, Schema, Table, Column
from schema_engine.registry import RelationshipRegistry
from schema_engine.serializer import serialize_rejected, serialize_registry

from tests.helpers import fake_profiler


class CandidateGenerationNeverAccepts(unittest.TestCase):
    def test_candidates_are_pure_generation(self):
        tables = {
            "orders": [
                {"column": "customer_id", "data_type": "integer", "nullable": False},
                {"column": "status", "data_type": "text", "nullable": True},
            ],
            "customers": [
                {"column": "id", "data_type": "integer", "nullable": False},
            ],
        }
        primary_keys = {"customers": ["id"]}
        gen = CandidateGenerator(tables, primary_keys, [], schema="public",
                                 db_type="postgresql", profiler=None)
        cands = gen.generate()
        self.assertTrue(cands)
        for c in cands:
            self.assertEqual(c.candidate_id, f"c{cands.index(c) + 1}")
            self.assertFalse(hasattr(c, "relationship_state"))
            self.assertFalse(hasattr(c, "accepted"))
            self.assertFalse(hasattr(c, "confidence_score"))
            self.assertTrue(hasattr(c, "generation_reasons"))
            self.assertTrue(c.generation_reasons)

    def test_lexical_target_is_name_signal(self):
        tables = {
            "orders": [{"column": "customer_id", "data_type": "integer", "nullable": False}],
            "customers": [{"column": "id", "data_type": "integer", "nullable": False}],
        }
        primary_keys = {"customers": ["id"]}
        gen = CandidateGenerator(tables, primary_keys, [], schema="public", db_type="postgresql")
        cands = gen.generate()
        orders = [c for c in cands if c.source_table == "orders"][0]
        self.assertIn("customers", orders.name_targets_from_name)

    def test_structural_lead_is_not_name_signal(self):
        # PART 18: a column that is a member of another table's COMPOSITE PK is
        # a structural lead, never a genuine LEXICAL signal.
        tables = {
            "inventory": [{"column": "personid", "data_type": "integer", "nullable": False}],
            "person": [{"column": "personid", "data_type": "integer", "nullable": False}],
            "contact": [
                {"column": "personid", "data_type": "integer", "nullable": False},
                {"column": "contactid", "data_type": "integer", "nullable": False},
            ],
        }
        primary_keys = {"person": ["personid"], "contact": ["personid", "contactid"]}
        gen = CandidateGenerator(tables, primary_keys, [], schema="public", db_type="postgresql")
        cands = gen.generate()
        inv = [c for c in cands if c.source_table == "inventory"][0]
        # the composite-key holder is searchable but does not count as a name hit
        self.assertIn("contact", inv.name_targets)
        self.assertNotIn("contact", inv.name_targets_from_name)


class EvidenceCollectionNeverAccepts(unittest.TestCase):
    def test_hit_is_evidence_not_decision(self):
        pk_index = [("customers", "id", {"1", "2", "3"})]
        p = fake_profiler(
            src={("orders", "customer_id"): {"1", "2", "3"}},
            hit=("customers", "id", 1.0, 3, False),
            pk_index=pk_index,
        )
        collector = EvidenceCollector(p, {"customers": ["id"]})
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "customer_id",
            "name_targets": {"customers"}, "name_targets_from_name": {"customers"},
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        hit = collector.collect(cgroup)
        self.assertIsNotNone(hit)
        self.assertFalse(hasattr(hit, "relationship_state"))
        self.assertFalse(hasattr(hit, "accepted"))
        self.assertEqual(hit.target_table, "customers")
        self.assertTrue(hit.name_hint)


class MissingEvidenceIsUnknownNotNegative(unittest.TestCase):
    def test_budget_exhausted_containment_is_unknown(self):
        pk_index = [("customers", "id", {"1", "2", "3"})]
        p = fake_profiler(
            src={("orders", "customer_id"): {"1", "2", "3"}},
            hit=("customers", "id", 1.0, 3, False),
            exhausted=True,
            pk_index=pk_index,
        )
        collector = EvidenceCollector(p, {"customers": ["id"]})
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "customer_id",
            "name_targets": {"customers"}, "name_targets_from_name": {"customers"},
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        hit = collector.collect(cgroup)
        sig = hit.signals["value_containment"]
        self.assertEqual(sig["status"], "UNKNOWN")  # not FALSE
        self.assertEqual(sig["kind"], "SAMPLED")

    def test_no_connection_containment_is_unknown(self):
        collector = EvidenceCollector(None, {"customers": ["id"]})
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "customer_id",
            "name_targets": set(), "name_targets_from_name": set(),
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        signals = collector._signals(cgroup, "customers", "id", 1.0, "UNKNOWN")
        self.assertEqual(signals["value_containment"]["status"], "UNKNOWN")
        self.assertEqual(signals["value_containment"]["reason"], "no_connection")
        # unavailable semantic metadata is UNKNOWN, never FALSE
        self.assertEqual(signals["table_semantic_similarity"]["status"], "UNKNOWN")


class DeclaredSeparateFromInferred(unittest.TestCase):
    def test_buckets_never_merge(self):
        reg = RelationshipRegistry()
        reg.add_declared([{"table_name": "orders", "column_name": "customer_id",
                           "references_table": "customers", "references_column": "id"}])
        cand = type("C", (), {
            "candidate_id": "c1", "source_table": "payments",
            "source_column": "order_id",
        })()
        rel = {"table": "payments", "column": "order_id",
               "references_table": "orders", "references_column": "order_id",
               "relationship_state": "CONFIRMED", "confidence_score": 90}
        reg.add_inferred(cand, rel, ["data_confirmed"])
        out = serialize_registry(reg)
        self.assertEqual(len(out["declared_relationships"]), 1)
        self.assertEqual(len(out["inferred_relationships"]), 1)
        self.assertEqual(out["inferred_relationships"][0]["table"], "payments")


class RejectedRetained(unittest.TestCase):
    def test_rejected_candidates_survive(self):
        reg = RelationshipRegistry()
        cand = type("C", (), {
            "candidate_id": "c7", "source_table": "orders",
            "source_column": "status",
        })()
        reg.add_rejected(cand, ["code_like_without_name_corroboration"])
        self.assertEqual(len(reg.rejected), 1)
        self.assertEqual(reg.rejected[0].state, "REJECTED")
        self.assertIn("code_like_without_name_corroboration", reg.rejected[0].reasons)
        self.assertEqual(reg.summary()["rejected"], 1)
        ser = serialize_rejected(reg)
        self.assertEqual(ser[0]["source"], "orders.status")
        self.assertEqual(ser[0]["reasons"],
                         ["code_like_without_name_corroboration"])


class CompositeFkMappingPreserved(unittest.TestCase):
    def test_ordinal_mapping_and_identity(self):
        fk = ForeignKey(
            constraint_name="fk_order_line",
            table_schema="public", table_name="order_lines",
            column_name="order_id",
            references_schema="public", references_table="orders",
            references_column="order_id", ordinal_position=1,
        )
        self.assertEqual(fk.qualified_source, "public.order_lines")
        self.assertEqual(fk.source_key(), ("public.order_lines", "order_id"))
        d = fk.as_declared_dict()
        self.assertEqual(d["table_name"], "public.order_lines")
        self.assertEqual(d["references_column"], "order_id")
        self.assertTrue(d["same_schema"])

    def test_composite_pk_survives_normalization(self):
        db = Database(database="x", db_type="postgresql")
        sch = Schema(name="public")
        t = Table(schema="public", name="order_lines",
                  primary_key=["order_id", "line_no"])
        sch.tables["order_lines"] = t
        db.schemas["public"] = sch
        self.assertEqual(db.primary_keys_dict()["public.order_lines"],
                         ["order_id", "line_no"])


class MultiSchemaIdentityPreserved(unittest.TestCase):
    def test_qualified_tables_never_collide(self):
        tables = {
            "sales.orders": [
                {"column": "order_id", "data_type": "integer", "nullable": False},
                {"column": "customer_id", "data_type": "integer", "nullable": False},
            ],
            "hr.orders": [
                {"column": "order_id", "data_type": "integer", "nullable": False},
                {"column": "employee_id", "data_type": "integer", "nullable": False},
            ],
            "sales.customers": [{"column": "id", "data_type": "integer", "nullable": False}],
            "hr.employees": [{"column": "id", "data_type": "integer", "nullable": False}],
        }
        primary_keys = {"sales.orders": ["order_id"], "hr.orders": ["order_id"],
                        "sales.customers": ["id"], "hr.employees": ["id"]}
        gen = CandidateGenerator(tables, primary_keys, [], schema="", db_type="postgresql")
        cands = gen.generate()
        sources = {c.source_table for c in cands}
        self.assertIn("sales.orders", sources)
        self.assertIn("hr.orders", sources)  # both survive, no collision
        by_col = {c.source_column: c.name_targets for c in cands}
        # the same-named `id` PKs in different schemas never cross-match
        self.assertEqual(by_col["customer_id"], {"sales.customers"})
        self.assertEqual(by_col["employee_id"], {"hr.employees"})


class SharedKeyIsUncertain(unittest.TestCase):
    def test_shared_domain_target_marked_ambiguous(self):
        # customer_id is a key column of TWO tables -> the data cannot name the
        # real holder; the collector marks the hit ambiguous (PART 11).
        table_pk = {"a": ["id"], "b": ["id"], "orders": []}
        pk_index = [("a", "id", {"1", "2", "3"}), ("b", "id", {"1", "2", "3"})]
        p = fake_profiler(
            src={("orders", "a_id"): {"1", "2", "3"}},
            hit=("b", "id", 1.0, 3, False),
            pk_index=pk_index,
        )
        collector = EvidenceCollector(p, table_pk)
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "a_id",
            "name_targets": {"a", "b"}, "name_targets_from_name": {"a"},
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        hit = collector.collect(cgroup)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.ambiguous)

    def test_genuine_name_hit_breaks_the_tie(self):
        # A real table-name match resolves the shared-key ambiguity (PART 11).
        table_pk = {"a": ["id"], "b": ["id"], "orders": []}
        pk_index = [("a", "id", {"1", "2", "3"}), ("b", "id", {"1", "2", "3"})]
        p = fake_profiler(
            src={("orders", "a_id"): {"1", "2", "3"}},
            hit=("a", "id", 1.0, 3, False),
            pk_index=pk_index,
        )
        collector = EvidenceCollector(p, table_pk)
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "a_id",
            "name_targets": {"a", "b"}, "name_targets_from_name": {"a"},
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        hit = collector.collect(cgroup)
        self.assertIsNotNone(hit)
        self.assertFalse(hit.ambiguous)


class CrossSchemaUsesSharedPipeline(unittest.TestCase):
    def _make_mapping(self):
        return {
            "tables": {
                "a.orders": {
                    "columns": [{"column": "customer_id", "data_type": "integer",
                                 "nullable": False}],
                    "primary_key": ["order_id"], "inferred_primary_key": [],
                    "empty": False,
                },
                "b.customers": {
                    "columns": [{"column": "id", "data_type": "integer",
                                 "nullable": False}],
                    "primary_key": ["id"], "inferred_primary_key": [],
                    "empty": False,
                },
            },
            "inferred_relationships": [],
            "declared_relationships": [],
            "relationship_edges": [],
        }

    def test_cross_schema_uses_shared_pipeline_and_state(self):
        import schema_agent
        import schema_engine.relationships as relmod

        real = relmod.infer_relationships

        def fake_pipeline(tables, primary_keys, declared, **kw):
            rel = {
                "table": "a.orders", "column": "customer_id",
                "references_table": "b.customers", "references_column": "id",
                "schema": "a", "references_schema": "b",
                "relationship_state": "CONFIRMED", "confidence": "data-confirmed",
                "confidence_score": 90, "review_status": "auto-accept",
                "note": "overlap", "same_schema": False,
            }
            return [rel], [], type("R", (), {"rejected": []})()

        relmod.infer_relationships = fake_pipeline
        try:
            mapping = self._make_mapping()
            schema_agent._infer_cross_schema_relationships(mapping, conn=object(),
                                                           db_type="postgresql")
        finally:
            relmod.infer_relationships = real

        self.assertEqual(len(mapping["inferred_relationships"]), 1)
        edge = mapping["inferred_relationships"][0]
        self.assertEqual(edge["references_table"], "b.customers")
        self.assertIn("cross-schema", edge["note"])

    def test_cross_schema_uncertain_never_emitted(self):
        import schema_agent
        import schema_engine.relationships as relmod

        real = relmod.infer_relationships

        def fake_pipeline(tables, primary_keys, declared, **kw):
            rel = {
                "table": "a.orders", "column": "customer_id",
                "references_table": "b.customers", "references_column": "id",
                "schema": "a", "references_schema": "b",
                "relationship_state": "UNCERTAIN", "confidence_score": 40,
                "review_status": "review", "note": "weak",
            }
            return [rel], [], type("R", (), {"rejected": []})()

        relmod.infer_relationships = fake_pipeline
        try:
            mapping = self._make_mapping()
            schema_agent._infer_cross_schema_relationships(mapping, conn=object(),
                                                           db_type="postgresql")
        finally:
            relmod.infer_relationships = real

        self.assertEqual(mapping["inferred_relationships"], [])
        self.assertEqual(len(mapping.get("_cross_schema_review", [])), 1)


if __name__ == "__main__":
    unittest.main()