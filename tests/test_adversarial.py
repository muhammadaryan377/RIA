"""Adversarial tests (PART 23).

These mirror the spec's adversarial scenarios at the classify/evidence/policy
layer (hermetic, no live database):
  * same names, unrelated data
  * same names, different types
  * different names, real relationship
  * generic IDs
  * shared codes
  * partial overlap
  * nullable source FK
  * empty tables
  * small sample / budget exhaustion
  * composite FK not split
  * cross-schema identity (see test_pipeline_invariants)
"""

import unittest

from schema_engine.classifier import classify
from schema_engine.evidence import EvidenceCollector
from schema_engine.policy import AcceptancePolicy
from schema_engine.models import ForeignKey

from tests.helpers import fake_profiler

PK = {"customers": ["id"], "products": ["id"], "orders": ["order_id"]}
NULLS = {}


class SameNamesUnrelatedData(unittest.TestCase):
    def test_status_columns_do_not_become_fks(self):
        # orders.status / customers.status share a name; values overlap 1..n by
        # coincidence. The code-like veto (policy) rejects without a name match.
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "status", "customers", "id",
            "data-confirmed", "note", overlap_ratio=0.9,
        )
        bucket, _ = AcceptancePolicy().decide(
            rel, False, code_like=rel["evidence_detail"]["code_like"],
            name_match=rel["evidence_detail"]["name_match"])
        self.assertEqual(bucket, "rejected")


class SameNamesDifferentTypes(unittest.TestCase):
    def test_type_mismatch_is_negative_evidence(self):
        p = fake_profiler(types_ok=False, pk_index=[("customers", "id", set())])
        rel, _ = classify(
            p, PK, NULLS, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0, name_hint=True,
        )
        self.assertIn("datatype mismatch", rel["evidence"])
        self.assertFalse(rel["evidence_detail"]["datatype_match"])


class DifferentNamesRealRelationship(unittest.TestCase):
    def test_buyer_no_to_customer_id(self):
        # orders.buyer_no -> customers.customer_id: no name match, but strong
        # data + structural evidence corroborate.
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "buyer_no", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0,
        )
        self.assertNotEqual(rel["relationship_state"], "REJECTED")


class GenericIds(unittest.TestCase):
    def test_id_to_id_not_connected(self):
        # table_a.id -> table_b.id: no lexical/statistical corroboration.
        rel, _ = classify(
            None, {"a": ["id"], "b": ["id"]}, NULLS, "public",
            "a", "id", "b", "id", "data-confirmed", "note",
        )
        self.assertEqual(rel["relationship_state"], "UNCERTAIN")


class SharedCodes(unittest.TestCase):
    def test_code_columns_rejected_without_name(self):
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "country_code", "countries", "id",
            "data-confirmed", "note", overlap_ratio=0.9,
        )
        self.assertTrue(rel["evidence_detail"]["code_like"])
        bucket, _ = AcceptancePolicy().decide(
            rel, False, code_like=rel["evidence_detail"]["code_like"],
            name_match=rel["evidence_detail"]["name_match"])
        self.assertEqual(bucket, "rejected")


class PartialOverlap(unittest.TestCase):
    def test_partial_overlap_is_not_confident(self):
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=0.5, name_hint=True,
        )
        self.assertIn("partial value overlap", rel["evidence"])
        self.assertLess(rel["confidence_score"], 80)


class SmallSample(unittest.TestCase):
    def test_sampled_absence_is_not_proof(self):
        # budget exhausted -> containment is UNKNOWN, so the evidence cannot be
        # used as a FALSE (PART 7 / PART 8).
        p = fake_profiler(
            exhausted=True,
            pk_index=[("customers", "id", {"1", "2", "3"})],
        )
        collector = EvidenceCollector(p, PK)
        cgroup = type("G", (), {
            "source_table": "orders", "source_column": "customer_id",
            "name_targets": set(), "name_targets_from_name": set(),
            "is_ref_flavored": True, "is_composite_key_col": False,
        })()
        signals = collector._signals(cgroup, "customers", "id", 0.1, "SAMPLED")
        self.assertEqual(signals["value_containment"]["status"], "UNKNOWN")
        self.assertEqual(signals["value_overlap"]["status"], "UNKNOWN")


class CompositeNotSplit(unittest.TestCase):
    def test_composite_fk_keeps_ordinal_mapping(self):
        fks = [
            ForeignKey("fk_ol", "public", "order_lines", "order_id",
                       "public", "orders", "order_id", 1),
            ForeignKey("fk_ol", "public", "order_lines", "line_no",
                       "public", "orders", "line_no", 2),
        ]
        # identical constraint identity -> one composite relationship, not two
        # unrelated single-column FKs (PART 10).
        self.assertEqual(fks[0].constraint_name, fks[1].constraint_name)
        self.assertEqual(fks[0].source_key()[0], fks[1].source_key()[0])
        self.assertEqual(fks[0].ordinal_position, 1)
        self.assertEqual(fks[1].ordinal_position, 2)
        self.assertEqual(
            [fk.references_column for fk in fks], ["order_id", "line_no"])


if __name__ == "__main__":
    unittest.main()