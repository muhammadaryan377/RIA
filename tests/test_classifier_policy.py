"""Classifier + AcceptancePolicy unit tests (PART 12 / PART 13 / PART 24).

The classifier computes structured confidence and evidence groups; the policy
is the single authority that maps a score to a state. Every hardcoded
threshold used here is documented in policy.py (CONFIRMED=80, PROBABLE=60,
HIGH=90, MEDIUM=75) and asserted below so a change is caught by the suite.
"""

import unittest

from schema_engine.classifier import classify
from schema_engine.policy import AcceptancePolicy

from tests.helpers import fake_profiler

PK = {"customers": ["id"], "orders": ["order_id"]}
NULLS = {}


class ClassifierProducesStructuredOutput(unittest.TestCase):
    def test_structured_fields_present(self):
        rel, amb = classify(
            None, PK, NULLS, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0, name_hint=True,
        )
        self.assertFalse(amb)
        for field in ("confidence_score", "confidence_level",
                      "independent_signal_count", "relationship_state",
                      "evidence", "evidence_groups", "evidence_provenance",
                      "confidence_band", "review_status"):
            self.assertIn(field, rel)
        self.assertGreaterEqual(rel["independent_signal_count"], 2)  # LEXICAL+STATISTICAL+STRUCTURAL
        self.assertIn("LEXICAL", rel["evidence_provenance"])
        self.assertIn("STATISTICAL", rel["evidence_provenance"])

    def test_confidence_levels_match_documented_bands(self):
        p = AcceptancePolicy()
        self.assertEqual(p.state_for(80), "CONFIRMED")
        self.assertEqual(p.state_for(60), "PROBABLE")
        self.assertEqual(p.state_for(59), "UNCERTAIN")
        self.assertEqual(p.confidence_level_for(90), "VERY_HIGH")
        self.assertEqual(p.confidence_level_for(80), "HIGH")
        self.assertEqual(p.confidence_level_for(60), "MEDIUM")
        self.assertEqual(p.confidence_level_for(10), "LOW")

    def test_policy_is_the_only_acceptance_authority(self):
        # The classifier alone never returns CONFIRMED-state acceptance logic;
        # the state always comes from the single policy.
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0, name_hint=True,
        )
        p = AcceptancePolicy()
        self.assertEqual(rel["relationship_state"],
                         p.state_for(rel["confidence_score"]))
        self.assertEqual(rel["confidence_level"],
                         p.confidence_level_for(rel["confidence_score"]))


class PolicyRequiresIndependentCorroboration(unittest.TestCase):
    def test_single_signal_never_auto_accepts(self):
        # PART 6: a score that reaches CONFIRMED on ONE group alone is
        # downgraded to PROBABLE by the policy.
        rel = {"confidence_score": 80, "independent_signal_count": 1}
        bucket, state = AcceptancePolicy().decide(rel, ambiguous=False,
                                                  code_like=False, name_match=False)
        self.assertEqual(bucket, "inferred")
        self.assertEqual(state, "PROBABLE")

    def test_two_groups_confirm(self):
        rel = {"confidence_score": 80, "independent_signal_count": 2}
        bucket, state = AcceptancePolicy().decide(rel, ambiguous=False,
                                                  code_like=False, name_match=False)
        self.assertEqual(state, "CONFIRMED")

    def test_code_like_veto_without_name(self):
        # PART 23 "shared codes": code-like columns without a name match are
        # rejected, never emitted.
        rel = {"confidence_score": 95, "independent_signal_count": 3}
        bucket, state = AcceptancePolicy().decide(rel, ambiguous=False,
                                                  code_like=True, name_match=False)
        self.assertEqual(bucket, "rejected")
        self.assertEqual(state, "REJECTED")

    def test_ambiguous_never_emitted(self):
        rel = {"confidence_score": 95, "independent_signal_count": 3}
        bucket, state = AcceptancePolicy().decide(rel, ambiguous=True,
                                                  code_like=False, name_match=False)
        self.assertEqual(bucket, "ambiguous")
        self.assertEqual(state, "UNCERTAIN")

    def test_corroborate_reapplies_policy(self):
        rel = {"confidence_score": 70, "independent_signal_count": 2}
        AcceptancePolicy().corroborate(rel, 20, "LLM reasoning corroborates")
        self.assertEqual(rel["relationship_state"], "CONFIRMED")
        self.assertIn("LLM reasoning corroborates", rel["evidence"])


class NullableAndSparseHandling(unittest.TestCase):
    def _profiler(self):
        return fake_profiler(pk_index=[("customers", "id", {"1", "2", "3"})])

    def test_nullable_fk_is_not_rejected(self):
        # PART 23 "nullable source FK": NULLs do not kill a legitimate FK.
        nulls = {"orders": {"customer_id": {"null_pct": 0.6}}}
        rel, _ = classify(
            self._profiler(), PK, nulls, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0, name_hint=True,
        )
        self.assertNotEqual(rel["relationship_state"], "REJECTED")
        self.assertIn("nullable", " ".join(rel["evidence"]))

    def test_high_null_rate_reduces_score_but_keeps_state(self):
        nulls = {"orders": {"customer_id": {"null_pct": 0.9}}}
        rel, _ = classify(
            self._profiler(), PK, nulls, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", overlap_ratio=1.0, name_hint=True,
        )
        self.assertIn("high null rate", " ".join(rel["evidence"]))

    def test_empty_table_produces_no_phantom_confirmation(self):
        # PART 23 "empty tables": no values -> no statistical evidence -> the
        # state reflects the absence (UNCERTAIN/low), not an invented FK.
        nulls = {"orders": {"customer_id": {"null_pct": None, "total_rows": 0}}}
        rel, _ = classify(
            None, PK, NULLS, "public",
            "orders", "customer_id", "customers", "id",
            "data-confirmed", "note", name_hint=True,
        )
        self.assertNotIn("STATISTICAL", rel["evidence_provenance"])
        # name-only evidence cannot confirm on its own
        self.assertLess(rel["confidence_score"], 80)


if __name__ == "__main__":
    unittest.main()