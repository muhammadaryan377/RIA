"""Acceptance policy (PART 13 / RULE 6).

The ONE authority that maps a classified relationship to its acceptance state.
No other layer (candidate generation, evidence collection, ranking, sampling,
LLM handling, output emission, cross-schema handling) may accept or reject a
relationship. Cross-schema and LLM suggestions arrive here like any other
candidate (PART 15 / PART 16 / RULE 5).

State vocabulary:
    DECLARED   - explicit database constraint (never competes with inference)
    CONFIRMED  - strong corroborating evidence across independent groups
    PROBABLE   - credible evidence, but not fully corroborated
    UNCERTAIN  - evidence cannot distinguish the relationship (review)
    REJECTED   - vetoed by hard evidence (recorded, never emitted)

Confidence model (PART 12): structured levels are the primary model. The legacy
`review_status` / `confidence_band` / `confidence` strings are derived here too
so downstream agents keep their contract while the engine uses one structured
state.

Thresholds (PART 24 - documented):
  CONFIRMED_SCORE = 80 : strong corroboration across independent groups.
                         A score of 80 is only reachable by combining
                         STRUCTURAL + STATISTICAL (or LEXICAL) evidence.
  PROBABLE_SCORE  = 60 : credible single-line-of-evidence.
  HIGH_BAND_SCORE = 90 : very strong, near-certain.
  MEDIUM_BAND     = 75 : used for the legacy band labels.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class AcceptancePolicy:
    CONFIRMED_SCORE = 80
    PROBABLE_SCORE = 60
    HIGH_BAND_SCORE = 90
    MEDIUM_BAND_SCORE = 75

    # Group names (PART 6). Corroboration is measured across these independent
    # categories; N variations of name matching count as ONE group.
    GROUPS = ("STRUCTURAL", "LEXICAL", "STATISTICAL", "SEMANTIC",
              "DECLARED_METADATA")

    def state_for(self, score: float) -> str:
        if score >= self.CONFIRMED_SCORE:
            return "CONFIRMED"
        if score >= self.PROBABLE_SCORE:
            return "PROBABLE"
        return "UNCERTAIN"

    def confidence_level_for(self, score: float) -> str:
        """Structured confidence (PART 12) - the primary model."""
        if score >= self.HIGH_BAND_SCORE:
            return "VERY_HIGH"
        if score >= self.CONFIRMED_SCORE:
            return "HIGH"
        if score >= self.PROBABLE_SCORE:
            return "MEDIUM"
        return "LOW"

    def review_status_for(self, score: float) -> str:
        if score >= self.CONFIRMED_SCORE:
            return "auto-accept"
        if score >= self.PROBABLE_SCORE:
            return "flagged"
        return "review"

    def band_for(self, score: float) -> str:
        if score >= self.HIGH_BAND_SCORE:
            return "High"
        if score >= self.MEDIUM_BAND_SCORE:
            return "Medium"
        if score >= self.PROBABLE_SCORE:
            return "Possible"
        return "Low"

    def decide(self, rel: dict, ambiguous: bool, code_like: bool,
               name_match: bool) -> Tuple[str, str]:
        """The single acceptance decision for an inferred candidate.

        Returns (bucket, state) where bucket is one of
        "inferred" | "ambiguous" | "rejected".

        Rules (all living here, PART 13):
          * an ambiguous candidate is never emitted (RULE 4) -> ambiguous bucket
          * a code-like column with no name corroboration is vetoed: its values
            overlap keys by coincidence, not by reference (PART 23, adversarial
            "shared codes") -> rejected bucket (recorded, never emitted)
          * otherwise the structured state comes from the score; CONFIRMED
            additionally requires corroboration across >= 2 independent
            evidence groups (PART 6) - a single strong signal must never
            produce a confident inferred FK.
        """
        if ambiguous:
            rel["relationship_state"] = "UNCERTAIN"
            rel["review_status"] = "review"
            rel["confidence_band"] = self.band_for(rel.get("confidence_score") or 0)
            rel["confidence_level"] = self.confidence_level_for(
                rel.get("confidence_score") or 0)
            return "ambiguous", "UNCERTAIN"

        if code_like and not name_match:
            rel["relationship_state"] = "REJECTED"
            rel["review_status"] = "rejected"
            rel["confidence_band"] = self.band_for(rel.get("confidence_score") or 0)
            rel["confidence_level"] = self.confidence_level_for(
                rel.get("confidence_score") or 0)
            return "rejected", "REJECTED"

        score = rel.get("confidence_score") or 0
        state = self.state_for(score)

        # PART 6: corroboration across independent evidence groups. A score
        # that reaches CONFIRMED on one group alone (e.g. pure data overlap
        # with no structural/lexical corroboration) is downgraded so a single
        # signal can never auto-accept an inferred FK.
        n_groups = rel.get("independent_signal_count")
        if state == "CONFIRMED" and (n_groups is None or n_groups < 2):
            state = "PROBABLE"

        rel["relationship_state"] = state
        rel["review_status"] = self.review_status_for(score)
        rel["confidence_band"] = self.band_for(score)
        rel["confidence_level"] = self.confidence_level_for(score)
        return "inferred", state

    def corroborate(self, rel: dict, score_delta: float, evidence: str) -> dict:
        """Re-apply the policy when an independent evidence group corroborates
        an already-classified relationship (e.g. LLM confirmation, PART 16).

        The state is recomputed from the accumulated score so acceptance always
        comes from this policy, never from the evidence source directly.
        """
        score = max(0, min(100, round((rel.get("confidence_score") or 0) + score_delta)))
        rel["confidence_score"] = score
        rel["confidence_band"] = self.band_for(score)
        rel["review_status"] = self.review_status_for(score)
        rel["relationship_state"] = self.state_for(score)
        rel["confidence_level"] = self.confidence_level_for(score)
        if evidence not in (rel.setdefault("evidence", []) or []):
            rel["evidence"].append(evidence)
        return rel