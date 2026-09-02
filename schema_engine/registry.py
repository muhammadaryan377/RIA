"""Relationship registry (PART 14).

Holds every candidate and its outcome — including REJECTED candidates that
the public output omits (backward compatibility) but the internal result
retains for debugging and benchmarking.

Buckets mirror the legacy contract exactly: `inferred` holds every non-
ambiguous classification (including low-score UNCERTAIN records), `ambiguous`
holds candidates the evidence could not distinguish, and `rejected` is never
emitted publicly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from schema_engine.candidates import CandidateGroup


@dataclass
class Outcome:
    candidate_id: str
    source_table: str
    source_column: str
    state: str  # DECLARED | CONFIRMED | PROBABLE | UNCERTAIN | REJECTED
    rel: Optional[dict] = None          # legacy-shaped relationship record
    reasons: List[str] = field(default_factory=list)
    evidence_signals: dict = field(default_factory=dict)


class RelationshipRegistry:
    def __init__(self):
        self.declared: List[dict] = []
        self._inferred: List[Outcome] = []
        self._ambiguous: List[Outcome] = []
        self._rejected: List[Outcome] = []
        self.profile_budget = None  # ProfileBudget instance (truthful reporting, PART 8)

    def add_declared(self, rels: List[dict]):
        self.declared = list(rels)

    def add_inferred(self, candidate: CandidateGroup, rel: dict, reasons: List[str],
                     evidence_signals: dict = None):
        self._inferred.append(Outcome(
            candidate_id=candidate.candidate_id,
            source_table=candidate.source_table,
            source_column=candidate.source_column,
            state=rel.get("relationship_state", "CONFIRMED"),
            rel=rel,
            reasons=reasons,
            evidence_signals=evidence_signals or {},
        ))

    def add_ambiguous(self, candidate: CandidateGroup, rel: dict, reasons: List[str],
                      evidence_signals: dict = None):
        rel = dict(rel)
        rel.setdefault("relationship_state", "UNCERTAIN")
        self._ambiguous.append(Outcome(
            candidate_id=candidate.candidate_id,
            source_table=candidate.source_table,
            source_column=candidate.source_column,
            state="UNCERTAIN",
            rel=rel,
            reasons=reasons,
            evidence_signals=evidence_signals or {},
        ))

    def add_rejected(self, candidate: CandidateGroup, reasons: List[str],
                     evidence_signals: dict = None):
        """A generated candidate that failed evidence/classification. Retained
        internally (PART 14); never emitted to the public output."""
        self._rejected.append(Outcome(
            candidate_id=candidate.candidate_id,
            source_table=candidate.source_table,
            source_column=candidate.source_column,
            state="REJECTED",
            reasons=reasons,
            evidence_signals=evidence_signals or {},
        ))

    # -- views -------------------------------------------------------- #

    @property
    def inferred(self) -> List[dict]:
        return [o.rel for o in self._inferred if o.rel]

    @property
    def ambiguous(self) -> List[dict]:
        return [o.rel for o in self._ambiguous if o.rel]

    @property
    def rejected(self) -> List[Outcome]:
        return list(self._rejected)

    @property
    def all_candidates(self) -> List[Outcome]:
        return list(self._inferred) + list(self._ambiguous) + list(self._rejected)

    def summary(self) -> dict:
        out = {
            "declared": len(self.declared),
            "inferred": len(self.inferred),
            "uncertain": len(self.ambiguous),
            "rejected": len(self.rejected),
            "total_candidates": len(self.all_candidates),
        }
        if self.profile_budget is not None:
            out["profile_budget_exhausted"] = bool(self.profile_budget.exhausted)
            out["queries_used"] = self.profile_budget.used
            out["queries_remaining"] = self.profile_budget.remaining
            out["profiling_status"] = (
                "exhausted" if self.profile_budget.exhausted else "active"
            )
        else:
            out["profile_budget_exhausted"] = None
            out["queries_used"] = None
            out["queries_remaining"] = None
            out["profiling_status"] = "unknown"
        return out