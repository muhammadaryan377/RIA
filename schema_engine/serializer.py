"""Output serialization (PART 19).

Converts the registry into the legacy `schema_mapping.json` shapes (so
downstream ARIA agents never break) plus structured relationship metadata
(state, confidence, evidence, reasons, provenance) as backward-compatible
extra fields. REJECTED candidates are retained for the internal report and
omitted from the public relationship lists (PART 14).
"""

from __future__ import annotations

from typing import List

from schema_engine.registry import RelationshipRegistry


def serialize_inferred(registry: RelationshipRegistry) -> List[dict]:
    return list(registry.inferred)


def serialize_ambiguous(registry: RelationshipRegistry) -> List[dict]:
    return list(registry.ambiguous)


def serialize_rejected(registry: RelationshipRegistry) -> List[dict]:
    """PART 14 rejection records: candidate, state, confidence, evidence, reasons."""
    out = []
    for o in registry.rejected:
        out.append({
            "candidate_id": o.candidate_id,
            "source": f"{o.source_table}.{o.source_column}",
            "state": o.state,
            "confidence": None,
            "evidence": o.evidence_signals,
            "reasons": o.reasons,
        })
    return out


def serialize_registry(registry: RelationshipRegistry) -> dict:
    """Rich, backward-compatible relationship report."""
    return {
        "declared_relationships": list(registry.declared),
        "inferred_relationships": serialize_inferred(registry),
        "ambiguous_relationships": serialize_ambiguous(registry),
        "rejected_candidates": serialize_rejected(registry),
        "summary": registry.summary(),
    }