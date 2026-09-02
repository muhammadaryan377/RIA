"""schema_engine — normalized relational-schema inference pipeline.

Layers (single acceptance authority, RULE 6):
    adapters       -> normalized model (Phase B)
    candidates     -> CandidateGenerator (never accepts)
    evidence       -> EvidenceCollector (never accepts)
    classifier     -> Classifier (scores evidence)
    policy         -> AcceptancePolicy (the one acceptance authority)
    registry       -> RelationshipRegistry (retains REJECTED candidates)
    serializer     -> legacy-compatible output
"""

from schema_engine.adapters import ADAPTERS, DatabaseAdapter
from schema_engine.extraction import extract_database
from schema_engine.relationships import infer_relationships

__all__ = [
    "ADAPTERS", "DatabaseAdapter", "extract_database", "infer_relationships",
]