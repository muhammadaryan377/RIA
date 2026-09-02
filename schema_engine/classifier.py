"""Relationship classifier (PART 12).

Turns a candidate + its evidence into a structured classification. The
confidence score, structured confidence level, independent-evidence-group
tracking and provenance are computed HERE; the acceptance state is delegated
to the single AcceptancePolicy - classification never accepts or emits
anything itself (RULE 6).

The score/evidence computation keeps the legacy additive scoring (Phase C
benchmark parity); brittle inline heuristics are surfaced as general evidence
signals (code_like, evidence_groups) that the policy consumes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from schema_engine.lexical import is_code_column, singularize, tokenize_words
from schema_engine.policy import AcceptancePolicy
from schema_engine.profiling import ValueProfiler


def classify(profiler: Optional[ValueProfiler],
             table_pk: Dict[str, list],
             null_stats: Dict[str, dict],
             schema: str,
             table_name: str,
             col_name,
             candidate_table: str,
             ref_col,
             confidence: str,
             note: str,
             overlap_ratio: Optional[float] = None,
             name_hint: bool = False,
             ambiguous: bool = False,
             llm_signal: Optional[str] = None) -> Tuple[dict, bool]:
    """Build the relationship record (legacy-compatible) + its ambiguity flag.

    Mirrors the legacy `_emit` scoring, adds structured confidence
    (confidence_level), independent-evidence-group tracking (PART 6),
    evidence_provenance, and the code_like veto signal the policy consumes.
    """
    policy = AcceptancePolicy()

    rel = {
        "table": table_name,
        "column": col_name,
        "references_table": candidate_table,
        "references_column": ref_col,
        "confidence": confidence,
        "note": note,
    }
    if isinstance(table_name, str) and "." in table_name:
        rel["schema"] = table_name.split(".", 1)[0]
    if isinstance(candidate_table, str) and "." in candidate_table:
        rel["references_schema"] = candidate_table.split(".", 1)[0]
    if ambiguous:
        rel["ambiguous"] = True

    # Cardinality: source distinct values (data available) vs referenced key.
    if profiler is not None and isinstance(col_name, str) and isinstance(ref_col, str):
        src_vals = profiler.source_values(table_name, col_name)
        tgt_vals = profiler._pk_values(candidate_table)
        if src_vals is not None and len(tgt_vals) > 0:
            ratio = len(src_vals) / len(tgt_vals)
            rel["cardinality"] = (
                "one-to-one" if ratio >= 0.9
                else "one-to-many" if ratio > 1.1
                else "many-to-one"
            )

    evidence: List[str] = []
    score = 0.0
    detail = {
        "name_match": False,
        "datatype_match": None,
        "target_unique": False,
        "value_overlap": overlap_ratio,
        "null_rate": None,
        "cardinality": rel.get("cardinality"),
        "same_schema": None,
    }

    def _table_schema(name: str) -> str:
        if isinstance(name, str) and "." in name:
            return name.split(".", 1)[0]
        return schema

    src_schema = _table_schema(table_name)
    tgt_schema = _table_schema(candidate_table)
    same_schema = bool(src_schema and tgt_schema and src_schema == tgt_schema)
    detail["same_schema"] = same_schema
    if same_schema:
        score += 5
        evidence.append("intra-schema relationship (supporting)")
    else:
        evidence.append("cross-schema relationship (not penalized)")

    name_hit = name_hint
    if not name_hit and isinstance(col_name, str):
        base = _singularize_base(col_name)
        # Match against the table name only, never the schema prefix (PART 18).
        tgt_part = (candidate_table or "").rsplit(".", 1)[-1]
        tgt_tokens = {singularize(tok) for tok in tokenize_words(tgt_part)}
        if base in tgt_tokens:
            name_hit = True
    detail["name_match"] = name_hit
    if name_hit:
        score += 25
        evidence.append("column name matches target table")

    if profiler is not None and isinstance(col_name, str) and isinstance(ref_col, str):
        dt_ok = profiler.types_compatible(table_name, col_name, candidate_table, ref_col)
        detail["datatype_match"] = bool(dt_ok)
        if dt_ok:
            score += 15
            evidence.append("datatype compatible")
        else:
            evidence.append("datatype mismatch")

    if overlap_ratio is not None:
        detail["value_overlap"] = overlap_ratio
        if overlap_ratio >= 0.99:
            score += 30
            evidence.append("~100% values found in target PK")
        elif overlap_ratio >= 0.95:
            score += 25
            evidence.append(">=95% values found in target PK")
        elif overlap_ratio >= 0.80:
            score += 18
            evidence.append(">=80% values found in target PK")
        else:
            evidence.append("partial value overlap")

    tgt_pk = set(table_pk.get(candidate_table, []))
    target_unique = False
    if isinstance(ref_col, str) and ref_col in tgt_pk:
        target_unique = True
        score += 20
        evidence.append("target is a primary key")
    elif not isinstance(ref_col, str):
        if all(c in tgt_pk for c in ref_col):
            target_unique = True
            score += 20
            evidence.append("target is a composite primary key")
    detail["target_unique"] = target_unique

    if rel.get("cardinality") in ("many-to-one", "one-to-many"):
        score += 10
        evidence.append("many source rows map to one target row")

    if profiler is not None and isinstance(col_name, str):
        ns = null_stats.get(table_name, {}).get(col_name, {})
        null_pct = ns.get("null_pct") if isinstance(ns, dict) else None
        if null_pct is not None and null_pct > 1:
            null_pct = null_pct / 100.0
        if null_pct is not None:
            detail["null_rate"] = null_pct
            if null_pct > 0.8:
                score -= 25
                evidence.append(f"high null rate ({null_pct:.0%}) - relationship optional")
            elif null_pct > 0.5:
                score -= 10
                evidence.append(f"nullable ({null_pct:.0%})")

    if llm_signal is not None:
        if llm_signal == "confirm":
            score += 20
            evidence.append("LLM reasoning corroborates this target")
        elif llm_signal == "add":
            score += 45
            evidence.append("LLM reasoning proposes this relationship (no data corroboration)")

    # ---- code_like: general semantic veto signal for the policy (PART 17) ----
    code_like = bool(isinstance(col_name, str) and is_code_column(col_name))
    detail["code_like"] = code_like

    # ---- independent evidence groups (PART 6) ---------------------------- #
    # GROUP rules: STRUCTURAL from declared metadata (target uniqueness,
    # datatype compatibility); LEXICAL from name matching; STATISTICAL from
    # measured value overlap / cardinality; SEMANTIC from LLM or semantic
    # similarity signals. DECLARED_METADATA is reserved for declared FKs.
    groups: Dict[str, bool] = {
        "STRUCTURAL": bool(target_unique) or detail["datatype_match"] is True,
        "LEXICAL": bool(name_hit),
        "STATISTICAL": overlap_ratio is not None,
        "SEMANTIC": llm_signal is not None,
        "DECLARED_METADATA": False,
    }
    independent_signal_count = sum(
        1 for g, v in groups.items() if v and g != "DECLARED_METADATA")
    provenance = [g for g, v in groups.items() if v and g != "DECLARED_METADATA"]

    rel["confidence_score"] = max(0, min(100, round(score)))
    rel["evidence"] = evidence
    rel["evidence_detail"] = detail
    rel["evidence_groups"] = groups
    rel["independent_signal_count"] = independent_signal_count
    rel["evidence_provenance"] = provenance
    rel["confidence_level"] = policy.confidence_level_for(rel["confidence_score"])
    rel["review_status"] = policy.review_status_for(rel["confidence_score"])
    rel["relationship_state"] = policy.state_for(rel["confidence_score"])
    rel["confidence_band"] = policy.band_for(rel["confidence_score"])

    rel["self_referencing"] = bool(
        isinstance(col_name, str) and table_name == candidate_table and col_name != ref_col
    )

    if rel.get("self_referencing"):
        rel["relationship_type"] = "self-referencing"
    elif rel.get("cardinality"):
        rel["relationship_type"] = rel["cardinality"]
    else:
        rel["relationship_type"] = "foreign-key"

    if ambiguous:
        rel["review_status"] = "review"
        rel["relationship_state"] = "UNCERTAIN"
        if "ambiguous candidate - data cannot distinguish best target" not in evidence:
            evidence.append("ambiguous candidate - data cannot distinguish best target")
            rel["evidence"] = evidence

    return rel, ambiguous


def _singularize_base(col_name) -> str:
    if isinstance(col_name, str):
        if col_name.endswith("_id"):
            return singularize(col_name[:-3])
        if col_name.endswith("id") and len(col_name) > 3:
            return singularize(col_name[:-2])
        return singularize(col_name)
    return ""