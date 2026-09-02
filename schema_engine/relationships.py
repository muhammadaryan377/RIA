"""Relationship inference orchestration.

The pipeline that drives candidate generation -> evidence collection ->
classification -> acceptance policy -> registry. This module is the single
entry point used by the schema agent and the cross-schema pass; every inferred
relationship passes through exactly the same layers (RULE 6, PART 15).

The acceptance decision for EVERY candidate (data path, name-only path,
composite path) is made by the single AcceptancePolicy.decide(). Candidates
that fail are retained in the registry (PART 14) even though the public output
omits them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from schema_engine.candidates import CandidateGroup, CandidateGenerator
from schema_engine.classifier import classify
from schema_engine.evidence import EvidenceCollector
from schema_engine.lexical import is_reference_to_other_table, normalize_identifier
from schema_engine.policy import AcceptancePolicy
from schema_engine.profiling import ValueProfiler
from schema_engine.registry import RelationshipRegistry

logger = logging.getLogger("aria.schema_engine.relationships")


def infer_relationships(tables, primary_keys, declared_fks, conn=None,
                        schema="public", db_type="postgresql", null_stats=None
                        ) -> Tuple[List[dict], List[dict], RelationshipRegistry]:
    """Infer relationships for columns with no declared FK constraint.

    Returns (inferred, ambiguous, registry). `inferred` and `ambiguous` are
    legacy-shaped relationship dicts; `registry` retains every candidate
    including REJECTED ones (PART 14) and the truthful profiling budget
    (PART 8).
    """
    null_stats = null_stats or {}
    table_pk: Dict[str, list] = {t: cols for t, cols in primary_keys.items()}

    profiler = ValueProfiler(conn, schema, db_type, tables, primary_keys) if conn is not None else None
    registry = RelationshipRegistry()
    registry.profile_budget = profiler.budget if profiler is not None else None
    registry.add_declared(declared_fks)

    generator = CandidateGenerator(tables, primary_keys, declared_fks,
                                   schema=schema, db_type=db_type, profiler=profiler)
    collector = EvidenceCollector(profiler, table_pk)
    policy = AcceptancePolicy()

    for cgroup in generator.generate():
        hit = collector.collect(cgroup)
        src_set = (profiler.source_values(cgroup.source_table, cgroup.source_column)
                   if profiler is not None else None)

        if hit is not None:
            target_table, target_col, ratio, matched, _amb = hit.as_tuple()
            if target_table in cgroup.name_targets:
                confidence = "data-confirmed"
                note = (f"No FK constraint declared; confirmed by value overlap "
                        f"({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
            else:
                confidence = "data-inferred"
                note = (f"No FK constraint declared and no name match; inferred from "
                        f"value overlap ({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
            tgt_pk = table_pk.get(target_table) or []
            ref_cols = [target_col] if len(tgt_pk) > 1 else (tgt_pk or [target_col])
            for ref_col in ref_cols:
                if cgroup.source_table == target_table and cgroup.source_column == ref_col:
                    continue  # a column can never reference itself
                rel, _amb = classify(profiler, table_pk, null_stats, schema,
                                     cgroup.source_table, cgroup.source_column,
                                     target_table, ref_col, confidence, note,
                                     overlap_ratio=ratio, name_hint=hit.name_hint,
                                     ambiguous=hit.ambiguous)
                bucket, _state = policy.decide(rel, hit.ambiguous,
                                               code_like=rel["evidence_detail"].get("code_like", False),
                                               name_match=rel["evidence_detail"].get("name_match", False))
                _route(registry, cgroup, bucket, rel, [confidence], hit.signals)
            continue  # data path decided this column (parity with legacy)

        # ---- name-only path (no data hit / offline) ----
        if cgroup.is_composite_key_col:
            registry.add_rejected(cgroup, ["composite_key_requires_data_evidence"])
            continue
        if not is_reference_to_other_table(cgroup.source_column, cgroup.source_table, tables):
            registry.add_rejected(cgroup, ["name_points_at_own_table"])
            continue
        emitted = False
        for candidate_table in sorted(cgroup.name_targets):
            cand_pk = table_pk.get(candidate_table) or []
            ref_cols = [cgroup.source_column] if len(cand_pk) > 1 else (cand_pk or [cgroup.source_column])
            for ref_col in ref_cols:
                rel, _amb = classify(
                    profiler, table_pk, null_stats, schema,
                    cgroup.source_table, cgroup.source_column,
                    candidate_table, ref_col,
                    "heuristic-name-match",
                    "No FK constraint declared in the database; relationship inferred from column naming convention.",
                    name_hint=True,
                )
                bucket, _state = policy.decide(rel, False,
                                               code_like=rel["evidence_detail"].get("code_like", False),
                                               name_match=True)
                _route(registry, cgroup, bucket, rel, ["heuristic_name_match"])
                emitted = True
        if not emitted:
            registry.add_rejected(cgroup, ["no_evidence"])

    _composite_pass(profiler, tables, primary_keys, table_pk, declared_fks,
                    schema, null_stats, registry, policy)

    return registry.inferred, registry.ambiguous, registry


def _route(registry: RelationshipRegistry, cgroup: CandidateGroup, bucket: str,
           rel: dict, reasons: List[str], signals: Optional[dict] = None):
    """Route a classified candidate to its registry bucket per the policy."""
    if bucket == "rejected":
        registry.add_rejected(cgroup, reasons + ["rejected_by_policy"], signals)
    elif bucket == "ambiguous":
        registry.add_ambiguous(cgroup, rel, reasons, signals)
    else:
        registry.add_inferred(cgroup, rel, reasons, signals)


def _composite_pass(profiler, tables, primary_keys, table_pk, declared_fks,
                    schema, null_stats, registry, policy):
    """Composite relationships: (a, b) -> (x, y) where the target PK is composite.

    Detected by name first, then confirmed by composite-tuple containment.
    """
    if profiler is None:
        return
    declared_tgt = {
        (fk["table_name"], fk["references_table"], fk["references_column"])
        for fk in declared_fks
    }
    for target_table, target_pk in sorted(table_pk.items()):
        if len(target_pk) < 2:
            continue
        for src_table, columns in tables.items():
            if src_table == target_table:
                continue
            src_pk = primary_keys.get(src_table, [])
            cols = {c["column"] for c in columns}
            matched = [c for c in target_pk if c in cols and c not in src_pk]
            if len(matched) < 2:
                continue
            if all((src_table, target_table, c) in declared_tgt for c in matched):
                continue
            matched_norm = set(normalize_identifier(c) for c in matched)
            src_names = set(normalize_identifier(c) for c in cols)
            table_norm = normalize_identifier(target_table)
            name_ok = any(
                n == table_norm + "_" + mn or mn in (n, n + "_id")
                for n in src_names for mn in matched_norm
            )
            if not name_ok:
                continue
            if not profiler.types_compatible(src_table, matched[0], target_table, target_pk[0]):
                continue
            ratio = _composite_containment(profiler, src_table, matched, target_table, target_pk)
            if ratio is None or ratio < profiler.MIN_OVERLAP:
                continue
            cgroup = CandidateGroup(
                candidate_id=f"composite-{src_table}-{target_table}",
                source_table=src_table,
                source_column=matched,
                name_targets={target_table},
                is_ref_flavored=True,
                is_composite_key_col=False,
                generation_reasons=["composite_key_alignment"],
            )
            note = (f"Composite FK inferred: ({', '.join(matched)}) references "
                    f"{target_table}({', '.join(target_pk)}) with {ratio:.0%} of "
                    f"source keys present in the target composite key.")
            rel, _amb = classify(
                profiler, table_pk, null_stats, schema,
                src_table, matched, target_table, target_pk,
                "composite-data-confirmed", note,
                overlap_ratio=ratio, name_hint=True,
            )
            bucket, _state = policy.decide(rel, False,
                                           code_like=False, name_match=True)
            _route(registry, cgroup, bucket, rel, ["composite_data_confirmed"])


def _composite_containment(profiler, src_table, matched, target_table, target_pk):
    """Fraction of source (matched...) tuples present in the target composite key."""
    try:
        matched_cols = [profiler.quote(c) for c in matched]
        tgt_cols = [profiler.quote(c) for c in target_pk]
        join_sql = " AND ".join(
            f"s.{mc} = t.{tc}" for mc, tc in zip(matched_cols, tgt_cols)
        )
        if profiler.db_type == "mysql":
            q = (
                f"SELECT COUNT(DISTINCT CASE WHEN t.{tgt_cols[0]} IS NOT NULL THEN s.{matched_cols[0]} END) "
                f"/ NULLIF(COUNT(DISTINCT s.{matched_cols[0]}),0) AS ratio "
                f"FROM {profiler.quote(src_table)} s "
                f"LEFT JOIN {profiler.quote(target_table)} t ON {join_sql}"
            )
        else:
            q = (
                f"SELECT COUNT(DISTINCT CASE WHEN t.{tgt_cols[0]} IS NOT NULL THEN s.{matched_cols[0]} END)::float "
                f"/ NULLIF(COUNT(DISTINCT s.{matched_cols[0]}),0) AS ratio "
                f"FROM {profiler.quote(profiler.schema)}.{profiler.quote(src_table)} s "
                f"LEFT JOIN {profiler.quote(profiler.schema)}.{profiler.quote(target_table)} t ON {join_sql}"
            )
        with profiler._dict_cursor() as cur:
            cur.execute(q)
            row = cur.fetchone()
            return row.get("ratio") if row else None
    except Exception as exc:
        logger.debug("composite check failed %s -> %s: %s", src_table, target_table, exc)
        return None