"""Evidence collection (PART 5).

The EvidenceCollector turns a generated candidate into structured evidence. It
NEVER accepts or rejects a relationship: it returns the best target found by
the data (or none) plus an explicit signal record where missing data is
`UNKNOWN`, never `FALSE` (PART 7/8).

The code-like veto is NOT applied here (RULE 6): a code-like column is marked
as a semantic evidence signal and the single AcceptancePolicy decides. This
keeps evidence collection a pure measurement layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from schema_engine.candidates import CandidateGroup
from schema_engine.lexical import is_code_column
from schema_engine.profiling import ValueProfiler


@dataclass
class Hit:
    """Best data-supported target for a candidate, with an evidence record."""

    target_table: str
    target_col: str
    ratio: float
    matched: int
    ambiguous: bool
    name_hint: bool
    containment_status: str  # "EXACT" | "SAMPLED" | "UNKNOWN"
    signals: Dict[str, dict] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, str, float, int, bool]:
        return (self.target_table, self.target_col, self.ratio, self.matched, self.ambiguous)


class EvidenceCollector:
    def __init__(self, profiler: Optional[ValueProfiler], table_pk: Dict[str, List[str]]):
        self.profiler = profiler
        self.table_pk = table_pk

    def collect(self, cgroup: CandidateGroup) -> Optional[Hit]:
        if self.profiler is None:
            return None
        src_set = self.profiler.source_values(cgroup.source_table, cgroup.source_column)
        rows = self.profiler.row_count(cgroup.source_table)
        exclude = (cgroup.source_table, cgroup.source_column)
        all_tables = set(t for t, _, _ in self.profiler.pk_index)

        hit = None
        containment_status = "UNKNOWN"
        # A shared-key column (its name is a key in many tables, e.g.
        # `businessentityid`, `customerid`) cannot be narrowed to one name
        # target: the data matches every holder of that key and a name match to
        # one holder is a key-concept match, not a table match (PART 11).
        # Those candidates are searched against the full pool and subject to
        # shared-key ambiguity detection.
        shared_key = False
        if self.profiler is not None:
            key_owners = [
                t for t, cols in self.table_pk.items() if cgroup.source_column in cols
            ]
            shared_key = len(key_owners) >= 2
        name_targets = cgroup.name_targets_from_name if not shared_key else set()
        if cgroup.is_ref_flavored or cgroup.is_composite_key_col:
            hit = self.profiler.strongest_overlap(
                src_set,
                only_tables=name_targets or None,
                min_source=2,
                exclude=exclude,
                require_unambiguous=not name_targets,
                source_table=cgroup.source_table,
                source_col=cgroup.source_column,
            )
            containment_status = "SAMPLED"
        elif not cgroup.name_targets:
            hit = self.profiler.strongest_overlap(
                src_set,
                high_cardinality=True,
                row_count=rows,
                exclude=exclude,
                source_table=cgroup.source_table,
                source_col=cgroup.source_column,
            )
            containment_status = "SAMPLED"

        if hit is None and name_targets and (cgroup.is_ref_flavored or cgroup.is_composite_key_col):
            hit = self.profiler.strongest_exact_overlap(
                cgroup.source_table, cgroup.source_column, name_targets,
                exclude=exclude,
            )
            containment_status = "EXACT"
        if hit is None and not name_targets and cgroup.is_ref_flavored and not shared_key:
            hit = self.profiler.strongest_exact_overlap(
                cgroup.source_table, cgroup.source_column, all_tables,
                exclude=exclude, require_unique_winner=True,
            )
            containment_status = "EXACT"

        if hit is None:
            return None

        target_table, target_col, ratio, matched, ambiguous = hit
        name_hint = bool(cgroup.name_targets_from_name and target_table in cgroup.name_targets_from_name)

        # PART 11: a data-only target is a key-concept match, not a table match,
        # when either (a) the target key column is a shared domain used by many
        # tables (`businessentityid`, `customer_id`, ...), or (b) the target
        # column is a member of a COMPOSITE key (the table is a junction/detail,
        # not a parent; matching its partial key is coincidence). In both cases
        # the value overlap cannot name the actual parent -> ambiguous.
        # The veto does NOT apply when:
        #   * the source column's name corroborates the specific target table
        #     (a genuine LEXICAL signal resolves the holder), or
        #   * the source column is itself a member of a composite key - the
        #     child embeds the parent's key by design (`productvendor.businessentityid`
        #     -> `vendor.businessentityid`, `purchaseorderdetail.purchaseorderid`
        #     -> `purchaseorderheader`), so the match is a real child->parent
        #     reference, not a coincidence.
        #   * the target is the source's OWN table (a self-reference): the
        #     shared-key veto is a cross-table concern (a key name reused as a
        #     parent identifier in several tables); a column referencing its
        #     own table's PK is a self-hierarchy, not a shared-domain coincidence,
        #     so other tables that happen to carry the same key as a composite
        #     member (e.g. `employee_territories.employee_id`) must not veto it.
        is_self_reference = (
            isinstance(cgroup.source_column, str)
            and target_table == cgroup.source_table
        )
        if (not name_hint and not cgroup.is_composite_key_col
                and isinstance(target_col, str) and not is_self_reference):
            tgt_pk = self.table_pk.get(target_table, [])
            target_is_composite_member = target_col in tgt_pk and len(tgt_pk) > 1
            tgt_key_owners = [
                t for t, cols in self.table_pk.items() if target_col in cols
            ]
            if target_is_composite_member or len(tgt_key_owners) >= 2:
                ambiguous = True

        signals = self._signals(cgroup, target_table, target_col, ratio,
                                containment_status)
        return Hit(
            target_table=target_table,
            target_col=target_col,
            ratio=ratio,
            matched=matched,
            ambiguous=ambiguous,
            name_hint=name_hint,
            containment_status=containment_status,
            signals=signals,
        )

    def _signals(self, cgroup: CandidateGroup, target_table: str, target_col: str,
                 ratio: float, containment_status: str) -> Dict[str, dict]:
        p = self.profiler
        signals: Dict[str, dict] = {}

        # value_containment: UNKNOWN when the profile budget ran out, else the
        # measured containment status. Sampling vs exact is explicit (PART 7);
        # a sampled check never masquerades as exact evidence.
        if p is None:
            signals["value_containment"] = {"status": "UNKNOWN", "reason": "no_connection"}
        else:
            signals["value_containment"] = {
                "status": "UNKNOWN" if p.budget.exhausted else
                          ("TRUE" if ratio >= p.MIN_OVERLAP else "FALSE"),
                "value": ratio,
                "kind": containment_status if containment_status in ("EXACT", "SAMPLED")
                        else "SAMPLED",
            }
        signals["value_overlap"] = {
            "status": "UNKNOWN" if p is None or p.budget.exhausted else "TRUE",
            "value": ratio,
        }

        # datatype compatibility (UNKNOWN when type metadata is absent).
        if p is None:
            signals["type_compatibility"] = {"status": "UNKNOWN", "reason": "no_connection"}
        else:
            ok = p.types_compatible(cgroup.source_table, cgroup.source_column, target_table, target_col)
            signals["type_compatibility"] = {"status": "TRUE" if ok else "FALSE"}

        # target uniqueness (declared metadata; UNKNOWN when not a PK/UNIQUE).
        tgt_pk = self.table_pk.get(target_table, [])
        signals["target_unique"] = {
            "status": "TRUE" if target_col in tgt_pk else "UNKNOWN",
        }

        # name similarity (lexical hint or data-driven target). Only a table-name
        # match is a genuine LEXICAL signal; a composite-PK-column lead is not
        # (PART 11/18).
        signals["name_similarity"] = {
            "status": "TRUE" if (cgroup.name_targets_from_name and target_table in cgroup.name_targets_from_name) else "FALSE",
        }

        # semantic signal: code-like column (the POLICY vetoes these unless a
        # name corroborates - here it is pure evidence, PART 17).
        signals["code_like"] = {
            "status": "TRUE" if is_code_column(cgroup.source_column) else "FALSE",
        }

        # table semantic similarity: no descriptions are available in the
        # deterministic path -> UNKNOWN (never FALSE).
        signals["table_semantic_similarity"] = {
            "status": "UNKNOWN", "reason": "no_semantic_metadata",
        }

        # source/target cardinality evidence (when measurable).
        if p is not None and isinstance(cgroup.source_column, str) and isinstance(target_col, str):
            sv = p.source_values(cgroup.source_table, cgroup.source_column)
            tv = p.source_values(target_table, target_col)
            if tv is not None and len(tv) > 0:
                ratio_c = len(sv) / len(tv)
                signals["cardinality_compatibility"] = {
                    "status": "TRUE",
                    "value": ("one-to-one" if ratio_c >= 0.9
                              else "one-to-many" if ratio_c > 1.1
                              else "many-to-one"),
                }
            else:
                signals["cardinality_compatibility"] = {"status": "UNKNOWN", "reason": "no_target_data"}
        else:
            signals["cardinality_compatibility"] = {"status": "UNKNOWN"}

        return signals