"""Candidate generation (PART 4).

The CandidateGenerator's ONLY job is to produce plausible
source-column -> target-column candidates. It NEVER accepts or rejects a
relationship and contains no `if score > X: accept` logic. Every candidate
carries the reasons it was generated so the classifier can treat each reason
as evidence availability (never as evidence weight).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from schema_engine.lexical import (
    ID_TOKENS, is_ref_flavored, normalize_identifier, singularize,
    table_name_candidates, tokenize_words,
)
from schema_engine.profiling import ValueProfiler


@dataclass
class CandidateGroup:
    """One source column + the target tables its *name* hints at.

    `name_targets` is purely lexical candidate generation; an empty set means
    "search data for the best target", which the evidence layer may do.
    """

    candidate_id: str
    source_table: str
    source_column: str
    name_targets: Set[str] = field(default_factory=set)
    name_targets_from_name: Set[str] = field(default_factory=set)
    is_ref_flavored: bool = False
    is_composite_key_col: bool = False
    generation_reasons: List[str] = field(default_factory=list)


class CandidateGenerator:
    def __init__(self, tables, primary_keys, declared_fks, schema="public",
                 db_type="postgresql", profiler: Optional[ValueProfiler] = None):
        self.tables = tables
        self.primary_keys = primary_keys
        self.schema = schema
        self.db_type = db_type
        self.profiler = profiler
        self.declared_pairs = {
            (fk["table_name"], fk["column_name"]) for fk in declared_fks
        }
        self.pk_owner: Dict[str, List[str]] = {}
        self.table_pk: Dict[str, List[str]] = {}
        for table_name, pk_cols in primary_keys.items():
            self.table_pk[table_name] = pk_cols
            for pk_col in pk_cols:
                self.pk_owner.setdefault(pk_col, []).append(table_name)

    def generate(self) -> List[CandidateGroup]:
        candidates: List[CandidateGroup] = []
        counter = 0
        for table_name, columns in self.tables.items():
            for col in columns:
                col_name = col["column"]

                if (table_name, col_name) in self.declared_pairs:
                    continue

                own_pk = self.primary_keys.get(table_name, [])
                is_composite_key_col = False
                if col_name in own_pk:
                    if len(own_pk) == 1:
                        continue  # sole PK = the table's own identity, never a FK
                    is_composite_key_col = True
                    if self._is_line_counter(table_name, col_name):
                        # A composite-PK member whose name is ALSO a key of
                        # another table is a child-embedding-parent-key
                        # (structural FK lead, e.g. `stocks.store_id` where
                        # `stores` has few rows), NOT a 1..n line counter.
                        # Only genuine line counters (item_id / line_no) are
                        # suppressed (PART 4 broad generation).
                        if not any(t != table_name for t in self.pk_owner.get(col_name, [])):
                            continue

                ref_flavored = is_ref_flavored(col_name)
                reasons = []
                if ref_flavored:
                    reasons.append("ref_flavored_name")
                else:
                    # Only data can find non-ref-flavored candidates; the column
                    # must still be a plausible key-ish type to be worth sampling.
                    if not (self.profiler and self.profiler.should_profile(col)):
                        continue
                    reasons.append("key_typed_numeric_or_idlike")

                # Lexical target hints (never an acceptance decision).
                name_targets, name_targets_from_name = self._lexical_targets(table_name, col_name)
                if is_composite_key_col:
                    reasons.append("composite_key_member")
                if name_targets:
                    reasons.append("lexical_name_hint")
                else:
                    reasons.append("data_only_search")

                counter += 1
                candidates.append(CandidateGroup(
                    candidate_id=f"c{counter}",
                    source_table=table_name,
                    source_column=col_name,
                    name_targets=name_targets,
                    name_targets_from_name=name_targets_from_name,
                    is_ref_flavored=ref_flavored,
                    is_composite_key_col=is_composite_key_col,
                    generation_reasons=reasons,
                ))
        return candidates

    def _is_line_counter(self, table_name: str, col_name: str) -> bool:
        """Composite PK members that are contiguous 1..n counters are line/sort
        keys (e.g. item_id=1,2,3...), not FKs; they overlap every small PK."""
        if self.profiler is None:
            return False
        values = self.profiler.source_values(table_name, col_name)
        if values is None or len(values) > 6:
            return False
        nums = sorted(
            v[1] if isinstance(v, tuple) else v
            for v in values
            if (isinstance(v, tuple) and len(v) >= 2 and isinstance(v[1], (int, float)))
            or isinstance(v, (int, float))
        )
        return bool(nums) and nums[0] == 1 and nums == list(range(1, len(nums) + 1))

    def _lexical_targets(self, table_name: str, col_name: str) -> Tuple[Set[str], Set[str]]:
        """Tables the column's name plausibly references (pure lexical hints).

        Mirror of the legacy `candidate_tables` construction: shared PK-column
        name, table-name token match, or a single same-named PK column of
        another table. Never an acceptance decision.

        Returns (name_targets, name_targets_from_name). The second set is the
        subset of targets reached by a TABLE-NAME match: only those count as a
        genuine LEXICAL corroboration. Targets reached purely because the
        source column happens to share a name with a member of a composite PK
        are structural leads (e.g. `personid` inside `businessentitycontact`'s
        composite key) and must not masquerade as a name signal (PART 11/18).
        """
        from_name = set(table_name_candidates(col_name, self.tables))
        targets = set(self.pk_owner.get(col_name, []))
        if not from_name:
            same_name = [
                other_table
                for other_table in self.tables
                if other_table != table_name
                and col_name in self.table_pk.get(other_table, [])
                and len(normalize_identifier(col_name)) >= 3
            ]
            if len(same_name) == 1:
                # The column is a member of another table's (composite) key.
                # That is a structural lead, NOT a table-name match (PART 18):
                # it must never count as a genuine LEXICAL signal, so it is
                # added to the searchable targets only, never to `from_name`.
                targets.update(same_name)
        targets.update(from_name)
        targets.discard(table_name)
        from_name.discard(table_name)
        return targets, from_name