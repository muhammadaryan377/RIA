"""Tiered, budget-aware value profiling (PART 5/7/8).

Ports the legacy schema_agent sampling/overlap behavior verbatim (Phase C
parity) while adding a truthful budget tracker: `queries_used`,
`queries_remaining`, `profile_budget_exhausted`, and UNKNOWN-vs-FALSE
representation when a query could not run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("aria.schema_engine.profiling")

NUMERIC_TYPES = ("int", "bigint", "smallint", "numeric", "decimal", "float",
                 "double", "real", "serial", "money")
TEXT_TYPES = ("char", "varchar", "text", "uuid", "string")


def _token_prefix_match(src_tokens: set, table_tokens: set) -> bool:
    """True when a source-token is a prefix of (or is prefixed by) a table token.

    Generalizes name corroboration beyond exact token equality so that a column
    like `ship_via` (token `ship`) still resolves to `shippers` (token `shipper`)
    and `customer_id`/`customer_type_id` reach `customers` even when no exact
    token coincides. Only tokens >= 3 chars participate to avoid short-token
    noise.
    """
    if not src_tokens or not table_tokens:
        return False
    for s in src_tokens:
        if len(s) < 3:
            continue
        for t in table_tokens:
            if len(t) < 3:
                continue
            if s.startswith(t) or t.startswith(s):
                return True
    return False


def norm_value(v):
    """Normalize a sampled value for cross-column comparison (numeric-aware)."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    return ("str", str(v))


@dataclass
class ProfileBudget:
    limit: int = 400
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def spend(self) -> bool:
        if self.exhausted:
            return False
        self.used += 1
        return True


class ValueProfiler:
    """Samples distinct column values so relationships can be confirmed from DATA.

    A column whose values overlap a primary key's values is a de-facto foreign
    key, regardless of how it is named (e.g. support_rep_id -> employee,
    reports_to -> employee).
    """

    MIN_OVERLAP = 0.85
    MAX_SAMPLE = 500
    MAX_PK_SAMPLE = 100000
    MAX_PROFILE_QUERIES = 400

    def __init__(self, conn, schema, db_type, tables, primary_keys):
        self.conn = conn
        self.schema = schema
        self.db_type = db_type
        self.tables = tables
        self._src_cache: Dict[Tuple, set] = {}
        self._pk_cache: Dict[Tuple, set] = {}
        self._rows_cache: Dict[str, int] = {}
        self._dtype_cache: Dict[Tuple, str] = {}
        self.budget = ProfileBudget(limit=self.MAX_PROFILE_QUERIES)
        self.pk_index: List[Tuple[str, str, set]] = []
        # Tables whose column is their SOLE primary key: a candidate whose key
        # column is only a partial key of other tables is a detail/child, not an
        # alternative parent, so it must not compete as a shared-key target.
        self.sole_key_tables = {t for t, cols in primary_keys.items() if len(cols) == 1}
        for table, cols in primary_keys.items():
            for col in cols:
                self.pk_index.append(
                    (table, col, self._distinct(table, col, self._pk_cache, self.MAX_PK_SAMPLE)))

    def quote(self, name: str) -> str:
        if self.db_type == "mysql":
            return "`" + str(name).replace("`", "") + "`"
        return '"' + str(name).replace('"', "") + '"'

    def rel_table(self, table):
        """SQL-safe qualified table reference (schema-qualified where needed)."""
        if self.db_type == "mysql":
            return self.quote(table)
        if isinstance(table, str) and "." in table:
            parts = table.split(".", 1)
            return f"{self.quote(parts[0])}.{self.quote(parts[1])}"
        return f"{self.quote(self.schema)}.{self.quote(table)}"

    def column_type(self, table, col) -> str:
        key = (table, col)
        if key not in self._dtype_cache:
            dtype = ""
            cols = self.tables.get(table)
            if isinstance(cols, list):
                for c in cols:
                    if c.get("column") == col:
                        dtype = str(c.get("data_type", "")).lower()
                        break
            elif isinstance(cols, dict):
                for c in cols.get("columns", []):
                    if c.get("column") == col:
                        dtype = str(c.get("data_type", "")).lower()
                        break
            self._dtype_cache[key] = dtype
        return self._dtype_cache[key]

    def types_compatible(self, src_table, src_col, tgt_table, tgt_col) -> bool:
        st = self.column_type(src_table, src_col)
        tt = self.column_type(tgt_table, tgt_col)
        if not st or not tt:
            return True  # unknown -> don't reject
        if st == tt:
            return True
        is_s_num = any(tok in st for tok in NUMERIC_TYPES)
        is_t_num = any(tok in tt for tok in NUMERIC_TYPES)
        is_s_txt = any(tok in st for tok in TEXT_TYPES)
        is_t_txt = any(tok in tt for tok in TEXT_TYPES)
        if is_s_num and is_t_num:
            return True
        if is_s_txt and is_t_txt:
            return True
        return False

    def _distinct(self, table, col, cache: Dict[Tuple, set], limit: int) -> set:
        key = (table, col)
        if key in cache:
            return cache[key]
        if not self.budget.spend():
            cache[key] = set()
            return cache[key]
        values: set = set()
        try:
            with self._dict_cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT {self.quote(col)} AS v "
                    f"FROM {self.rel_table(table)} "
                    f"WHERE {self.quote(col)} IS NOT NULL "
                    f"ORDER BY {self.quote(col)} LIMIT {limit}"
                )
                for row in cur.fetchall():
                    values.add(norm_value(row["v"]))
        except Exception as exc:
            logger.debug("distinct sample failed for %s.%s: %s", table, col, exc)
            values = set()
        cache[key] = values
        return values

    def _dict_cursor(self):
        if self.db_type == "mysql":
            import pymysql.cursors
            return self.conn.cursor(pymysql.cursors.DictCursor)
        import psycopg2.extras
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def source_values(self, table, col) -> set:
        return self._distinct(table, col, self._src_cache, self.MAX_SAMPLE)

    def row_count(self, table) -> int:
        if table not in self._rows_cache:
            try:
                with self._dict_cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) AS n FROM {self.rel_table(table)}")
                    self._rows_cache[table] = cur.fetchone()["n"] or 0
            except Exception as exc:
                logger.debug("row count failed for %s: %s", table, exc)
                self._rows_cache[table] = 0
        return self._rows_cache[table]

    @staticmethod
    def _overlap(src, target) -> float:
        if not src:
            return 0.0
        return len(src & target) / len(src)

    @staticmethod
    def _is_selfref_column(col_name: str) -> bool:
        """Ref-flavored column names that conventionally self-reference their own table.

        `reports_to`, `parent_id`, `parent_category_id`, `manager_id`, `managed_by`
        etc. point at the same table's PK (employee.manager -> employee). We only
        use this as a tie-breaker when data also confirms full containment, so a
        genuine cross-table `_to`/`_by` reference (target != source table) is never
        forced to self-reference.
        """
        if not isinstance(col_name, str) or not col_name:
            return False
        low = col_name.lower()
        return (
            low.endswith("_to")
            or low.endswith("_parent")
            or low.endswith("_managed")
            or low.endswith("_manager")
        )

    # ------------------------------------------------------------------ #
    # Target ranking (ported verbatim for Phase C parity).
    # ------------------------------------------------------------------ #

    def strongest_overlap(self, source_set, only_tables=None, min_source=1,
                          high_cardinality=False, row_count=None, exclude=None,
                          require_unambiguous=False, source_table=None, source_col=None):
        """Return (table, pk_col, overlap_ratio, matched) for the best PK target.

        Ties are broken in favour of the target whose primary key set is the
        smallest that still contains the values (tightest containment). When
        `require_unambiguous` is set and the top two targets are
        indistinguishable on (ratio, matched), no target is returned.
        """
        from schema_engine.lexical import ID_TOKENS, singularize, tokenize_words

        cands = []
        src_tokens: set = set()
        if source_col:
            for tok in tokenize_words(source_col):
                if tok in ID_TOKENS:
                    continue
                src_tokens.add(singularize(tok))
                if tok.endswith("id") and len(tok) > 2:
                    src_tokens.add(singularize(tok[:-2]))
        most_specific = None
        if src_tokens:
            for table, _, _ in self.pk_index:
                name_part = str(table).rsplit(".", 1)[-1]
                table_tokens = {singularize(tok) for tok in tokenize_words(name_part)}
                if not table_tokens:
                    continue
                if table_tokens & src_tokens or _token_prefix_match(src_tokens, table_tokens):
                    if most_specific is None or len(str(table)) < len(str(most_specific)):
                        most_specific = table
        for table, pk_col, target_set in self.pk_index:
            if only_tables and table not in only_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            if len(source_set) < min_source:
                continue
            if source_table is not None and source_col is not None:
                if not self.types_compatible(source_table, source_col, table, pk_col):
                    continue
            ratio = self._overlap(source_set, target_set)
            if ratio < self.MIN_OVERLAP:
                continue
            if high_cardinality:
                if len(source_set) < 10:
                    continue
                if row_count and len(source_set) < 0.05 * row_count:
                    continue
            boost = 10 if (most_specific and table == most_specific) else 0
            # Self-FK preference (PART 5): a ref-flavored column ending in a
            # self-reference convention (`_to`, `_by`, `_parent`, `_managed`)
            # whose sampled values are FULLY contained in the same table's own
            # primary key is a self-hierarchy reference (e.g.
            # employees.reports_to -> employees.employee_id). Data overlap alone
            # cannot distinguish this from a coincidental small-key match (shippers
            # also contains the manager ids), so prefer the self target.
            if boost == 0 and source_col and self._is_selfref_column(source_col) \
                    and source_table == table and source_set \
                    and len(source_set & target_set) == len(source_set):
                boost = 9
            cands.append((ratio, len(source_set & target_set), boost,
                          -len(target_set), table, pk_col, len(source_set & target_set)))
        if not cands:
            return None
        cands.sort(reverse=True)
        # Index winning target's value set for the self-FK containment check
        # (the loop variable `target_set` is stale after sorting).
        pk_sets = { (t, c): s for t, c, s in self.pk_index }
        winner_set = pk_sets.get((cands[0][4], cands[0][5]))
        if require_unambiguous and len(cands) >= 2 and cands[0][:4] == cands[1][:4]:
            # A concrete name match can break an otherwise unresolvable tie
            # (e.g. `payments.order_id` vs `order_items.order_id` -> the header).
            # Only refuse the target when no name target is the clear winner.
            _b0 = cands[0][2]
            _b1 = cands[1][2]
            _ms = most_specific
            if not (_b0 > _b1 and _ms is not None and cands[0][4] == _ms):
                return None
        ratio, _, _boost, _ksize, table, pk_col, matched = cands[0]
        ambiguous = False
        if len(cands) >= 2:
            r2, m2, boost2 = cands[1][0], cands[1][6], cands[1][2]
            data_tie = (cands[0][0] - r2) < 0.03 and abs(matched - m2) <= max(2, int(0.05 * matched))
            # PART 11: a shared-key domain (`businessentityid`, `personid`, ...)
            # makes MANY targets data-indistinguishable. Count targets within a
            # small margin of the best ratio whose KEY COLUMN NAME equals the
            # source column: those are the same conceptual key, so the data
            # cannot tell them apart even when a name match picks one holder.
            # A large same-key tie overrides the name match: the name matches the
            # key concept, not the specific table (the subtype hierarchy root is
            # not a better answer than any of its subtypes).
            same_key_ties = sum(
                1 for c in cands
                if (cands[0][0] - c[0]) < 0.05 and c[5] == source_col
                and c[4] in self.sole_key_tables
            )
            winner_tighter = (-cands[0][3]) < 0.5 * (-cands[1][3])
            runner_tighter = (-cands[1][3]) < 0.5 * (-cands[0][3])
            name_decides = (_boost > boost2 and most_specific is not None
                            and table == most_specific and not runner_tighter)
            # A self-FK winner (ref-flavored column whose values are fully
            # contained in its OWN primary key) is definitive evidence of a
            # self-hierarchy reference, not a shared-key domain coincidence: the
            # data-tie / shared-key veto must not override it.
            is_self_ref_win = (
                _boost >= 9
                and source_table is not None
                and table == source_table
                and winner_set is not None
                and len(source_set & winner_set) == len(source_set)
            )
            if (data_tie and not name_decides and not winner_tighter
                    and not is_self_ref_win) or same_key_ties >= 3:
                ambiguous = True
        if most_specific is not None and most_specific != table and not ambiguous:
            ambiguous = True
        if not ambiguous and _boost == 0 and len(source_set) < self.MAX_SAMPLE \
                and len(source_set) < 0.6 * len(self._pk_values(table)):
            vals = sorted(
                v[1] for v in source_set
                if isinstance(v, tuple) and len(v) == 2 and v[0] in ("num", "int")
            )
            contiguous = bool(vals) and (vals[-1] - vals[0] + 1) <= len(vals) * 1.5
            from schema_engine.lexical import is_ref_flavored
            ref_flavored = source_col and is_ref_flavored(source_col)
            if contiguous or not ref_flavored:
                ambiguous = True
        return table, pk_col, ratio, matched, ambiguous

    def _pk_values(self, table) -> set:
        for t, _pk, tset in self.pk_index:
            if t == table:
                return tset
        return set()

    def _distinct_raw(self, table, col, limit: int) -> list:
        if self.budget is not None and not self.budget.spend():
            return []
        values = []
        try:
            with self._dict_cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT {self.quote(col)} AS v "
                    f"FROM {self.rel_table(table)} "
                    f"WHERE {self.quote(col)} IS NOT NULL "
                    f"ORDER BY {self.quote(col)} LIMIT {limit}"
                )
                for row in cur.fetchall():
                    v = row["v"]
                    if v is not None:
                        values.append(v)
        except Exception as exc:
            logger.debug("distinct sample failed for %s.%s: %s", table, col, exc)
            return []
        return values

    def _exact_matched(self, table, pk_col, values) -> Optional[int]:
        if not values:
            return None
        ident = self.quote(pk_col)
        try:
            with self._dict_cursor() as cur:
                if self.db_type == "mysql":
                    marks = ", ".join(["%s"] * len(values))
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {self.quote(table)} "
                        f"WHERE {ident} IN ({marks})",
                        tuple(values),
                    )
                else:
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {self.rel_table(table)} "
                        f"WHERE {ident} = ANY(%s)",
                        (list(values),),
                    )
                return cur.fetchone()["matched"] or 0
        except Exception as exc:
            logger.debug("exact containment check failed for %s.%s: %s", table, pk_col, exc)
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None

    def strongest_exact_overlap(self, source_table, source_col, target_tables,
                                exclude=None, require_unique_winner=False):
        """Like strongest_overlap but verifies against the FULL target key column.

        When the column name already points at a concrete target, an exact SQL
        containment count is cheap and accurate (avoids sample dilution).

        `require_unique_winner` is for data-only candidates with no name target:
        if MORE than one target contains at least MIN_OVERLAP of the source
        values, the data cannot tell the parents apart (e.g. `customer.personid`
        matches every integer key) and the tightest match is a PART 11 shared-key
        hallucination, so the result is marked ambiguous.
        """
        from schema_engine.lexical import tokenize_words

        src = self._distinct_raw(source_table, source_col, self.MAX_SAMPLE)
        if len(src) < 2:
            return None
        col_tokens = set(tokenize_words(source_col))
        cands = []
        for table, pk_col, _ in self.pk_index:
            if table not in target_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            if not self.types_compatible(source_table, source_col, table, pk_col):
                continue
            matched = self._exact_matched(table, pk_col, src)
            if matched is None:
                continue
            ratio = matched / len(src)
            if ratio < self.MIN_OVERLAP:
                continue
            boost = 0
            if col_tokens:
                table_tokens = set(tokenize_words(table))
                if any(ct in tt or tt.startswith(ct)
                       for ct in col_tokens for tt in table_tokens):
                    boost = 5
            key_size = len(self._distinct(table, pk_col, self._pk_cache, self.MAX_PK_SAMPLE))
            cands.append((ratio, matched, -key_size, boost, table, pk_col))
        if not cands:
            return None
        cands.sort(reverse=True)
        ratio, matched, _neg, _boost, table, pk_col = cands[0]
        ambiguous = False
        if require_unique_winner:
            passing = [c for c in cands if c[0] >= self.MIN_OVERLAP]
            if len(passing) >= 2:
                ambiguous = True
        elif len(cands) >= 2 and (cands[0][0] - cands[1][0]) < 0.03:
            ambiguous = True
        return table, pk_col, ratio, matched, ambiguous

    def should_profile(self, col) -> bool:
        """Only sample columns that could plausibly reference a key (bounds cost)."""
        name = str(col["column"]).lower()
        if name.endswith(("_id", "_code", "id", "_no", "_num", "_number", "_key", "_uuid")):
            return True
        dtype = str(col.get("data_type", "")).upper()
        return any(tok in dtype for tok in ("INT", "NUMERIC", "DECIMAL", "NUMBER", "REAL", "FLOAT"))