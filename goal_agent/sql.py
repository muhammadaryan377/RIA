"""Component: SQL generation, repair and semantic validation (read-only, extremes, missing joins, unknown columns, fan-out)."""

import difflib
import json
import re
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from llm_provider import LLMProvider, create_provider
from core.validation import unknown_sql_tables, empty_sql_tables, unknown_sql_columns
from core.config import SCHEMA_DIR, BASE_DIR
from schema_engine.lexical import singularize

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None

logger = logging.getLogger(__name__)

class SqlMixin:
    # ---- _is_read_only_sql (_is_read_only_sql) ----

    def _is_read_only_sql(self, sql):
        """True only for read-only SQL (SELECT / CTE-WITH). Deterministically
        rejects INSERT/UPDATE/DELETE/DDL/DCL/DML before the DB round-trip.
        Returns False for empty/None SQL (not a valid statement)."""
        if not sql:
            return False
        stripped = self._STRIP_STRINGS_RE.sub(" ", sql or "")
        return self._DESTRUCTIVE_RE.search(stripped) is None

    # ---- _warnings_for_sql (_warnings_for_sql) ----

    def _warnings_for_sql(self, sql):
        """Warn when the final query joins through an UNCERTAIN/REVIEW
        relationship (spec §5: do not silently use uncertain relationships) or
        through a low-confidence inferred relationship (spec §6)."""
        warnings = []
        if not sql:
            return warnings
        ref_tables = {t.lower() for t in self._sql_structure(sql)["in_from"]}
        if not ref_tables:
            return warnings
        for key in sorted(self.uncertain_edge_keys):
            src_table = key.split("->")[0].split(".")[0].lower()
            tgt_table = key.split("->")[1].split(".")[0].lower()
            if src_table in ref_tables and tgt_table in ref_tables:
                warnings.append(
                    f"Relationship {key} is UNCERTAIN/REVIEW in the Schema Agent "
                    "output. It was used in this query - validate the join manually."
                )
        low_trust = {"MEDIUM_CONFIDENCE_INFERRED", "LOW_CONFIDENCE_INFERRED"}
        for edge in getattr(self, "relationship_edges", []):
            if edge.get("trust") not in low_trust:
                continue
            if (edge["left_table"].lower() in ref_tables
                    and edge["right_table"].lower() in ref_tables):
                warnings.append(
                    f"Relationship {edge['left_table']}.{edge['left_column']} -> "
                    f"{edge['right_table']}.{edge['right_column']} is "
                    f"{edge['trust']} in the Schema Agent output. It was used in "
                    "this query - validate the join manually."
                )
        return warnings

    # ---- _clean_sql (_clean_sql) ----

    def _clean_sql(self, raw_sql):
        if not raw_sql or not str(raw_sql).strip():
            return "SELECT 1;"
        cleaned = str(raw_sql).strip()
        cleaned = re.sub(r"```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)
        # Only strip double-quotes that wrap the ENTIRE statement (the LLM
        # sometimes wraps its answer in quotes).  Never strip quotes around
        # individual identifiers — they protect reserved words and
        # case-sensitive names (e.g. SELECT "order" FROM "order").
        if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
            cleaned = cleaned[1:-1]
        # Small local models often ECHO the prompt's instruction sentences into
        # the answer ("with null values last. The final SQL statement is
        # provided without explanations or markdown; ..."). A real statement is
        # SELECT, or a genuine CTE (WITH <name> AS ( ... )). Plain English
        # "with ..." must NOT be mistaken for a WITH clause (it silently
        # corrupts the SQL), and the statement start must be the FIRST
        # top-level keyword: taking the last SELECT would capture inner
        # SELECTs from subqueries/CTEs and corrupt the query.
        stmt_pattern = re.compile(
            r"\bWITH\s+[A-Za-z_]\w*\s+AS\s*\(|\bSELECT\b|"
            r"\bINSERT\s+INTO\b|\bUPDATE\b|\bDELETE\s+FROM\b",
            re.IGNORECASE,
        )
        tail_pattern = re.compile(
            r"(?i)\b(the final sql statement|without explanations|no explanations|"
            r"here(?:'s| is) (?:the|your) final|output only[^\n]*|markdown|"
            r"with null values (?:last|first)[^\n]*|"
            r"this (?:query|sql) (?:is|uses|returns)[^\n]*)"
        )
        statements = []
        for fragment in cleaned.split(";"):
            match = stmt_pattern.search(fragment)
            if not match:
                continue
            frag = fragment[match.start():].strip()
            cut = tail_pattern.search(frag)
            if cut:
                frag = frag[:cut.start()].strip()
            if frag:
                statements.append(frag)
        if statements:
            return statements[-1].rstrip(";") + ";"
        # No real statement found: return the raw text so the validation layer /
        # repair loop sees it as broken and asks the model to fix it.
        return str(raw_sql).strip() + ";"

    # ---- _find_nested_aggregates (_find_nested_aggregates) ----

    def _find_nested_aggregates(self, sql):
        """Return the nested-aggregate snippets found in sql, e.g. AVG(SUM(...)).

        Matches an aggregate keyword directly wrapping another aggregate keyword.
        """
        matches = re.findall(
            r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(AVG|SUM|COUNT|MIN|MAX)\s*\(",
            sql,
            re.IGNORECASE,
        )
        return [f"{a}({b}" for a, b in matches]

    # ---- _goal_asks_for_aggregation (_goal_asks_for_aggregation) ----

    def _goal_asks_for_aggregation(self, user_goal):
        """True if the goal text clearly wants an aggregation / breakdown, in
        which case a bare `SELECT *` answer is almost certainly wrong."""
        goal = (user_goal or "").lower()
        markers = (
            "total", "count", "how many", "number of", "sum of", "average", "avg",
            "mean", "per ", "by ", "maximum", "minimum", "highest", "lowest",
            "top ", "share", "percentage", "breakdown", "distribution",
        )
        return any(m in goal for m in markers)

    # ---- _is_extremes_goal (_is_extremes_goal) ----

    def _is_extremes_goal(self, user_goal):
        """True if the goal explicitly asks for BOTH the top and bottom of a
        ranking (e.g. 'highest and lowest', 'most and least', 'max and min',
        'best and worst', 'top and bottom')."""
        goal = re.sub(r"[^a-z\s]", " ", (user_goal or "").lower())
        words = set(goal.split())
        has_max = bool(words & self._EXTREMES_MAX)
        has_min = bool(words & self._EXTREMES_MIN)
        return has_max and has_min

    # ---- _fix_extremes_sql (_fix_extremes_sql) ----

    def _fix_extremes_sql(self, user_goal, sql):
        """If the goal asks for both extremes and `sql` is a single-sided
        `ORDER BY ... (ASC|DESC) ... LIMIT 1`, strip the trailing LIMIT so the
        full ranking comes back. Returns (sql, changed)."""
        if not sql or not self._is_extremes_goal(user_goal):
            return sql, False
        if not re.search(r"\bORDER\s+BY\b[\s\S]*\b(ASC|DESC)\b", sql, re.IGNORECASE):
            return sql, False
        new_sql, n = re.subn(
            r"\s+LIMIT\s+\d+(?:\s+OFFSET\s+\d+)?(\s*;?\s*)$",
            r"\1",
            sql.strip(),
            count=1,
            flags=re.IGNORECASE,
        )
        if n:
            logger.warning(
                "Extremes goal detected; stripped trailing LIMIT so both the "
                "highest and lowest appear in the result set."
            )
            return new_sql.strip(), True
        return sql, False

    # ---- _sql_structure (_sql_structure) ----

    def _sql_structure(self, sql):
        """Return a dict describing the FROM/JOIN structure of `sql`:

          - in_from:    set of table names present in FROM/JOIN clauses
          - alias_to_name, name_to_alias: alias mappings
          - first_ref:  reference token (alias or bare name) of the FIRST
                        FROM table — the only table guaranteed visible at the
                        point right after the FROM clause.
          - intro:      table name -> byte position just after the clause that
                        introduces it (end of the FROM/JOIN block).

        Implemented as a positional scan that skips parenthesized subqueries
        so a JOIN/ON keyword inside a subquery is never mistaken for a
        top-level clause.
        """
        out = {
            "in_from": set(),
            "alias_to_name": {},
            "name_to_alias": {},
            "first_ref": None,
            "intro": {},
        }
        pos = 0
        first = True
        depth = 0
        while pos < len(sql):
            ch = sql[pos]
            if ch == "(":
                depth += 1
                pos += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                pos += 1
                continue
            if depth > 0:
                pos += 1
                continue
            m = re.search(
                r"\b(?:FROM|JOIN)\s+([\"'`]?)([A-Za-z_][A-Za-z0-9_]*)\1\b",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            table = m.group(2)
            out["in_from"].add(table)
            after = pos + m.end()
            alias = None
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if am:
                alias = am.group(1)
            else:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() not in self._SQL_KEYWORDS:
                    alias = am.group(1)
            if alias and alias.lower() not in self._SQL_KEYWORDS:
                out["alias_to_name"][alias] = table
                out["name_to_alias"][table] = alias
                after += am.end()
            if first:
                out["first_ref"] = out["name_to_alias"].get(table, table)
                first = False
            out["intro"][table] = self._clause_end(sql, after)
            pos = after
        return out

    # ---- _clause_end (_clause_end) ----

    def _clause_end(self, sql, start):
        """Byte position where the clause starting at `start` ends: the next
        top-level JOIN/WHERE/GROUP/ORDER/HAVING/LIMIT/OFFSET keyword or `;`."""
        depth = 0
        i = start
        n = len(sql)
        while i < n:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ";":
                return i
            elif depth == 0:
                m = re.match(
                    r"\b(?:JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|OFFSET)\b",
                    sql[i:], re.IGNORECASE,
                )
                if m:
                    return i
            i += 1
        return n

    # ---- _missing_refs (_missing_refs) ----

    def _missing_refs(self, sql):
        """Return a list of {'table', 'ref'} for tables referenced via a
        qualified column but missing from FROM/JOIN. `ref` is the token the
        query actually uses for that table (an alias like `b`, or the bare
        table name). Unknown prefixes are resolved to the single schema table
        whose name starts with that prefix (the model invented an alias without
        declaring `FROM <table> <alias>`)."""
        if not sql:
            return []
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}
        if not in_from_lc:
            return []
        table_names_lc = {t.lower() for t in self.tables}
        found = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql):
            prefix = m.group(1).lower()
            if prefix in struct["alias_to_name"]:
                table = struct["alias_to_name"][prefix].lower()
            elif prefix in table_names_lc:
                table = prefix
            else:
                candidates = [t for t in self.tables if t.lower().startswith(prefix)]
                if len(candidates) != 1:
                    continue
                table = candidates[0].lower()
            if table not in in_from_lc and table in table_names_lc:
                found[table] = prefix
        return [{"table": t, "ref": r} for t, r in sorted(found.items())]

    # ---- _find_missing_from_tables (_find_missing_from_tables) ----

    def _find_missing_from_tables(self, sql):
        """Return table names that are referenced (via `tbl.col` or a JOIN ON
        condition) but never appear in a FROM/JOIN clause."""
        return [m["table"] for m in self._missing_refs(sql)]

    # ---- _inject_missing_joins (_inject_missing_joins) ----

    def _inject_missing_joins(self, sql, missing):
        """Deterministically inject JOINs for tables that are referenced but
        missing from FROM/JOIN. Only inject when there is a clear FK edge to the
        FIRST table in FROM â€” the one guaranteed to be visible at the insertion
        point (a table referenced in an earlier ON clause must be introduced
        before that clause). Returns (new_sql, injected_count) or (sql, 0)."""
        if not missing:
            return sql, 0
        struct = self._sql_structure(sql)
        first_ref = struct["first_ref"]
        intro_pos = struct["intro"]
        if first_ref is None or not intro_pos:
            return sql, 0
        first_table = struct["alias_to_name"].get(first_ref, first_ref)
        if first_table not in intro_pos:
            return sql, 0
        edges = self._fk_edges()
        joins = []
        for item in self._missing_refs(sql):
            table = item["table"]
            ref = item["ref"]
            real = next((t for t in self.tables if t.lower() == table), table)
            alias_txt = "" if ref == table else f" {ref}"
            candidate = None
            for (src, col, ref_t, ref_col) in edges:
                if src.lower() == table and ref_t.lower() == first_table.lower():
                    candidate = f"JOIN {real}{alias_txt} ON {ref}.{col} = {first_ref}.{ref_col}"
                    break
            if candidate is None:
                for (src, col, ref_t, ref_col) in edges:
                    if ref_t.lower() == table and src.lower() == first_table.lower():
                        candidate = f"JOIN {real}{alias_txt} ON {first_ref}.{col} = {ref}.{ref_col}"
                        break
            if candidate:
                joins.append(candidate)
        if not joins:
            return sql, 0
        inject_at = intro_pos[first_table]
        injection = " " + " ".join(joins)
        new_sql = sql[:inject_at] + injection + " " + sql[inject_at:]
        return new_sql, len(joins)

    # ---- _join_clauses (_join_clauses) ----

    def _join_clauses(self, sql):
        """Return (start, end, table, on_start, on_end) for every JOIN clause,
        where on_start/on_end span the ON condition (None if no ON clause)."""
        clauses = []
        pos = 0
        while True:
            m = re.search(
                r"\bJOIN\s+([\"'`]?)([A-Za-z_][A-Za-z0-9_]*)\1\b",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            start = pos + m.start()
            table = m.group(2)
            after = pos + m.end()
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if not am:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() in self._SQL_KEYWORDS:
                    am = None
            if am:
                after += am.end()
            om = re.search(r"\bON\b", sql[after:], re.IGNORECASE)
            if not om:
                end = self._clause_end(sql, after)
                clauses.append((start, end, table, None, None))
            else:
                cond_start = after + om.end()
                cond_end = self._clause_end(sql, cond_start)
                clauses.append((start, cond_end, table, cond_start, cond_end))
            pos = cond_end if om else end
        return clauses

    # ---- _repair_unknown_columns (_repair_unknown_columns) ----

    def _repair_unknown_columns(self, sql, columns_by_table, bad_columns):
        """Deterministically fix hallucinated column references WITHOUT an LLM.

        Handles two failure modes the LLM repair loop repeatedly fails on:
          1. `tbl.col` where `col` actually lives on another table in the query's
             schema -> rewrite the qualifier to the owning table (the missing-FROM
             guard then injects the required FK join).
          2. a JOIN whose ON condition uses an unresolvable column -> drop that
             hallucinated JOIN entirely (only when the joined table is not used
             anywhere else in the query).

        JOIN ON predicates whose rewrite would self-compare the same table
        (``rooms.room_id = rooms.room_id``) are refused and instead dropped or
        left for the LLM. Returns (repaired_sql, changed).
        """
        if not bad_columns or not sql:
            return sql, False
        bad_set = {b.lower() for b in bad_columns}
        table_cols_lc = {t: {c.lower() for c in cols} for t, cols in columns_by_table.items()}
        col_owners = {}
        for t, cols in table_cols_lc.items():
            for c in cols:
                col_owners.setdefault(c, []).append(t)
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}

        def resolve_prefix(prefix):
            p = prefix.lower()
            if p in struct["alias_to_name"]:
                return struct["alias_to_name"][p].lower()
            if p in table_cols_lc:
                return p
            cands = [t for t in table_cols_lc if t.startswith(p)]
            return cands[0].lower() if len(cands) == 1 else None

        def near_match(table, col):
            norm = re.sub(r"[^a-z0-9]", "", col.lower())
            for c in table_cols_lc.get(table, ()):
                if re.sub(r"[^a-z0-9]", "", c) == norm:
                    return c
            for c in table_cols_lc.get(table, ()):
                if len(c) >= 4 and (norm.rstrip("s") == c.rstrip("s") or norm == c[:-1]):
                    return c
            return None

        on_regions = [(s, e) for (_, _, _, s, e) in self._join_clauses(sql) if s is not None]

        def other_side_table(pos, on_start, on_end):
            """Resolve the table of the operand on the other side of the `=`
            nearest to `pos` within an ON condition."""
            seg = sql[on_start:on_end]
            local = pos - on_start
            eq = seg.rfind("=", 0, local)
            eq2 = seg.find("=", local)
            bounds = []
            if eq != -1:
                bounds.append((seg.rfind("=", 0, eq - 1), eq))
            if eq2 != -1:
                bounds.append((eq, seg.find("=", eq2 + 1)))
            region = None
            for lo, hi in bounds:
                if lo <= local <= hi:
                    region = seg[lo + 1:hi if hi != -1 else len(seg)]
                    break
            if not region:
                return None
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b", region)
            return resolve_prefix(m.group(1)) if m else None

        def in_on(pos):
            return next(((s, e) for s, e in on_regions if s <= pos < e), None)

        # --- pass 1: rewrite qualifiers to the column's real owner -------------
        replacements = []
        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql
        ):
            prefix, col = m.group(1), m.group(2)
            if col == "*":
                continue
            tbl = resolve_prefix(prefix)
            if tbl is None or f"{tbl}.{col}".lower() not in bad_set:
                continue
            real = near_match(tbl, col)
            if real:
                replacements.append((m.start(), m.end(), f"{prefix}.{real}"))
                continue
            owners = [t for t in col_owners.get(col.lower(), []) if t != tbl and t not in in_from_lc]
            if len(owners) != 1:
                continue
            region = in_on(m.start())
            if region:
                other = other_side_table(m.start(), region[0], region[1])
                if other == owners[0]:
                    continue  # would become X.x = X.x (self-comparison); drop/LLM instead
            replacements.append((m.start(), m.end(), f"{owners[0]}.{col}"))
        if replacements:
            out = sql
            for start, end, repl in sorted(replacements, reverse=True):
                out = out[:start] + repl + out[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out, True

        # --- pass 2: drop hallucinated JOINs (unresolvable ON column) ----------
        for (start, end, table, on_start, on_end) in self._join_clauses(sql):
            if on_start is None:
                continue
            cond = sql[on_start:on_end]
            used_elsewhere = False
            alias = struct["alias_to_name"].get(table)
            for token in {table, alias} - {None}:
                remaining = sql[:start] + sql[end:]
                if re.search(rf"\b{re.escape(token)}\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
                             remaining, re.IGNORECASE):
                    used_elsewhere = True
                    break
            if used_elsewhere:
                continue
            cond_bad = [b for b in bad_set if re.search(
                rf"\b{re.escape(b.split('.')[0])}\s*\.\s*{re.escape(b.split('.')[1])}\b",
                cond, re.IGNORECASE)]
            if not cond_bad:
                continue
            out = sql[:start] + sql[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out.strip(), True
        return sql, False

    # ---- _find_bare_select_star (_find_bare_select_star) ----

    def _find_bare_select_star(self, sql):
        """Return a short description if `sql` is a bare `SELECT *` (no aggregate
        function, no GROUP BY). None otherwise. Never flags `SELECT t.*` that is
        accompanied by an aggregate."""
        m = re.search(r"\bselect\b([\s\S]*?)\bfrom\b", sql, re.IGNORECASE)
        if not m:
            return None
        select_part = m.group(1)
        if "*" not in select_part:
            return None
        has_agg = re.search(
            r"\b(?:count|sum|avg|average|min|max|stddev|std|variance|var_pop|var_samp|median|percentile_cont|percentile_disc)\s*\(",
            sql, re.IGNORECASE,
        )
        if has_agg:
            return None
        return "SELECT * with no aggregation"

    # ---- _find_suspicious_pk_joins (_find_suspicious_pk_joins) ----

    def _find_suspicious_pk_joins(self, sql):
        """Return equality join pairs that equate the PRIMARY KEYS of two
        DIFFERENT tables. In a normalized schema that is almost always a bug
        (e.g. `tracks.id = artists.id`); real joins go PK -> FK.

        Exception: a column that is both a member of a COMPOSITE primary key and
        a foreign key to the joined parent (e.g. order_details.product_id when
        order_details PK is (order_id, product_id)) is a legitimate FK join and
        is never flagged."""
        pk_of = {
            name: set(info.get("primary_key", []) or info.get("inferred_primary_key", []) or [])
            for name, info in self.tables.items()
        }
        # {table: {column: (referenced_table, referenced_column)}} from the FK edges.
        fk_of = {}
        for name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                fk_of.setdefault(name, {})[fk["column"]] = (
                    fk["referenced_table"], fk["referenced_column"],
                )

        # Resolve aliases to table names: FROM x [AS] a, JOIN y [AS] b.
        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"'`]?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        # Capture each ON clause and the equality conditions inside it.
        for on_m in re.finditer(
            r"\bJOIN\s+([\"'`]?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            joined_table, joined_alias, on_clause = on_m.group(2), on_m.group(3).lower(), on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in pk_of.get(ta, set()) and cb in pk_of.get(tb, set()):
                    if (ca in fk_of.get(ta, {}) and fk_of[ta][ca] == (tb, cb)) or (
                        cb in fk_of.get(tb, {}) and fk_of[tb][cb] == (ta, ca)
                    ):
                        continue
                    suspects.append(f"{ta}.{ca} = {tb}.{cb} (joins primary keys of unrelated tables)")
        return suspects

    # ---- _find_fk_mismatch_joins (_find_fk_mismatch_joins) ----

    def _find_fk_mismatch_joins(self, sql):
        """Return join equalities where an FK column is joined to a table it does
        NOT reference. E.g. tracks.album_id (FK -> albums.id) joined to artists.id
        is wrong: the FK definition says it must join albums, not artists."""
        # {table: {column: (referenced_table, referenced_column)}}
        fk_of = {}
        for name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                fk_of.setdefault(name, {})[fk["column"]] = (fk["referenced_table"], fk["referenced_column"])

        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"'`]?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        for on_m in re.finditer(
            r"\bJOIN\s+([\"'`]?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            on_clause = on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in fk_of.get(ta, {}):
                    ref_table, ref_col = fk_of[ta][ca]
                    if ref_table != tb or ref_col != cb:
                        suspects.append(
                            f"{ta}.{ca} references {ref_table}.{ref_col} but is joined to {tb}.{cb}"
                        )
                elif cb in fk_of.get(tb, {}):
                    ref_table, ref_col = fk_of[tb][cb]
                    if ref_table != ta or ref_col != ca:
                        suspects.append(
                            f"{tb}.{cb} references {ref_table}.{ref_col} but is joined to {ta}.{ca}"
                        )
        return suspects

    # ---- _llm_unavailable (_llm_unavailable) ----

    def _llm_unavailable(self, exc):
        """True when the language-model provider is down or rate-limited, as
        opposed to a genuine SQL-generation problem we could repair."""
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
            return True
        if any(k in msg for k in ("connection", "timed out", "timeout", "connect", "refused")):
            return True
        if "ollama" in msg and ("down" in msg or "refused" in msg or "connect" in msg):
            return True
        return False

    # ---- _generate_sql (_generate_sql) ----

    def _generate_sql(self, user_goal, join_path, kpi_map, ir=None, join_plan=None):
        schema_ddl = self._build_schema_ddl(join_path, full=False)
        if not schema_ddl:
            return "SELECT 1;"

        kpi_desc = ", ".join(k["description"] for k in kpi_map["kpis"]) or "none"

        ir_summary = ""
        if ir:
            bits = []
            if ir.get("aggregation"):
                bits.append(f"aggregation: {ir['aggregation']}")
            resolved_metrics = [m for m in ir.get("metrics", []) if m.get("resolved_expression")]
            if resolved_metrics:
                def _render_measure(m):
                    agg = m.get("aggregation")
                    expr = (m.get("resolved_expression") or "").strip()
                    if not expr:
                        return None
                    if expr.upper() == "COUNT(*)" or agg == "COUNT":
                        return expr
                    if agg:
                        return f"{agg}({expr})"
                    return expr
                bits.append("resolved measures: " + ", ".join(
                    r for r in (_render_measure(m) for m in resolved_metrics) if r))
            if ir.get("dimensions"):
                bits.append("dimensions: " + ", ".join(
                    d["concept"] for d in ir["dimensions"]))
            if ir.get("time"):
                t = ir["time"]
                bits.append(
                    f"time: {t.get('constraint')} {t.get('value', '')} "
                    f"on {t.get('column') or 'the relevant date column'}")
            if ir.get("ranking"):
                r = ir["ranking"]
                bits.append(
                    f"ranking: top {r.get('limit', 'N')} by the measure "
                    f"({r['direction']})")
            if ir.get("filters"):
                bits.append("filters: " + ", ".join(
                    f"{f.get('column') or 'date'} {f['operator']} {f['value']}"
                    for f in ir["filters"]))
            if bits:
                ir_summary = (
                    "### Resolved goal plan (deterministic, treat as ground truth)\n"
                    + "\n".join(f"- {b}" for b in bits) + "\n"
                )

        join_plan_text = ""
        if join_plan and join_plan.get("edges"):
            lines = [
                f"  {e['left_table']}.{e['left_column']} = "
                f"{e['right_table']}.{e['right_column']} [{e.get('trust', 'declared')}]"
                for e in join_plan["edges"]
            ]
            join_plan_text = (
                "### Deterministic join plan (MUST be followed exactly)\n"
                "Join ONLY through these validated relationships:\n"
                + "\n".join(lines) + "\n"
            )

        if self.dialect == "sqlite":
            dialect_note = (
                "### SQLite notes\n"
                "Use standard SQLite syntax: strftime('%Y', col) / strftime('%Y-%m', col) "
                "for dates instead of EXTRACT, and || for concatenation.\n"
            )
        elif self.dialect == "mysql":
            dialect_note = (
                "### MySQL notes\n"
                "Quote identifiers with backticks when needed. Use DATE_FORMAT(col, '%Y') / "
                "DATE_FORMAT(col, '%Y-%m') for date grouping instead of EXTRACT/strftime. "
                "Use NOW()/CURDATE() for current time, LIKE (there is no ILIKE), LIMIT for "
                "paging, and CONCAT() instead of ||. Backslash is the string escape character.\n"
            )
        else:
            dialect_note = ""

        prompt = f"""
### Task
Generate a single {self.dialect} query that answers the user's business goal.
User goal: {user_goal}
Aligned KPIs: {kpi_desc}
Dimensions: {', '.join(kpi_map['dimensions']) if kpi_map['dimensions'] else 'auto-detect'}

{ir_summary}
### Database Schema
The query will run on a database with the following schema:
{schema_ddl}
Preferred starting tables (inspect these first, but you may use any table
needed to answer the goal, joining via the FOREIGN KEY relationships above):
{join_path}
{join_plan_text}{dialect_note}### SQL
Output only the final SQL statement. No explanations, no markdown, nothing after it.
Rules:
- Aggregate functions must NEVER be nested (AVG(SUM(x)), SUM(COUNT(x)), etc. are invalid SQL).
- An average per group is SUM(x) / COUNT(DISTINCT key) or a subquery.
- If the goal asks for BOTH extremes (e.g. "highest AND lowest", "most and least",
  "top and bottom", "max and min", "best and worst"), do NOT use LIMIT 1: return the
  full ranked list (ORDER BY ... ASC or DESC, no LIMIT) so both ends are present.
- If the goal asks for a single extreme ("the highest", "the lowest", "top 5"), a
  one-sided ORDER BY + LIMIT is fine.
- Do not add a trailing semicolon only statement marker; a single trailing semicolon is fine.
"""
        try:
            raw = self.llm.complete("sql", prompt, temperature=0.1, num_predict=500)
            return self._clean_sql(raw)
        except Exception as exc:
            logger.warning(f"LLM SQL generation failed: {exc}.")
            if self._llm_unavailable(exc):
                # Provider down/exhausted: do NOT fall back to a bare SELECT *
                # (that would only trigger pointless repair loops). Signal the
                # caller to degrade gracefully.
                return None
            if self._goal_asks_for_aggregation(user_goal):
                # Spec §15: a bare SELECT * is NOT a valid substitute for an
                # aggregated answer. Fail controlled instead of silently
                # returning raw rows that mislead the user.
                return None
            return self._fallback_sql(join_path)

    # ---- _semantic_validate_sql (_semantic_validate_sql) ----

    def _semantic_validate_sql(self, ir, sql, join_path, join_plan=None):
        """Semantic SQL validation (spec §9/§10): the final SQL must actually
        implement the resolved Goal IR â€” required tables in FROM, measures
        aggregated, filters/time/ranking present â€” and must respect the
        explicit deterministic join plan.

        Returns (issues, warnings). Empty issues = PASS. Deterministic; the
        LLM is NOT the authority on the plan. Runs after every repair so a
        repaired query is re-validated before execution.
        """
        issues = []
        warnings = []
        if not ir or not sql:
            return issues, warnings
        struct = self._sql_structure(sql)
        in_from = {t.lower() for t in struct["in_from"]}
        sql_lower = sql.lower()
        sql_upper = sql.upper()

        required = set()
        for m in ir.get("metrics", []):
            if m.get("resolved_table"):
                required.add(m["resolved_table"])
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                required.add(d["resolved_table"])
        if not required and join_path:
            required.add(join_path[0])
        missing = [t for t in sorted(required) if t.lower() not in in_from]
        if missing:
            issues.append(
                f"required table(s) {', '.join(missing)} are missing from FROM/JOIN"
            )

        if ir.get("aggregation") == "COUNT" and "COUNT(" not in sql_upper:
            issues.append("the goal asks for a count but the SQL has no COUNT(...)")
        elif ir.get("aggregation") and not re.search(
                r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", sql_upper):
            issues.append(
                f"the goal requires {ir['aggregation']} aggregation but the SQL "
                "has no aggregate function"
            )

        for m in ir.get("metrics", []):
            agg = m.get("aggregation")
            col = m.get("resolved_column")
            if not agg or not col:
                continue
            if agg.upper() == "COUNT":
                continue
            if f"{agg.upper()}(" not in sql_upper:
                issues.append(
                    f"the resolved metric requires {agg.upper()} over "
                    f"{m.get('resolved_expression')} but the SQL has no {agg.upper()}"
                )
            elif col.lower() not in sql_lower:
                issues.append(
                    f"the resolved measure column {m.get('resolved_expression')} "
                    "is not referenced by the SQL"
                )

        where_l = self._where_clause(sql).lower()
        for f in ir.get("filters", []):
            col = f.get("column")
            if not col:
                continue
            token = col.split(".")[-1].lower()
            if token and token not in where_l:
                issues.append(
                    f"the requested filter on {col} is not applied in the SQL WHERE clause"
                )

        if ir.get("time") and ir["time"].get("column"):
            date_token = ir["time"]["column"].split(".")[-1].lower()
            if date_token and date_token not in sql_lower:
                issues.append(
                    "the requested time constraint is not applied to any date column"
                )

        if ir.get("ranking"):
            if ir["ranking"].get("limit") is not None:
                if not re.search(r"\bLIMIT\s+\d+", sql_upper):
                    issues.append(
                        "the goal asks for a top/bottom N but the SQL has no LIMIT"
                    )
            if not re.search(r"\bORDER\s+BY\b", sql_upper):
                issues.append(
                    "the goal asks for a ranked result but the SQL has no ORDER BY"
                )

        if join_plan and join_plan.get("edges"):
            joined_pairs = set()
            for j in self._extract_sql_joins(sql):
                joined_pairs.add((j["left_table"].lower(), j["right_table"].lower()))
                joined_pairs.add((j["right_table"].lower(), j["left_table"].lower()))
            plan_nodes = {t.lower() for t in join_plan.get("nodes", [])}
            for e in join_plan["edges"]:
                pair = (e["left_table"].lower(), e["right_table"].lower())
                if pair not in joined_pairs:
                    issues.append(
                        f"the planned join {e['left_table']}.{e['left_column']} = "
                        f"{e['right_table']}.{e['right_column']} is missing from the SQL"
                    )
            extra = [t for t in in_from if t not in plan_nodes]
            if extra:
                warnings.append(
                    f"SQL uses table(s) {', '.join(sorted(extra))} outside the "
                    "deterministic join plan"
                )

        # ---- GROUP-BY granularity (spec §12 / Fix 3: AOV over-grouping) ----
        # A scalar overall aggregate (e.g. "average order value", "total sales")
        # must yield a single row: the outer SELECT must NOT carry a GROUP BY.
        # An LLM frequently over-groups AOV (e.g. AVG(total) GROUP BY customer_id),
        # splitting the single scalar metric into per-group rows and returning the
        # wrong answer. Flag it so the query is repaired to a single-row aggregate.
        #
        # Per-entity / "top N" / "by X" / "per X" / comparison goals are NOT
        # flagged: they carry intent ranking/comparison/distribution/trend, a
        # ranking block, a comparison block, or explicit dimensions, so a GROUP BY
        # is correct for them.
        #
        # Only the OUTERMOST (parenthesis-depth-0) GROUP BY is considered, so a
        # correct per-order AOV subquery
        #   SELECT AVG(v) FROM (SELECT SUM(x) AS v FROM ... GROUP BY order_id) t
        # is not mis-flagged -- its GROUP BY lives inside a subquery.
        _gb_match = re.search(r"\bGROUP\s+BY\s+", sql, re.IGNORECASE)
        _outer_gb = []
        if _gb_match:
            _depth = sql[:_gb_match.start()].count("(") - sql[:_gb_match.start()].count(")")
            if _depth == 0:
                _outer_gb = self._extract_sql_group_by(sql)
        if (
            _outer_gb
            and ir.get("aggregation")
            and ir.get("intent") == "summary"
            and not ir.get("ranking")
            and not ir.get("comparison")
            and not ir.get("dimensions")
            and re.search(r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", sql_upper)
        ):
            issues.append(
                "the goal asks for an overall aggregate (no breakdown) but the SQL "
                "groups by "
                + ", ".join(_outer_gb)
                + " (over-grouping: a single aggregate metric must not be split "
                "into per-group rows)"
            )

        for w in self._detect_fan_out(ir, join_plan, in_from):
            warnings.append(w)
        return issues, warnings

    # ---- _detect_fan_out (_detect_fan_out) ----

    def _detect_fan_out(self, ir, join_plan, in_from):
        """Spec §18: warn when a SUM/AVG metric is summed over a table that is
        an ANCESTOR of the row-grain table. In a many-to-one chain such as
        customers -> orders -> order_items, summing an order_items grain by a
        customers-level value would multiply it by the number of matching child
        rows (row fan-out / double counting). Deterministic; warning only.
        """
        if not ir or not join_plan or not join_plan.get("edges"):
            return []
        metrics = [m for m in ir.get("metrics", []) if m.get("resolved_table")]
        if not metrics:
            return []
        sums = [
            m for m in metrics
            if str(m.get("aggregation", "")).upper() in ("SUM", "AVG")
        ]
        if not sums:
            return []
        grain = None
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                grain = d["resolved_table"]
                break
        if not grain:
            grain = next(iter(in_from)) if in_from else None
        if not grain:
            return []
        child_to_parent = {}
        for e in join_plan["edges"]:
            if e.get("cardinality") == "many_to_one":
                child_to_parent.setdefault(
                    e["left_table"].lower(), set()
                ).add(e["right_table"].lower())
        if not child_to_parent:
            return []
        warnings = []
        for m in sums:
            mt = m["resolved_table"].lower()
            if mt == grain.lower():
                continue
            if mt not in in_from or grain.lower() not in in_from:
                continue
            # Walk up from the grain table along many-to-one (child -> parent)
            # edges. If we reach the metric's table, the metric table is an
            # ancestor of the grain -> row fan-out is possible.
            seen, stack = {grain.lower()}, [grain.lower()]
            reached = False
            while stack:
                cur = stack.pop()
                for parent in child_to_parent.get(cur, []):
                    if parent == mt:
                        reached = True
                        break
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
                if reached:
                    break
            if reached:
                warnings.append(
                    f"possible row fan-out: the {m.get('aggregation')} over "
                    f"{m.get('resolved_expression') or m.get('resolved_column')} is "
                    f"taken at table {m['resolved_table']}, an ancestor of the row "
                    f"grain table {grain}; the metric may be multiplied across "
                    "matching child rows"
                )
        return warnings

    # ---- _fallback_sql (_fallback_sql) ----

    def _fallback_sql(self, join_path):
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            if self.dialect == "sqlite":
                return "SELECT name FROM sqlite_master WHERE type='table' LIMIT 10;"
            return "SELECT table_name FROM information_schema.tables LIMIT 10;"
        return f'SELECT * FROM "{relevant[0]}" LIMIT 50;'

    # ---- _row_counts (_row_counts) ----

    def _row_counts(self):
        """Count rows per table (cached per agent) so a genuinely empty table
        can be told apart from a faulty query."""
        if self._row_counts_cache is not None:
            return self._row_counts_cache
        counts = {}
        try:
            with self.engine.connect() as conn:
                for table_name in self.tables:
                    try:
                        result = conn.execute(text(f'SELECT count(*) FROM "{table_name}"'))
                        counts[table_name] = int(result.scalar() or 0)
                    except Exception:
                        counts[table_name] = -1
        except Exception:
            pass
        self._row_counts_cache = counts
        return counts

