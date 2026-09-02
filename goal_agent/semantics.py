"""Component: word / synonym normalization and generic business-concept matching (spec section 5)."""

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


class SemanticMixin:
    # ---- _normalize_tokens (_normalize_tokens) ----

    def _normalize_tokens(self, raw_text):
        tokens = set()
        for word in re.findall(r"[a-zA-Z0-9_]+", str(raw_text)):
            tokens.add(word.lower())
            tokens.update(part for part in word.split("_") if part)
        return {t for t in tokens if t}

    # ---- _has_resolvable_analytical_structure (_has_resolvable_analytical_structure) ----

    def _has_resolvable_analytical_structure(self, goal_l):
        """True when the goal carries a metric concept AND an operation/grouping
        word, so the semantic layer downstream should attempt it rather than
        asking for clarification up front (spec §7)."""
        tokens = self._normalize_tokens(goal_l)
        concept = tokens & self._METRIC_CONCEPT_WORDS
        operation = tokens & self._OPERATION_WORDS
        return bool(concept and operation)

    # ---- _synonym_bases (_synonym_bases) ----

    def _synonym_bases(self, word):
        """Set of canonical meaning keys for a word (or the word itself when
        it is not a known business concept). Singular/plural and the generic
        synonym groups are both expanded, so 'client' -> {'client','customer'}."""
        w = str(word).lower().strip()
        bases = {w}
        for base, group in self.SEMANTIC_SYNONYMS.items():
            if w == base or w in group:
                bases.add(base)
        if w.endswith("ies") and len(w) > 4:
            bases |= self._synonym_bases(w[:-3] + "y")
        if w.endswith("s") and w[:-1] and w[:-1] not in bases:
            bases |= self._synonym_bases(w[:-1])
        return bases

    # ---- _schema_token_matches_goal (_schema_token_matches_goal) ----

    def _schema_token_matches_goal(self, schema_token, goal_token):
        """True when a goal word means the same as a schema word, using exact,
        singular/plural and synonym matching (spec §5). Underscore-parts of the
        schema token (model_year -> model, year) are compared individually."""
        gt = str(goal_token).lower().strip()
        st = str(schema_token).lower().strip()
        gb = self._synonym_bases(gt)
        if gb & self._synonym_bases(st):
            return True
        for part in st.split("_"):
            if not part:
                continue
            if gb & self._synonym_bases(part):
                return True
        return False

    # ---- _match_tables (_match_tables) ----

    def _match_tables(self, goal_l):
        """Tables whose name means something mentioned in the goal, scored by
        how many goal tokens matched (synonym-aware). Database-agnostic."""
        goal_tokens = self._normalize_tokens(goal_l)
        if not goal_tokens:
            return []
        scored = []
        for table in self.tables:
            if self.tables[table].get("empty"):
                continue
            parts = set(table.lower().split("_"))
            score = 0
            for tok in goal_tokens:
                if tok in self._ANALYTICAL_GOAL_WORDS:
                    continue
                if any(
                    self._schema_token_matches_goal(part, tok)
                    or self._schema_token_matches_goal(tok, part)
                    for part in parts
                ):
                    score += 1
            if score:
                scored.append((score, table))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    # ---- _match_columns (_match_columns) ----

    def _match_columns(self, goal_l, table):
        """Columns of `table` whose name means something mentioned in the goal
        (synonym-aware). Returns (score, column) sorted descending."""
        goal_tokens = self._normalize_tokens(goal_l)
        if not goal_tokens:
            return []
        scored = []
        for col in self.tables[table].get("columns", {}):
            parts = set(col.lower().split("_"))
            score = 0
            for tok in goal_tokens:
                if tok in self._ANALYTICAL_GOAL_WORDS:
                    continue
                if any(
                    self._schema_token_matches_goal(part, tok)
                    or self._schema_token_matches_goal(tok, part)
                    for part in parts
                ):
                    score += 1
            if score:
                scored.append((score, col))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    # ---- _build_schema_word_tokens (_build_schema_word_tokens) ----

    def _build_schema_word_tokens(self):
        """Every token piece of the table and column names in the schema
        mapping. Used as the vocabulary for Google-style typo correction."""
        vocab = set()
        for table_name, info in self.tables.items():
            vocab |= self._normalize_tokens(table_name)
            for col in info.get("columns", {}):
                vocab |= self._normalize_tokens(col)
        return vocab

    # ---- _correct_goal (_correct_goal) ----

    def _correct_goal(self, user_goal):
        """Auto-correct obvious typos in schema words, Google-search style.

        Returns (corrected_goal, [(typo_word, suggestion), ...]). A token is
        corrected only when it is NOT a protected business/function word, does
        NOT already resolve to a schema reference (exact or plural), and has a
        STRONG fuzzy match (>= 0.8) to a table or column name token. The
        original wording is preserved in the contract's goal.original_question
        and a "Did you mean ..." warning is emitted, so a wrong guess is never
        silent.
        """
        corrections = []
        if not user_goal:
            return user_goal, corrections
        vocab = self._schema_word_tokens
        protected = self._PROTECTED_GOAL_WORDS
        corrected = user_goal
        seen = set()
        # Pre-index vocab by first 2 chars for O(bucket_size) lookups.
        _vocab_index = defaultdict(set)
        for v in vocab:
            if len(v) >= 2:
                _vocab_index[v[:2]].add(v)
        for word in re.findall(r"[a-zA-Z0-9_]+", user_goal):
            low = word.lower()
            if low in seen or not low.isalpha():
                continue
            seen.add(low)
            if low in protected:
                continue
            if low in vocab or (low + "s") in vocab or (low.endswith("s") and low[:-1] in vocab):
                continue
            prefix = low[:2] if len(low) >= 2 else low
            candidates = _vocab_index.get(prefix, vocab)
            best, score = None, 0.0
            for cand in candidates:
                ratio = difflib.SequenceMatcher(None, low, cand).ratio()
                if ratio > score:
                    best, score = cand, ratio
            if best and score >= 0.8:
                corrections.append((word, best))
                corrected = re.sub(
                    r"\b" + re.escape(word) + r"\b", best, corrected,
                    flags=re.IGNORECASE,
                )
        return corrected, corrections

    # ---- _natural_label (_natural_label) ----

    def _natural_label(self, identifier):
        """Turn an identifier into a plain-English phrase: 'order_details'
        -> 'order detail', 'unit_price' -> 'unit price'."""
        words = [w for w in re.split(r"[^a-zA-Z0-9]+", str(identifier)) if w]
        if not words:
            return str(identifier)
        return " ".join(singularize(w.lower()) for w in words)

    # ---- _pluralize_word (_pluralize_word) ----

    def _pluralize_word(self, word):
        """Best-effort English plural of a single word."""
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        return word + "s"

    # ---- _plural_label (_plural_label) ----

    def _plural_label(self, identifier):
        """Natural plural phrase: 'categories' -> 'category' -> 'categories'."""
        words = self._natural_label(identifier).split()
        if not words:
            return self._natural_label(identifier)
        words[-1] = self._pluralize_word(words[-1])
        return " ".join(words)

    # ---- _classify_column (_classify_column) ----

    def _classify_column(self, table_name, col):
        data_type = str(
            self.tables[table_name]["columns"][col].get("data_type", "")
        ).lower()
        if any(token in data_type for token in self._NUMERIC_TYPES):
            return "numeric"
        if any(token in data_type for token in self._DATE_TYPES):
            return "date"
        return "text"

    # ---- _is_entity_table (_is_entity_table) ----

    def _is_entity_table(self, table_name):
        """A table is an 'entity' when other tables reference it, or it has a
        real text/label column (so it can be a subject of questions). Pure
        junction/fact tables with only keys+numbers are not entities."""
        for other, other_info in self.tables.items():
            if other == table_name:
                continue
            for fk in other_info.get("foreign_keys", []):
                if fk["referenced_table"] == table_name:
                    return True
        for col in self.tables[table_name]["columns"]:
            low = col.lower()
            if low == "id" or low.endswith("_id"):
                continue
            if self._classify_column(table_name, col) == "text":
                return True
        return False

