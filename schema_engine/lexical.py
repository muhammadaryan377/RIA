"""Lexical/name helpers (database-independent relational principles, PART 17).

These implement general naming-convention reasoning: identifier
normalization, tokenization, singular/plural handling and "is this an
identifier-flavored name" detection. They contain no database-specific lists
(AdventureWorks/Northwind/etc.) — only general English convention rules.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

# Tokens that signal "this word is about identifiers/keys", never content words.
ID_TOKENS = {"id", "no", "num", "number", "code", "key", "fk", "ref", "uuid", "sk", "nk"}

# Suffixes that conventionally mark a column as referencing another entity.
# General relational conventions (PART 4 / PART 17): "_id" plus the common
# `_ref` / `_key` / `_no` / `_num` families used across arbitrary schemas. The
# `_code` family is deliberately excluded: it is ambiguous (status/country/
# currency codes collide with PKs by coincidence) and is handled by the
# code-like evidence signal + policy veto instead.
REF_FLAVORED_SUFFIXES = ("_id", "_by", "_to", "_from", "_via", "_for", "_at",
                         "_with", "_against", "_on", "_of", "_type",
                         "_ref", "_key", "_no", "_num", "_nbr",
                         "_uuid", "_sk", "_nk")


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def singularize(name: str) -> str:
    """Best-effort singular of an English identifier."""
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        return name[:-1]
    return name


def is_ref_flavored(name) -> bool:
    """Expanded FK-name signal: catches _id, _by, _to, _from, _via, _for, etc."""
    if not isinstance(name, str):
        return False
    low = name.lower()
    if low == "id":
        return True
    for suffix in REF_FLAVORED_SUFFIXES:
        if low.endswith(suffix):
            return True
    if low.endswith("id") and len(low) > 2:
        return True
    return False


def tokenize_words(name: str) -> List[str]:
    """Split an identifier into lowercase word tokens (snake_case, camelCase...).

    "MediaTypeId" -> ["media", "type", "id"]; "support_rep_id" -> ["support", "rep", "id"].
    This is what lets name matching work on ANY naming convention, not just X_id.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if w]


def table_name_forms(base: str) -> Set[str]:
    """Singular/plural forms a table name could take for `base`."""
    base = singularize(base)
    forms = {base, base + "s", base + "es"}
    if base.endswith("y") and len(base) > 1:
        forms.add(base[:-1] + "ies")
    forms.add(singularize(base))
    return forms


def table_name_candidates(col_name: str, tables: Iterable[str]) -> List[str]:
    """For 'X_id' (or 'Xid'), return tables whose name (singular or plural) is X.

    Generalized token match: any naming style, not just X_id (PART 4 lexical
    signal). A bare trailing "id" with no separator (`storeid`, `personid`) is
    stripped so it matches the same way `store_id` does. Pure candidate
    generation - it never accepts a relationship.
    """
    base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
    forms = table_name_forms(base)
    names = [t for t in tables if str(t).lower() in forms]
    if not names:
        strip_id = base[:-2] if base.endswith("id") and len(base) > 3 else base
        col_tokens = {singularize(tok) for tok in tokenize_words(strip_id)
                      if tok not in ID_TOKENS}
        if col_tokens:
            for t in tables:
                # Table-name tokens only: the schema prefix is context, not a
                # name-match signal (PART 18), so `personid` must not match a
                # table merely because its schema is named "person".
                t_part = str(t).rsplit(".", 1)[-1]
                t_tokens = {singularize(tok) for tok in tokenize_words(t_part)}
                if col_tokens <= t_tokens:
                    names.append(t)
    return sorted(set(names))


def most_specific_table(base_tokens: Set[str], tables: Iterable[str]) -> str | None:
    """Shortest table whose name tokens intersect `base_tokens` (name specificity).

    `orders` outranks `olist_order_items_dataset` because it is a shorter,
    more specific name for the token `order`.
    """
    best = None
    for t in tables:
        name_part = str(t).rsplit(".", 1)[-1]
        table_tokens = {singularize(tok) for tok in tokenize_words(name_part)}
        if not table_tokens:
            continue
        if table_tokens & base_tokens:
            if best is None or len(str(t)) < len(str(best)):
                best = t
    return best


def is_reference_to_other_table(col_name: str, table_name: str, tables: Iterable[str]) -> bool:
    """True when `col_name` (X_id style) names ANOTHER table (X/Xs/Xes/Xies).

    A column like `order_id` in a table that also has an `orders` table is
    almost certainly a foreign key, NOT the table's own primary key.
    """
    base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
    if not base or base in ID_TOKENS:
        return False
    forms = table_name_forms(base)

    def _matches(t) -> bool:
        if normalize_identifier(t) in forms:
            return True
        return any(singularize(tok) in forms for tok in tokenize_words(t))

    best = None
    for t in tables:
        if not _matches(t):
            continue
        if best is None or len(str(t)) < len(str(best)):
            best = t
    return best is not None and best != table_name


def is_code_column(col_name: str) -> bool:
    """True for categorical label/code columns that collide with PKs by chance.

    `status`, `country_code`, `category`, `year`, `department` etc. often share
    values with primary keys (1..n) and must be penalized heavily as FK
    candidates — they are almost never real references.

    NOTE: this is a general *semantic* signal (code-like words). The word list
    will be converted into a general evidence signal in Phase E.
    """
    low = col_name.lower()
    base = low[:-3] if low.endswith("_id") else low
    tokens = set(tokenize_words(base))
    code_words = {
        "status", "category", "subcategory", "country", "city", "state",
        "region", "year", "month", "day", "week", "department", "gender",
        "channel", "segment", "tier", "type", "flag", "currency", "language",
        "color", "size", "brand", "name", "label", "grade", "level",
    }
    return any(t in code_words for t in tokens)