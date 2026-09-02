import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import psycopg2
import psycopg2.extras
from psycopg2 import sql

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    pooling = None

try:
    import llm_provider as _llm_provider
except ImportError:
    _llm_provider = None

try:
    import schema_engine.relationships as _se_relationships
except ImportError:
    _se_relationships = None

from core.config import SCHEMA_DIR

logger = logging.getLogger("aria.schema_agent")

JSON_INDENT = 2
K = 2

from .inference import normalize_identifier


# ---- _SCHEMA_REASONING_ROLE (654-654) ----

_SCHEMA_REASONING_ROLE = "schema"

# ---- _SCHEMA_REASONING_PROMPT (656-688) ----

_SCHEMA_REASONING_PROMPT = """\
You are a careful database schema analyst for a BI system. Below is a schema
summary (tables, columns, data types, nullability, declared/inferred primary
keys, and UNVERIFIED candidate relationships). Reason about it and return STRICT
JSON only, with exactly this shape:

{
  "tables": {
    "<table_name>": {
      "description": "one-line plain-English purpose of this table",
      "pk_candidates": ["column names likely to be a primary key, ONLY if the table has none declared or inferred; else []"],
      "measures": ["numeric columns that are business measurements to aggregate (never ids, codes, flags or timestamps)"],
      "dimensions": ["categorical/text columns useful for grouping and filtering"]
    }
  },
  "relationships": [
    {"table": "...", "column": "...", "references_table": "...", "references_column": "...",
     "kind": "confirm", "reason": "short justification"}
  ]
}

Rules:
- Refer ONLY to tables and columns that exist in the schema above. Never invent names.
- "kind": "confirm" for a candidate relationship you believe is a real foreign key;
  "add" for a genuine FK-like link the candidates missed. Skip same-name columns
  that merely coincide (e.g. two unrelated "amount" columns).
- CRITICAL: Look for foreign keys beyond X_id naming. Columns like reports_to,
  ship_via, created_by, parent_id, managed_by, etc. may reference other tables.
  A column referencing the same table (self-reference) is valid (e.g. reports_to
  in employees referencing employee_id).
- Use exact, case-sensitive table and column names.
- When unsure, omit rather than guess. No markdown, no prose, JSON only.
"""

# ---- _SCHEMA_REASONING_PROMPT_TAIL (691-691) ----

_SCHEMA_REASONING_PROMPT_TAIL = "\nSchema:\n"

# ---- _NUMERIC_TYPE_HINTS (851-853) ----

_NUMERIC_TYPE_HINTS = (
    "int", "numeric", "decimal", "float", "real", "double", "serial", "money",
)

# ---- _is_numeric_type (856-858) ----

def _is_numeric_type(data_type):
    t = (data_type or "").lower()
    return any(h in t for h in _NUMERIC_TYPE_HINTS)

# ---- _column_data_type (861-865) ----

def _column_data_type(table_info, col_name):
    for c in table_info.get("columns", []):
        if c["column"] == col_name:
            return c.get("data_type", "")
    return ""

# ---- _build_reasoning_brief (740-811) ----

def _build_reasoning_brief(mapping, max_candidates=40):
    """Focused briefing for the reasoning LLM (PART 16).

    Deterministic candidate generation and evidence collection have already
    happened before this point. The LLM is only asked for SEMANTIC evidence on
    the genuinely ambiguous/high-value candidates plus descriptions / PK
    suggestions for tables that lack a key. This is NOT a lossy whole-schema
    dump: the payload is bounded by the number of candidates under review, so
    the LLM is never treated as an oracle over the entire database. Everything
    it returns is still routed through the deterministic classifier + policy
    (RULE 5).
    """
    tables = mapping.get("tables", {})
    if not tables:
        return ""
    lines = [f"Database: {mapping.get('database', 'unknown')} "
             f"(schema: {mapping.get('schema', '?')})"]

    def _pk(table_name):
        info = tables.get(table_name, {}) or {}
        pk = info.get("primary_key") or info.get("inferred_primary_key") or []
        return ", ".join(pk) if isinstance(pk, (list, tuple)) and pk else "none"

    lines.append("TABLES:")
    for table_name in tables:
        lines.append(f"  {table_name} (PK: {_pk(table_name)})")

    missing_pk = sorted(t for t in tables if _pk(t) == "none")
    declared = mapping.get("declared_relationships", []) or []
    inferred = mapping.get("inferred_relationships", []) or []
    ambiguous = mapping.get("ambiguous_relationships", []) or []
    candidates = inferred + ambiguous

    review_tables = set()
    for r in candidates:
        review_tables.add(r.get("table"))
        review_tables.add(r.get("references_table"))

    detail_tables = sorted((set(missing_pk) | review_tables) & set(tables))
    if detail_tables:
        lines.append("COLUMN CONTEXT (tables under review / lacking a key):")
        for t in detail_tables:
            cols = ", ".join(c["column"] for c in tables[t].get("columns", []))
            lines.append(f"  {t}: {cols}")

    if declared:
        lines.append("DECLARED FOREIGN KEYS (authoritative - do not second-guess):")
        for d in declared:
            lines.append(f"  {d['table_name']}.{d['column_name']} -> "
                         f"{d['references_table']}.{d['references_column']}")

    if candidates:
        lines.append(f"CANDIDATES FOR SEMANTIC REVIEW ({len(candidates)}):")
        for r in candidates[:max_candidates]:
            src = r.get("table", r.get("table_name"))
            col = r.get("column", r.get("column_name"))
            tgt_t = r.get("references_table")
            tgt_c = r.get("references_column")
            extra = ""
            state = r.get("relationship_state")
            if state:
                extra += f" state={state}"
            if r.get("confidence"):
                extra += f" confidence={r.get('confidence')}"
            score = r.get("confidence_score")
            if score is not None:
                extra += f" score={score}"
            lines.append(f"  {src}.{col} -> {tgt_t}.{tgt_c}{extra}")
        if len(candidates) > max_candidates:
            lines.append(f"... ({len(candidates) - max_candidates} more candidates omitted)")

    return "\n".join(lines)

# ---- _parse_schema_reasoning (814-848) ----

def _parse_schema_reasoning(content):
    """Robustly parse the reasoning response into a dict ({} on any failure).

    Falls back to the largest object-bounded prefix when the model's reply is
    truncated mid-JSON (the token cap can be hit before the closing brace), so
    a partial but otherwise-correct answer is not discarded wholesale.
    """
    if not content or not isinstance(content, str):
        return {}
    text = content.strip()
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Truncation salvage: try each `{` boundary before the final `}` and keep
    # the largest prefix that still parses as a dict.
    for open_idx in range(start, end):
        if text[open_idx] != "{":
            continue
        try:
            data = json.loads(text[open_idx:end + 1])
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}

# ---- _apply_schema_reasoning (868-1026) ----

def _apply_schema_reasoning(mapping, parsed, schema=None):
    """Merge validated LLM suggestions into the mapping (never invents names)."""
    tables = mapping.setdefault("tables", {})
    declared = mapping.get("declared_relationships", [])
    declared_pairs = {(d["table_name"], d["column_name"]) for d in declared}
    inferred = mapping.setdefault("inferred_relationships", [])
    edges = mapping.setdefault("relationship_edges", [])

    def pair(r):
        return (r.get("table"), r.get("column"), r.get("references_table"), r.get("references_column"))

    # -- table-level annotations ------------------------------------------
    llm_tables = parsed.get("tables") or {}
    if isinstance(llm_tables, dict):
        for table_name, info in llm_tables.items():
            if table_name not in tables or not isinstance(info, dict):
                continue
            table_info = tables[table_name]
            real_columns = {c["column"] for c in table_info.get("columns", [])}

            desc = info.get("description")
            if isinstance(desc, str) and desc.strip():
                table_info["description"] = desc.strip()

            measures = [c for c in (info.get("measures") or []) if isinstance(c, str)]
            dims = [c for c in (info.get("dimensions") or []) if isinstance(c, str)]
            measures = [c for c in measures if c in real_columns and _is_numeric_type(_column_data_type(table_info, c))]
            dims = [c for c in dims if c in real_columns and c not in measures]
            if measures or dims:
                table_info.setdefault("semantic_tags", {}).update(
                    {"measures": measures, "dimensions": dims}
                )

            existing_pk = table_info.get("primary_key") or table_info.get("inferred_primary_key") or []
            pk_cands = [c for c in (info.get("pk_candidates") or []) if isinstance(c, str) and c in real_columns]
            if not existing_pk and len(pk_cands) == 1:
                cand = pk_cands[0]
                nullable = next((c.get("nullable") for c in table_info.get("columns", []) if c["column"] == cand), True)
                null_pct = table_info.get("null_stats", {}).get(cand, {}).get("null_pct")
                if nullable is False or null_pct in (0, 0.0):
                    table_info["inferred_primary_key"] = [cand]
                    if not table_info.get("primary_key"):
                        table_info["primary_key"] = [cand]

    # -- relationship confirm / add ----------------------------------------
    llm_rels = parsed.get("relationships") or []
    if isinstance(llm_rels, list):
        from schema_engine.classifier import classify
        from schema_engine.policy import AcceptancePolicy

        policy = AcceptancePolicy()
        table_pk = {t: list(info.get("primary_key") or []) for t, info in tables.items()}
        null_stats = {t: (info.get("null_stats") or {}) for t, info in tables.items()}
        review_stash = mapping.setdefault("_llm_review", [])

        for rel in llm_rels:
            if not isinstance(rel, dict):
                continue
            table = rel.get("table")
            column = rel.get("column")
            ref_table = rel.get("references_table")
            ref_col = rel.get("references_column")
            kind = rel.get("kind")
            if not all(isinstance(x, str) and x for x in (table, column, ref_table, ref_col)):
                continue
            if table not in tables or ref_table not in tables:
                continue  # hallucinated table
            if column not in {c["column"] for c in tables[table].get("columns", [])}:
                continue
            if ref_col not in {c["column"] for c in tables[ref_table].get("columns", [])}:
                continue
            # The source column must look like a reference (id-like name, FK
            # pattern like _by/_to/_via, or an exact match to the target column):
            # this stops the LLM from "confirming" a numeric measure (e.g.
            # `total`) as a foreign key.
            col_norm = normalize_identifier(column)
            ref_norm = normalize_identifier(ref_col)
            col_dtype = _column_data_type(tables.get(table, {}), column).lower()
            is_id_like = col_norm.endswith("id")
            is_fk_pattern = column.lower().endswith(("_id", "_by", "_to", "_from", "_via", "_for", "_at", "_with", "_against", "_on", "_of", "_type"))
            is_numeric_type = any(tok in col_dtype for tok in ("int", "numeric", "decimal", "serial"))
            if not (is_id_like or is_fk_pattern or col_norm == ref_norm or is_numeric_type):
                continue
            if (table, column) in declared_pairs:
                continue  # already constrained; don't second-guess the DB

            new_pair = (table, column, ref_table, ref_col)
            if kind == "confirm":
                existing = [r for r in inferred
                            if r.get("table") == table and r.get("column") == column]
                if existing:
                    # Only upgrade confidence if LLM agrees with the existing
                    # heuristic target. Never replace a heuristic target with
                    # the LLM's guess — small LLMs often pick wrong targets.
                    for r in existing:
                        if (r.get("references_table") == ref_table
                                and r.get("references_column") == ref_col):
                            r["confidence"] = "llm-confirmed"
                            r["note"] = ("Confirmed by LLM reasoning "
                                         "(no FK constraint declared in the database).")
                            policy.corroborate(r, 20, "LLM reasoning corroborates this target")
                            for e in edges:
                                if (e.get("source_table") == table
                                        and e.get("source_column") == column
                                        and e.get("target_table") == ref_table):
                                    e["confidence"] = "llm-confirmed"
                                    e["confidence_score"] = r.get("confidence_score")
                                    e["confidence_band"] = r.get("confidence_band")
                                    e["relationship_state"] = r.get("relationship_state")
                                    e["review_status"] = r.get("review_status")
                else:
                    # No existing heuristic — LLM is the only signal; the
                    # policy decides whether it is strong enough to emit.
                    new_rel, _amb = classify(
                        None, table_pk, null_stats, schema,
                        table, column, ref_table, ref_col,
                        "llm-confirmed",
                        "Confirmed by LLM reasoning (no FK constraint declared in the database).",
                        name_hint=False, llm_signal="confirm",
                    )
                    if new_rel["relationship_state"] == "UNCERTAIN":
                        new_rel["review_status"] = "review"
                        review_stash.append(new_rel)
                    else:
                        inferred.append(new_rel)
                        edges.append({
                            "source_table": table, "source_column": column,
                            "target_table": ref_table, "target_column": ref_col,
                            "type": "inferred", "confidence": "llm-confirmed",
                            "confidence_score": new_rel.get("confidence_score"),
                            "confidence_band": new_rel.get("confidence_band"),
                            "relationship_state": new_rel.get("relationship_state"),
                            "review_status": new_rel.get("review_status"),
                        })
            elif kind == "add" and new_pair not in {pair(r) for r in inferred}:
                # Only add when this column has no inferred FK yet (no ambiguity);
                # the policy decides the state from LLM + name + PK evidence.
                if not any(r.get("table") == table and r.get("column") == column for r in inferred):
                    new_rel, _amb = classify(
                        None, table_pk, null_stats, schema,
                        table, column, ref_table, ref_col,
                        "llm-reasoned",
                        "Proposed by LLM reasoning (no FK constraint declared in the database).",
                        name_hint=False, llm_signal="add",
                    )
                    if new_rel["relationship_state"] == "UNCERTAIN":
                        new_rel["review_status"] = "review"
                        review_stash.append(new_rel)
                    else:
                        inferred.append(new_rel)
                        edges.append({
                            "source_table": table, "source_column": column,
                            "target_table": ref_table, "target_column": ref_col,
                            "type": "inferred", "confidence": "llm-reasoned",
                            "confidence_score": new_rel.get("confidence_score"),
                            "confidence_band": new_rel.get("confidence_band"),
                            "relationship_state": new_rel.get("relationship_state"),
                            "review_status": new_rel.get("review_status"),
                        })

# ---- enrich_with_llm (694-737) ----

def enrich_with_llm(mapping, llm, schema=None):
    """Optional LLM-assisted reasoning pass over a built schema mapping.

    Works identically for every provider (Ollama 'local' or Groq 'cloud').
    Returns the mapping unchanged when no LLM is given, the call fails/times
    out, or the response is unusable.
    """
    if llm is None:
        return mapping

    summary = _build_reasoning_brief(mapping)
    if not summary.strip():
        return mapping

    try:
        timeout = 300 if getattr(llm, "provider", None) == "local" else 60
        content = llm.chat(
            _SCHEMA_REASONING_ROLE,
            messages=[{"role": "user", "content": _SCHEMA_REASONING_PROMPT + _SCHEMA_REASONING_PROMPT_TAIL + summary}],
            temperature=0.0,
            num_predict=2500,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"WARNING: schema LLM reasoning skipped ({exc}); keeping heuristic results.")
        return mapping

    parsed = _parse_schema_reasoning(content)
    if not parsed or not (isinstance(parsed.get("tables"), dict)
                          or isinstance(parsed.get("relationships"), list)):
        print("WARNING: schema LLM reasoning returned no usable JSON; keeping heuristic results.")
        return mapping

    try:
        model = llm.model_for(_SCHEMA_REASONING_ROLE)
    except Exception:
        model = None
    mapping["llm_reasoning"] = {
        "provider": getattr(llm, "provider", None),
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _apply_schema_reasoning(mapping, parsed, schema=schema)
    return mapping
