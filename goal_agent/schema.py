"""Component: schema normalization and relationship graph (confidence-annotated edges) plus join planning (spec sections 5, 10)."""

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

class SchemaMixin:
    # ---- _normalize_schema (_normalize_schema) ----

    def _normalize_schema(self, raw_schema):
        raw_tables = raw_schema.get("tables", {})
        if not isinstance(raw_tables, dict) or not raw_tables:
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise ValueError(
                "Schema mapping has no tables (stale or hand-edited JSON?). "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Re-run the schema extraction for this database.")
            )

        tables = {}
        all_edges = raw_schema.get("relationship_edges", [])
        declared = raw_schema.get("declared_relationships", [])
        inferred = raw_schema.get("inferred_relationships", [])
        known_tables = set(raw_tables)

        def add_fk(table_name, source_column, target_table, target_column, rel_type, confidence=None):
            tables.setdefault(table_name, {
                "columns": {}, "primary_key": [], "foreign_keys": [],
            })
            tables[table_name]["foreign_keys"].append({
                "column": source_column,
                "referenced_table": target_table,
                "referenced_column": target_column,
                "type": rel_type,
                "confidence": confidence,
            })

        for table_name, info in raw_schema.get("tables", {}).items():
            columns = info.get("columns", [])
            col_map = {}
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict):
                        name = col.get("column")
                        col_map[name] = {
                            "data_type": col.get("data_type", "TEXT"),
                            "nullable": col.get("nullable", True),
                        }
                    else:
                        col_map[str(col)] = {"data_type": "TEXT", "nullable": True}
            else:
                col_map = columns

            pk = info.get("primary_key", []) or info.get("inferred_primary_key", []) or []
            for key in pk:
                if key in col_map:
                    col_map[key]["is_primary_key"] = True

            tables[table_name] = {
                "columns": col_map, "primary_key": pk, "foreign_keys": [],
                "empty": bool(info.get("empty")),
                "row_count": info.get("row_count"),
            }

        for edge in all_edges:
            add_fk(
                edge.get("source_table"), edge.get("source_column"),
                edge.get("target_table"), edge.get("target_column"),
                edge.get("type", "declared"), edge.get("confidence"),
            )
        for rel in declared:
            add_fk(rel.get("table_name"), rel.get("column_name"),
                   rel.get("references_table"), rel.get("references_column"),
                   "declared", None)
        for rel in inferred:
            conf = rel.get("confidence")
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                # Schema Agent can emit non-numeric confidence labels such as
                # "llm-confirmed" / "llm-reasoned"; the numeric confidence_score
                # (0-100) is the real measure. Prefer it, never crash on labels.
                conf = rel.get("confidence_score")
                if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                    conf = None
            add_fk(rel.get("table"), rel.get("column"),
                   rel.get("references_table"), rel.get("references_column"),
                   "inferred", conf)

        for table_name, info in tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk["referenced_table"]
                if target and target not in tables:
                    logger.warning(
                        "Schema mapping references unknown table '%s' from %s.%s "
                        "(stale/manually-edited JSON?)",
                        target, table_name, fk.get("column"),
                    )
                elif target and fk.get("referenced_column") not in tables[target]["columns"]:
                    logger.warning(
                        "Schema mapping references unknown column '%s.%s' from %s.%s",
                        target, fk.get("referenced_column"), table_name, fk.get("column"),
                    )

        return tables

    # ---- _build_relationship_graph (_build_relationship_graph) ----

    def _build_relationship_graph(self):
        """Bidirectional relationship graph (spec §5): for every FK edge
        A.column -> B.column the graph permits traversal A->B AND B->A. The
        authoritative FK direction is preserved in `relationship_edges`; this
        dict only supports traversal in both directions."""
        graph = defaultdict(list)
        for table_name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk["referenced_table"]
                if target in self.tables:
                    graph[table_name].append(target)
                    graph[target].append(table_name)
        return graph

    # ---- _relationship_trust (_relationship_trust) ----

    def _relationship_trust(self, rel_type, confidence):
        """Explicit relationship trust level (spec §6).

        DECLARED                     -> safe for deterministic join planning
        HIGH_CONFIDENCE_INFERRED     -> may be used when required
        MEDIUM_CONFIDENCE_INFERRED   -> candidate only / needs stronger evidence
        LOW_CONFIDENCE_INFERRED      -> never joined automatically
        """
        if rel_type == "declared":
            return "DECLARED"
        try:
            conf = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            # Non-numeric confidence labels ("llm-confirmed", "llm-reasoned"):
            # never crash on schema output. A confirmation label is treated as
            # high confidence; everything else is treated conservatively.
            label = str(confidence).lower()
            if "confirm" in label or "high" in label:
                conf = 1.0
            elif "reason" in label or "medium" in label:
                conf = 0.7
            else:
                conf = 0.0
        if conf > 1.0:
            conf = conf / 100.0
        if conf >= 0.8:
            return "HIGH_CONFIDENCE_INFERRED"
        if conf >= 0.6:
            return "MEDIUM_CONFIDENCE_INFERRED"
        return "LOW_CONFIDENCE_INFERRED"

    # ---- _build_relationship_edges (_build_relationship_edges) ----

    def _build_relationship_edges(self):
        """Every FK edge with full metadata (spec §5): both endpoints, the
        original FK direction, relationship type, confidence and trust level."""
        edges = []
        for table_name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk["referenced_table"]
                if target not in self.tables:
                    continue
                rel_type = fk.get("type", "declared")
                conf = fk.get("confidence")
                cardinality = None
                right_info = self.tables.get(target)
                if right_info:
                    is_unique = (
                        fk["referenced_column"] in right_info.get("primary_key", [])
                    )
                    cardinality = "many_to_one" if is_unique else "many_to_many"
                edges.append({
                    "left_table": table_name,
                    "left_column": fk["column"],
                    "right_table": target,
                    "right_column": fk["referenced_column"],
                    "relationship_type": rel_type,
                    "confidence": conf,
                    "trust": self._relationship_trust(rel_type, conf),
                    "cardinality": cardinality,
                })
        return edges

    # ---- _collect_uncertain_edges (_collect_uncertain_edges) ----

    def _collect_uncertain_edges(self, raw_schema):
        """Collect edges whose Schema Agent state is UNCERTAIN/REVIEW. These are
        never silently used: when a query joins through one, a warning is added
        to the goal contract (spec §5). Ambiguous relationships are all
        uncertain by definition."""
        keys = set()
        for rel in raw_schema.get("ambiguous_relationships", []) or []:
            keys.add(
                f"{rel.get('table')}.{rel.get('column')}"
                f"->{rel.get('references_table')}.{rel.get('references_column')}"
            )
        for rel in raw_schema.get("inferred_relationships", []) or []:
            state = (rel.get("relationship_state") or "").upper()
            if state in self.UNCERTAIN_STATES:
                keys.add(
                    f"{rel.get('table')}.{rel.get('column')}"
                    f"->{rel.get('references_table')}.{rel.get('references_column')}"
                )
        for edge in raw_schema.get("relationship_edges", []) or []:
            if edge.get("type") != "inferred":
                continue
            state = (edge.get("relationship_state") or "").upper()
            if state in self.UNCERTAIN_STATES:
                keys.add(
                    f"{edge.get('source_table')}.{edge.get('source_column')}"
                    f"->{edge.get('target_table')}.{edge.get('target_column')}"
                )
        return keys

    # ---- _build_schema_ddl (_build_schema_ddl) ----

    def _build_schema_ddl(self, join_path, full=False):
        """Build DDL for the tables in the join path.

        When full=True the DDL also includes every other table in the schema
        (FK-connected first), so the LLM can discover the metric tables it
        needs even when the keyword scoring missed them. The join-path tables
        are always listed first and marked as preferred.
        """
        relevant = [t for t in join_path if t in self.tables]
        logger.info("Relevant tables (join path): %s", relevant)

        tables = list(relevant)
        if full:
            remaining = [t for t in self.tables if t not in relevant]
            # Prefer tables that connect to the join path via foreign keys.
            connected = set()
            for t in relevant:
                for nb in self.relationship_graph.get(t, []):
                    if nb in remaining:
                        connected.add(nb)
            tables += [t for t in remaining if t in connected]
            tables += [t for t in remaining if t not in connected]

        ddl_parts = []
        for table_name in tables:
            info = self.tables[table_name]
            columns = info.get("columns", {})
            col_defs = []
            for column_name, col_info in columns.items():
                data_type = col_info.get("data_type", "TEXT")
                is_pk = column_name in info.get("primary_key", []) or col_info.get("is_primary_key")
                suffix = " PRIMARY KEY" if is_pk else ""
                col_defs.append(f'    "{column_name}" {data_type}{suffix}')
            fk_lines = []
            for fk in info.get("foreign_keys", []):
                fk_lines.append(
                    f'    FOREIGN KEY ("{fk["column"]}") REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}")'
                )
            all_defs = ",\n".join(col_defs + fk_lines)
            ddl_parts.append(f'CREATE TABLE "{table_name}" (\n{all_defs}\n);')
        return "\n\n".join(ddl_parts)

    # ---- _fk_edges (_fk_edges) ----

    def _fk_edges(self):
        if getattr(self, "_fk_edges_cache", None) is None:
            edges = []
            for table_name, info in self.tables.items():
                for fk in info.get("foreign_keys", []):
                    edges.append((
                        table_name,
                        fk["column"],
                        fk["referenced_table"],
                        fk["referenced_column"],
                    ))
            self._fk_edges_cache = edges
        return self._fk_edges_cache

    # ---- _required_tables_from_ir (_required_tables_from_ir) ----

    def _required_tables_from_ir(self, ir, join_path):
        """The tables the goal actually needs (spec §8): resolved measure +
        dimension tables. Join planning and semantic validation run over this
        set only, so neighbor-expanded context tables never become mandatory
        joins."""
        if not ir:
            return list(join_path)
        required = set()
        for m in ir.get("metrics", []):
            if m.get("resolved_table"):
                required.add(m["resolved_table"])
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                required.add(d["resolved_table"])
        return sorted(required) if required else list(join_path)

    # ---- _plan_joins (_plan_joins) ----

    def _plan_joins(self, required_tables):
        """Explicit deterministic join plan (spec §7/§8): the minimum connected
        subgraph of the relationship graph that covers all required tables.

        Returns {"required_tables", "nodes", "edges"} where every edge carries
        table/column endpoints, relationship type and trust level. Declared
        relationships are preferred; LOW_CONFIDENCE_INFERRED edges are only
        used when there is no other way to connect a required table.
        """
        req = [t for t in required_tables if t in self.tables]
        if not req:
            return {"required_tables": [], "nodes": [], "edges": []}

        trust_rank = {"DECLARED": 0, "HIGH_CONFIDENCE_INFERRED": 1,
                      "MEDIUM_CONFIDENCE_INFERRED": 2, "LOW_CONFIDENCE_INFERRED": 3}
        adj = defaultdict(list)
        for e in self.relationship_edges:
            rank = trust_rank.get(e.get("trust"), 4)
            adj[e["left_table"]].append((rank, e))
            adj[e["right_table"]].append((rank, e))
        for table in adj:
            adj[table].sort(key=lambda pair: pair[0])

        connected = {req[0]}
        nodes = [req[0]]
        edges = []
        remaining = [t for t in req[1:] if t != req[0]]

        while remaining:
            target, path_tables, path_edges = None, None, None
            for t in remaining:
                found = self._bfs_join_path(adj, connected, t)
                if found is not None:
                    path_tables, path_edges = found
                    target = t
                    break
            if target is None:
                for t in remaining:
                    if t not in nodes:
                        nodes.append(t)
                break
            for e in path_edges:
                if e not in edges:
                    edges.append(e)
            for n in path_tables:
                if n not in connected:
                    connected.add(n)
                    nodes.append(n)
            remaining = [t for t in remaining if t != target]

        return {"required_tables": req, "nodes": nodes, "edges": edges}

    # ---- _bfs_join_path (_bfs_join_path) ----

    def _bfs_join_path(self, adj, connected, target):
        """Shortest BFS path (by hop count) from the connected component to
        `target`, returning (tables_on_path, edges_on_path) or None."""
        queue = [[n] for n in connected]
        seen = set(connected)
        while queue:
            path = queue.pop(0)
            tail = path[-1]
            for _rank, e in adj.get(tail, []):
                nb = e["right_table"] if e["left_table"] == tail else e["left_table"]
                if nb in seen:
                    continue
                new_path = path + [nb]
                if nb == target:
                    path_tables = new_path
                    path_edges = []
                    for a, b in zip(path_tables, path_tables[1:]):
                        for _r, edge in adj[a]:
                            if edge["left_table"] == b or edge["right_table"] == b:
                                path_edges.append(edge)
                                break
                    return path_tables, path_edges
                seen.add(nb)
                queue.append(new_path)
        return None

