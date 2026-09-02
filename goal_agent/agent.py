"""Goal Agent orchestration: builds the engine, runs the linear / LangGraph pipelines and composes all components."""

import difflib
import json
import os
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None


from .constants import ConstantsMixin
from .semantics import SemanticMixin
from .intent import IntentMixin
from .schema import SchemaMixin
from .grounding import GroundingMixin
from .sql import SqlMixin
from .sql_contract import SqlContractMixin
from .suggestions import SuggestionsMixin
from .data import DataMixin

DEFAULT_DB_URI = (
    "postgresql://{user}:{password}@{host}:{port}/{dbname}"
).format(
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "postgres"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "postgres"),
)

class GoalAgent(ConstantsMixin, SemanticMixin, IntentMixin, SchemaMixin,
                GroundingMixin, SqlMixin, SqlContractMixin, SuggestionsMixin,
                DataMixin):
    # ---- __init__ (__init__) ----

    def __init__(self, schema_json_path=None, db_uri=None,
                 provider=None, dialect="postgresql"):
        if schema_json_path:
            schema_file = Path(schema_json_path)
        else:
            # Absolute project-relative default so the process CWD never matters.
            schema_file = SCHEMA_DIR / "schema_mapping_latest.json"
        if not schema_file.exists():
            # Absolute project-relative fallback so the process CWD never matters.
            fallback = BASE_DIR / "schema_mapping.json"
            if fallback.exists():
                schema_file = fallback
        if not schema_file.exists():
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise FileNotFoundError(
                f"Schema mapping not found: {schema_file}. "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Run the schema extraction first (python schema_agent.py <database>).")
            )
        with open(schema_file, "r", encoding="utf-8") as f:
            self.full_schema = json.load(f)

        self.tables = self._normalize_schema(self.full_schema)
        self.uncertain_edge_keys = self._collect_uncertain_edges(self.full_schema)
        self.relationship_graph = self._build_relationship_graph()
        self.relationship_edges = self._build_relationship_edges()
        self.dialect = dialect
        self.schema = (self.full_schema.get("schema") or "public").strip()
        self.engine = self._build_engine(db_uri)
        self.llm = provider or create_provider()
        self.kpi_index = self._build_kpi_index()
        self._row_counts_cache = None
        self.preprocessing_report = None
        self._last_execution_ms = None
        self._last_semantic_warnings = []
        self._schema_word_tokens = self._build_schema_word_tokens()
        self._active_original_goal = None
        self._typo_warnings = []

    # ---- _build_engine (_build_engine) ----

    def _build_engine(self, db_uri):
        """Build the SQLAlchemy engine. For non-public PostgreSQL schemas the
        search_path is applied via connect_args (survives pool resets), so bare
        table names from the schema mapping resolve correctly."""
        uri = db_uri or DEFAULT_DB_URI
        if self.dialect == "postgresql" and self.schema and self.schema != "public":
            return create_engine(uri, connect_args={"options": f"-csearch_path={self.schema}"})
        return create_engine(uri)

    # ---- process_goal (process_goal) ----

    def process_goal(self, user_goal, output_path="processed_data.json"):
        # The contract writers below target `output_path` (e.g. a per-user path
        # under artifacts/processed_data/). Create the parent directory so the
        # API route never crashes on a missing folder for a new user.
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            corrected_goal, corrections = self._correct_goal(user_goal)
            self._active_original_goal = user_goal
            self._typo_warnings = [
                f"Did you mean '{suggestion}' for '{typo}' in your question?"
                for typo, suggestion in corrections
            ]
            entity = self._detect_metrics_overview(corrected_goal)
            if entity is not None:
                logger.info(
                    "Open-ended metrics goal -> deterministic overview for '%s'.",
                    entity,
                )
                return self._process_metrics_overview(
                    corrected_goal, entity, output_path
                )
            needs, question = self._needs_clarification(corrected_goal)
            if needs:
                if corrections:
                    suggestions = ", ".join(f"'{suggestion}'" for _, suggestion in corrections)
                    question += f" Did you mean: {suggestions}?"
                logger.info("Goal is ambiguous; returning needs_clarification.")
                return self._clarify(user_goal, question, output_path)
            if LANGGRAPH_AVAILABLE and StateGraph is not None and END is not None:
                return self._process_goal_langgraph(corrected_goal, output_path)
            return self._process_goal_linear(corrected_goal, output_path)
        except Exception as exc:
            if isinstance(exc, (MemoryError, RecursionError, OverflowError)):
                logger.critical("Fatal exception in process_goal: %s", exc, exc_info=True)
                raise
            return self._graceful_failure(user_goal, output_path, exc)

    # ---- _process_goal_linear (_process_goal_linear) ----

    def _process_goal_linear(self, user_goal, output_path="processed_data.json"):
        logger.info("Goal Agent: '%s'", user_goal)

        kpi_map = self.map_goal_to_kpi(user_goal)
        logger.info("Aligned KPIs: %s | dimensions: %s", kpi_map['kpis'], kpi_map['dimensions'])

        ir = self._build_goal_ir(user_goal)
        relevant_tables = self._get_relevant_tables(user_goal)
        join_path = self._determine_join_path(relevant_tables)
        ir = self._resolve_semantics(user_goal, ir, join_path)
        kpi_map = self._enrich_kpi_map(kpi_map, ir)
        join_plan = self._plan_joins(self._required_tables_from_ir(ir, join_path))
        if join_plan["edges"]:
            join_path = join_plan["nodes"]
        ir["join_plan"] = join_plan["edges"]
        logger.info("Join path: %s", join_path)

        raw_sql = self._generate_sql(user_goal, join_path, kpi_map, ir=ir, join_plan=join_plan)
        logger.info("Generated SQL: %s", raw_sql)

        data_rows, final_sql, note = self._execute_with_retry(
            user_goal, join_path, kpi_map, raw_sql, ir=ir, join_plan=join_plan
        )
        cleaned_data = self._clean_data(data_rows)

        failed = bool(note and note.get("failed"))
        output = {
            "user_goal": self._active_original_goal or user_goal,
            "kpi_alignment": kpi_map,
            "join_path": join_path,
            "sql_used": final_sql,
            "row_count": len(cleaned_data),
            "data": cleaned_data,
            "message": (note or {}).get("message") if note else None,
            "preprocessing": self.preprocessing_report,
            "missing_values_handled": "preserved (NULLs kept as-is)",
            "timestamp": datetime.now().isoformat(),
        }
        output.update(self._build_output_contract(
            user_goal, kpi_map, join_path, final_sql, cleaned_data,
            success=(not failed), message=output["message"],
            status="query_failed" if failed else "success",
            warnings=self._typo_warnings + self._warnings_for_sql(final_sql)
                    + self._last_semantic_warnings,
            execution_time_ms=self._last_execution_ms,
        ))

        out_path = Path(output_path)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        logger.info("Goal Agent done! Saved to %s", output_path)
        return output_path

    # ---- _process_goal_langgraph (_process_goal_langgraph) ----

    def _process_goal_langgraph(self, user_goal, output_path):
        def load_schema(state):
            state["kpi_map"] = self.map_goal_to_kpi(state["user_goal"])
            state["relevant_tables"] = self._get_relevant_tables(state["user_goal"])
            state["join_path"] = self._determine_join_path(state["relevant_tables"])
            return state

        def understand_goal(state):
            state["goal_ir"] = self._build_goal_ir(state["user_goal"])
            return state

        def resolve_semantics(state):
            state["goal_ir"] = self._resolve_semantics(
                state["user_goal"], state["goal_ir"], state["join_path"]
            )
            state["kpi_map"] = self._enrich_kpi_map(state["kpi_map"], state["goal_ir"])
            return state

        def plan_joins(state):
            state["join_plan"] = self._plan_joins(
                self._required_tables_from_ir(state.get("goal_ir"), state["join_path"])
            )
            if state["join_plan"]["edges"]:
                state["join_path"] = state["join_plan"]["nodes"]
            if state.get("goal_ir"):
                state["goal_ir"]["join_plan"] = state["join_plan"]["edges"]
            return state

        def generate_sql(state):
            state["raw_sql"] = self._generate_sql(
                state["user_goal"], state["join_path"], state["kpi_map"],
                ir=state.get("goal_ir"), join_plan=state.get("join_plan"),
            )
            return state

        def execute_sql(state):
            rows, final_sql, note = self._execute_with_retry(
                state["user_goal"], state["join_path"], state["kpi_map"],
                state["raw_sql"],
                ir=state.get("goal_ir"), join_plan=state.get("join_plan"),
            )
            state["rows"] = self._clean_data(rows)
            state["final_sql"] = final_sql
            state["_failed"] = bool(note and note.get("failed"))
            state["message"] = (note or {}).get("message") if note else None
            state["preprocessing"] = self.preprocessing_report
            return state

        def finalize(state):
            failed = bool(state.get("message") and state.get("_failed"))
            output = {
                "user_goal": self._active_original_goal or state["user_goal"],
                "kpi_alignment": state["kpi_map"],
                "join_path": state["join_path"],
                "goal_ir": state.get("goal_ir"),
                "sql_used": state["final_sql"],
                "row_count": len(state["rows"]),
                "data": state["rows"],
                "message": state.get("message"),
                "preprocessing": state.get("preprocessing"),
                "missing_values_handled": "preserved (NULLs kept as-is)",
                "timestamp": datetime.now().isoformat(),
            }
            output.update(self._build_output_contract(
                state["user_goal"], state["kpi_map"], state["join_path"],
                state["final_sql"], state["rows"],
                success=(not failed),
                message=output["message"],
                status="query_failed" if failed else "success",
                warnings=self._typo_warnings + self._warnings_for_sql(state["final_sql"])
                + self._last_semantic_warnings,
                execution_time_ms=self._last_execution_ms,
            ))
            Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
            state["result_path"] = output_path
            return state

        workflow = StateGraph(dict)
        workflow.add_node("load_schema", load_schema)
        workflow.add_node("understand_goal", understand_goal)
        workflow.add_node("resolve_semantics", resolve_semantics)
        workflow.add_node("plan_joins", plan_joins)
        workflow.add_node("generate_sql", generate_sql)
        workflow.add_node("execute_sql", execute_sql)
        workflow.add_node("finalize", finalize)
        workflow.set_entry_point("load_schema")
        workflow.add_edge("load_schema", "understand_goal")
        workflow.add_edge("understand_goal", "resolve_semantics")
        workflow.add_edge("resolve_semantics", "plan_joins")
        workflow.add_edge("plan_joins", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "finalize")
        workflow.add_edge("finalize", END)

        app = workflow.compile()
        result = app.invoke({
            "user_goal": user_goal, "kpi_map": {}, "relevant_tables": [],
            "join_path": [], "raw_sql": "", "rows": [], "final_sql": "",
        })
        logger.info("Goal Agent done! Saved to %s", result['result_path'])
        return result["result_path"]

