"""Industry-level additive Insight Agent for ARIA.

This is the highest capability layer. It does not modify ``insight_agent.py``
or change the current API wiring. It extends ``AdvancedInsightAgent`` and then
adds goal-aware decision intelligence on top of the already generated output.

Existing output remains intact; new decision fields are appended.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from insight_agent import sanitize_json
from insight_agent_advanced import AdvancedInsightAgent
from insight_extensions import DecisionIntelligence

log = logging.getLogger("aria.insight.industry")


class IndustryInsightAgent(AdvancedInsightAgent):
    """ARIA Insight Agent with evidence-backed decision intelligence."""

    VERSION = "3.0"

    def __init__(self, provider=None):
        super().__init__(provider=provider)
        self.decision_engine = DecisionIntelligence()

    def analyze(self, processed_data_path="processed_data.json", output_path="insights.json"):
        # First run every working capability from the additive advanced layer.
        super().analyze(processed_data_path, output_path)

        output_path = Path(output_path)
        try:
            insights = json.loads(output_path.read_text(encoding="utf-8"))
            processed = self.load_processed_data(processed_data_path)
            df = pd.DataFrame(processed.get("data") or [])

            numeric_cols, category_cols, datetime_cols = self._classify_columns(df)
            df = self._coerce_columns(df, numeric_cols, datetime_cols)

            decision = self.decision_engine.analyze(
                df,
                processed,
                numeric_cols,
                category_cols,
                datetime_cols,
                hypotheses=insights.get("hypotheses") or [],
                anomalies=insights.get("anomalies") or [],
                trends=insights.get("trends") or [],
                drivers=insights.get("drivers") or {},
                segments=insights.get("segments") or [],
                encoded_prediction=insights.get("encoded_prediction"),
            )

            insights["analysis_version"] = self.VERSION
            insights["decision_intelligence"] = decision

            # Expose the most useful decision fields at top-level as well so the
            # frontend can consume them without knowing the nested extension.
            for key in (
                "goal_profile",
                "period_comparison",
                "contribution_analysis",
                "root_cause_drilldown",
                "retail_signals",
                "business_impact",
                "ranked_insights",
                "evidence_model",
            ):
                insights[key] = decision.get(key)

            insights["decision_brief"] = self._decision_brief(decision, insights)
            output_path.write_text(
                json.dumps(sanitize_json(insights), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            # Never sacrifice the already-created advanced insight output because
            # one extension failed. This preserves the project's fail-soft design.
            log.exception("Decision intelligence extension failed: %s", exc)
            try:
                insights = json.loads(output_path.read_text(encoding="utf-8"))
                prior_warning = insights.get("warning")
                extension_warning = f"decision_intelligence: {exc}"
                insights["warning"] = (
                    f"{prior_warning}; {extension_warning}" if prior_warning else extension_warning
                )
                insights["decision_intelligence"] = {
                    "version": DecisionIntelligence.VERSION,
                    "warning": str(exc),
                    "ranked_insights": [],
                }
                output_path.write_text(
                    json.dumps(sanitize_json(insights), indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

        print(f"Industry Insight Agent done! Saved to {output_path}")
        return str(output_path)

    @staticmethod
    def _decision_brief(decision, insights):
        ranked = decision.get("ranked_insights") or []
        evidence = decision.get("evidence_model") or {}
        impact = decision.get("business_impact")
        confidence = insights.get("insight_confidence") or {}

        headline = (
            ranked[0].get("statement")
            if ranked
            else (insights.get("executive_summary") or {}).get("headline")
        )
        return {
            "headline": headline,
            "confidence": confidence,
            "top_insights": ranked[:5],
            "business_impact": impact,
            "facts": (evidence.get("facts") or [])[:5],
            "associations": (evidence.get("associations") or [])[:3],
            "hypotheses": (evidence.get("hypotheses") or [])[:3],
            "next_checks": (evidence.get("next_checks") or [])[:4],
        }


# Convenient alias if/when this layer is explicitly selected for application use.
InsightAgent = IndustryInsightAgent


__all__ = ["IndustryInsightAgent", "InsightAgent", "sanitize_json"]
