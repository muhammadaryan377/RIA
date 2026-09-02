"""Advanced ARIA Insight Agent.

This module layers the earlier Insight Agent upgrade on top of the current RIA
implementation instead of replacing working chart/forecast/prescription code.

Pipeline stays unchanged:
    Schema Agent -> Goal Agent -> processed_data.json -> Insight Agent -> insights.json

Added capabilities:
- robust, manager-friendly anomaly detection,
- chart recommendation with explanation,
- encoded tabular ML prediction,
- numeric driver/association analysis,
- segment concentration analysis,
- data-quality diagnostics,
- evidence-based insight confidence score,
- executive summary and grounded story generation.

The existing ``predictions`` time-series output is preserved for compatibility.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insight_agent import InsightAgent as BaseInsightAgent
from insight_agent import sanitize_json
from prediction.tabular import TabularPredictor

log = logging.getLogger("aria.insight.advanced")


def _is_id_like(name: str) -> bool:
    low = str(name).lower().strip()
    return low in {"id", "key", "code", "codeid"} or low.endswith("_id") or low.endswith("_key")


def _human(name: str) -> str:
    return str(name).replace("_", " ").strip()


def _fmt(value: Any) -> str:
    try:
        number = float(value)
        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:.2f}M"
        if abs(number) >= 1_000:
            return f"{number:,.0f}"
        if abs(number) >= 100:
            return f"{number:,.1f}"
        return f"{number:,.2f}"
    except Exception:
        return str(value)


class AdvancedInsightAgent(BaseInsightAgent):
    """Production-oriented Insight Agent that preserves the current public contract."""

    VERSION = "2.1"

    def __init__(self, provider=None):
        super().__init__(provider=provider)
        self.encoded_prediction = TabularPredictor()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_columns(df: pd.DataFrame, numeric_cols: list[str], datetime_cols: list[str]):
        out = df.copy()
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in datetime_cols:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                try:
                    if getattr(out[col].dt, "tz", None) is not None:
                        out[col] = out[col].dt.tz_localize(None)
                except Exception:
                    pass
        return out

    @staticmethod
    def _usable_measures(numeric_cols: list[str]) -> list[str]:
        return [c for c in numeric_cols if not _is_id_like(c)]

    # ------------------------------------------------------------------
    # Robust anomaly analysis
    # ------------------------------------------------------------------

    def detect_anomalies(self, df, numeric_cols, category_cols=None, datetime_cols=None):
        """Detect robust outliers and explain them without statistical jargon.

        IQR is preferred because retail data is commonly skewed. When IQR is
        zero, a conservative standard-deviation fallback is used.
        """
        anomalies = []
        category_cols = category_cols or []
        datetime_cols = datetime_cols or []
        context_cols = datetime_cols[:1] + category_cols[:2]

        for col in self._usable_measures(numeric_cols):
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if len(valid) < 5:
                continue

            median = float(valid.median())
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            mean = float(valid.mean())
            std = float(valid.std()) if len(valid) > 1 else 0.0

            if iqr > 0:
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                severe_low = q1 - 3.0 * iqr
                severe_high = q3 + 3.0 * iqr
                mask = (series < low) | (series > high)
                method = "robust_range"
            elif std > 0 and math.isfinite(std):
                low = mean - 2.5 * std
                high = mean + 2.5 * std
                severe_low = mean - 3.5 * std
                severe_high = mean + 3.5 * std
                mask = (series < low) | (series > high)
                method = "fallback_range"
            else:
                continue

            for idx in series[mask].index.tolist():
                value = float(series.loc[idx])
                direction = "high" if value > high else "low"
                severity = "high" if value > severe_high or value < severe_low else "medium"
                pct = ((value - median) / abs(median) * 100.0) if abs(median) > 1e-12 else None

                context_items = []
                for ctx_col in context_cols:
                    if ctx_col not in df.columns:
                        continue
                    ctx_value = df.loc[idx, ctx_col]
                    if pd.notna(ctx_value):
                        if isinstance(ctx_value, pd.Timestamp):
                            ctx_value = ctx_value.isoformat()
                        context_items.append(f"{_human(ctx_col)}={ctx_value}")
                context = ", ".join(context_items) or None

                if pct is not None:
                    comparison = f"about {abs(pct):.0f}% {'above' if pct > 0 else 'below'} its usual level"
                else:
                    comparison = f"far from its usual level of about {_fmt(median)}"
                where = f" for {context}" if context else ""
                intro = "This is a strong unusual signal" if severity == "high" else "This value is unusual"
                natural = (
                    f"{intro}: {_human(col)} is {_fmt(value)}{where}, {comparison}. "
                    "It is worth checking what caused it."
                )

                anomalies.append(
                    {
                        "column": col,
                        "row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "value": round(value, 2),
                        "direction": direction,
                        "severity": severity,
                        "typical_value": round(median, 2),
                        "expected_range": [round(float(low), 2), round(float(high), 2)],
                        "difference_pct_from_typical": round(float(pct), 2) if pct is not None else None,
                        "method": method,
                        "context": context,
                        "natural_language": natural,
                    }
                )

        anomalies.sort(
            key=lambda item: (
                0 if item.get("severity") == "high" else 1,
                -abs(item.get("difference_pct_from_typical") or 0),
            )
        )
        return anomalies[:30]

    @staticmethod
    def summarize_anomalies(anomalies: list[dict[str, Any]]) -> str:
        if not anomalies:
            return "No strong unusual values were found in the main business measures."
        high = sum(1 for item in anomalies if item.get("severity") == "high")
        lead = anomalies[0].get("natural_language", "")
        if high:
            return f"ARIA found {len(anomalies)} unusual values, including {high} strong signals. {lead}"
        return f"ARIA found {len(anomalies)} unusual values. {lead}"

    # ------------------------------------------------------------------
    # Trend language
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_trends(trends: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for trend in trends or []:
            item = dict(trend)
            columns = item.get("columns") or ([item.get("column")] if item.get("column") else [])
            label = ", ".join(_human(c) for c in columns if c) or "the main measure"
            direction = item.get("direction", "mixed")
            pct = item.get("pct_change")
            if pct is None:
                sentence = f"{label} shows a {direction} movement over the available time period."
            else:
                sentence = f"{label} shows a {direction} movement over time, changing by about {pct:+.1f}%."
            item["natural_language"] = sentence
            enriched.append(item)
        return enriched

    # ------------------------------------------------------------------
    # Chart recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _chart_reason(chart_type: str | None, spec: dict[str, Any]) -> str:
        reasons = {
            "line": "A line chart is best for showing how the measure changes over time.",
            "area": "An area chart shows both the time trend and the size of the measure.",
            "bar": "A bar chart is best for comparing a numeric measure across separate groups.",
            "bar_horizontal": "A horizontal bar chart keeps category labels readable while comparing groups.",
            "bar_stacked": "A stacked bar chart compares several measures or parts across the same groups.",
            "pie": "A pie chart is suitable here because there are only a few groups and the goal is to compare their shares.",
            "doughnut": "A doughnut chart is suitable for a small number of category shares.",
            "scatter": "A scatter chart is best for checking whether two numeric measures move together.",
            "bubble": "A bubble chart compares two numeric axes while using a third measure for point size.",
            "histogram": "A histogram shows the distribution of one numeric measure.",
            "radar": "A radar chart compares several measures across a small set of dimensions.",
        }
        return spec.get("why_this_chart") or reasons.get(
            chart_type,
            "This chart matches the data types and comparison required by the result.",
        )

    def chart_recommendation(self, dashboard: list[dict[str, Any]]) -> dict[str, Any]:
        visual = [item for item in (dashboard or []) if item.get("chart_type") != "kpi_card"]
        if not visual:
            return {
                "primary_chart_type": None,
                "title": None,
                "reason": "The current result does not contain enough compatible columns for a meaningful chart.",
                "alternatives": [],
            }
        primary = visual[0]
        alternatives = []
        for item in visual[1:4]:
            ctype = item.get("chart_type")
            alternatives.append(
                {
                    "chart_type": ctype,
                    "title": item.get("title"),
                    "reason": self._chart_reason(ctype, item),
                }
            )
        ctype = primary.get("chart_type")
        return {
            "primary_chart_type": ctype,
            "title": primary.get("title"),
            "reason": self._chart_reason(ctype, primary),
            "alternatives": alternatives,
        }

    # ------------------------------------------------------------------
    # Driver and segment analysis
    # ------------------------------------------------------------------

    def driver_analysis(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        measures = self._usable_measures(numeric_cols)
        if len(measures) < 2:
            return {"target": measures[0] if measures else None, "associations": [], "note": "Not enough numeric measures for driver analysis."}

        target = measures[0]
        base = pd.to_numeric(df[target], errors="coerce")
        associations = []
        for col in measures[1:]:
            other = pd.to_numeric(df[col], errors="coerce")
            pair = pd.DataFrame({"target": base, "other": other}).dropna()
            if len(pair) < 5 or pair["target"].nunique() < 2 or pair["other"].nunique() < 2:
                continue
            corr = float(pair["target"].corr(pair["other"]))
            if not math.isfinite(corr):
                continue
            strength = "strong" if abs(corr) >= 0.7 else ("moderate" if abs(corr) >= 0.4 else "weak")
            associations.append(
                {
                    "feature": col,
                    "correlation": round(corr, 4),
                    "direction": "positive" if corr >= 0 else "negative",
                    "strength": strength,
                    "natural_language": (
                        f"{_human(col)} has a {strength} {'positive' if corr >= 0 else 'negative'} association "
                        f"with {_human(target)} in this result. This is an association, not proof of cause."
                    ),
                }
            )
        associations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return {
            "target": target,
            "associations": associations[:8],
            "note": "Correlation is used only as supporting evidence and is never presented as causation.",
        }

    def segment_analysis(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        category_cols: list[str],
    ) -> list[dict[str, Any]]:
        measures = self._usable_measures(numeric_cols)
        if not measures or not category_cols:
            return []
        measure = measures[0]
        output = []

        for dim in category_cols[:2]:
            try:
                temp = df[[dim, measure]].copy()
                temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
                temp = temp.dropna(subset=[dim, measure])
                grouped = temp.groupby(dim)[measure].sum().sort_values(ascending=False)
            except Exception:
                continue
            if grouped.empty:
                continue

            total = float(grouped.sum())
            top_label = grouped.index[0]
            top_value = float(grouped.iloc[0])
            share = (top_value / total * 100.0) if total > 0 else None
            bottom_label = grouped.index[-1]
            bottom_value = float(grouped.iloc[-1])
            concentration = (
                float(grouped.head(min(3, len(grouped))).sum()) / total * 100.0
                if total > 0
                else None
            )

            output.append(
                {
                    "dimension": dim,
                    "measure": measure,
                    "group_count": int(len(grouped)),
                    "top_segment": {
                        "label": str(top_label),
                        "value": round(top_value, 2),
                        "share_pct": round(share, 2) if share is not None else None,
                    },
                    "bottom_segment": {
                        "label": str(bottom_label),
                        "value": round(bottom_value, 2),
                    },
                    "top3_share_pct": round(concentration, 2) if concentration is not None else None,
                    "natural_language": (
                        f"{top_label} is the leading {_human(dim)} group for {_human(measure)}"
                        + (f", contributing about {share:.1f}% of the total." if share is not None else ".")
                    ),
                }
            )
        return output

    # ------------------------------------------------------------------
    # Quality and confidence
    # ------------------------------------------------------------------

    @staticmethod
    def data_quality(df: pd.DataFrame) -> dict[str, Any]:
        rows, cols = df.shape
        cells = rows * cols
        null_cells = int(df.isna().sum().sum()) if cells else 0
        null_pct = (null_cells / cells * 100.0) if cells else 0.0
        duplicates = int(df.duplicated().sum()) if rows else 0
        high_null = [str(c) for c in df.columns if rows and float(df[c].isna().mean()) >= 0.40]
        constants = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
        return {
            "rows": int(rows),
            "columns": int(cols),
            "null_cells": null_cells,
            "null_percentage": round(null_pct, 2),
            "duplicate_rows": duplicates,
            "duplicate_percentage": round(duplicates / rows * 100.0, 2) if rows else 0.0,
            "high_null_columns": high_null,
            "constant_columns": constants,
        }

    @staticmethod
    def insight_confidence(
        quality: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        trends: list[dict[str, Any]],
        encoded_prediction: dict[str, Any] | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        score = 45.0
        rows = int(quality.get("rows", 0))
        score += min(20.0, math.log10(max(rows, 1)) * 7.0)
        score -= min(18.0, float(quality.get("null_percentage", 0)) * 0.35)
        score -= min(10.0, float(quality.get("duplicate_percentage", 0)) * 0.25)
        score += min(10.0, len(hypotheses) * 2.0)
        score += min(5.0, len(trends) * 1.5)
        score += min(5.0, len(anomalies) * 0.5)

        if encoded_prediction:
            reliability = encoded_prediction.get("reliability")
            score += {"high": 8.0, "medium": 4.0, "low": -3.0}.get(reliability, 0.0)
        score -= min(12.0, len(warnings) * 3.0)
        score = max(0.0, min(100.0, score))

        level = "high" if score >= 80 else ("medium" if score >= 60 else "low")
        reasons = []
        if rows < 20:
            reasons.append("small result set")
        if quality.get("null_percentage", 0) >= 20:
            reasons.append("substantial missing data")
        if quality.get("duplicate_percentage", 0) >= 10:
            reasons.append("many duplicate rows")
        if warnings:
            reasons.append("one or more analysis steps used a fallback")
        if not reasons:
            reasons.append("result size and data quality are adequate for exploratory BI analysis")

        return {"score": round(score, 1), "level": level, "reasons": reasons}

    # ------------------------------------------------------------------
    # Executive synthesis + story
    # ------------------------------------------------------------------

    def executive_summary(
        self,
        processed_data: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        trends: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        drivers: dict[str, Any],
        confidence: dict[str, Any],
    ) -> dict[str, Any]:
        headline = (
            hypotheses[0].get("hypothesis")
            if hypotheses
            else (
                trends[0].get("natural_language")
                if trends
                else "ARIA completed the analysis, but no single dominant business signal was found."
            )
        )
        evidence = []
        if trends:
            evidence.append(trends[0].get("natural_language"))
        if segments:
            evidence.append(segments[0].get("natural_language"))
        if anomalies:
            evidence.append(anomalies[0].get("natural_language"))
        associations = drivers.get("associations") or []
        if associations:
            evidence.append(associations[0].get("natural_language"))

        checks = []
        if anomalies:
            checks.append("Verify the strongest unusual record against source data and recent business events.")
        if segments:
            checks.append(f"Compare the leading and weakest {_human(segments[0]['dimension'])} groups before taking action.")
        if associations:
            checks.append("Test the strongest association with a controlled business comparison before treating it as a cause.")
        if not checks:
            checks.append("Collect another comparable period or segment before making a high-impact decision.")

        return {
            "headline": headline,
            "confidence": confidence,
            "supporting_evidence": [x for x in evidence if x][:4],
            "recommended_checks": checks[:3],
            "goal": processed_data.get("user_goal"),
        }

    def generate_business_story(
        self,
        processed_data,
        kpis,
        trends,
        anomalies,
        hypotheses,
        predictions=None,
        encoded_prediction=None,
        executive_summary=None,
        segments=None,
        drivers=None,
    ):
        executive_summary = executive_summary or {}
        segments = segments or []
        drivers = drivers or {}
        parts = []

        headline = executive_summary.get("headline")
        if headline:
            parts.append(f"Main finding: {headline}")
        elif hypotheses:
            parts.append(f"Main finding: {hypotheses[0].get('hypothesis')}")

        if trends:
            parts.append(trends[0].get("natural_language", ""))
        if segments:
            parts.append(segments[0].get("natural_language", ""))
        if anomalies:
            parts.append(anomalies[0].get("natural_language", ""))
        associations = drivers.get("associations") or []
        if associations:
            parts.append(associations[0].get("natural_language", ""))
        if predictions and predictions.get("valid") and predictions.get("points"):
            parts.append(
                f"The existing time-series forecast expects {_human(predictions.get('column'))} to change by about "
                f"{predictions.get('projected_pct_change')}% over the next {predictions.get('horizon')} "
                f"{predictions.get('period') or 'periods'}."
            )
        if encoded_prediction and encoded_prediction.get("valid"):
            parts.append(encoded_prediction.get("natural_language", ""))
        if not anomalies:
            parts.append("No strong unusual values were found in the main business measures.")

        base = " ".join(part for part in parts if part).strip()
        if not base:
            base = "ARIA analysed the result but did not find a strong signal worth highlighting."

        context = {
            "goal": processed_data.get("user_goal"),
            "kpis": kpis[:4],
            "executive_summary": executive_summary,
            "encoded_prediction": (
                encoded_prediction.get("natural_language")
                if encoded_prediction and encoded_prediction.get("valid")
                else None
            ),
        }
        prompt = f"""
You are ARIA's retail and sales Insight Agent. Rewrite the draft for a busy,
non-technical manager.

DRAFT:
{base}

VERIFIED FACTS:
{json.dumps(context, indent=2, default=str)}

Rules:
- Keep every claim grounded in the supplied facts.
- Use short, easy sentences.
- Start with the most important finding and explain why it matters.
- Never claim that correlation or association proves a cause.
- If a cause is unknown, say it is worth checking rather than inventing one.
- Do not use statistical or machine-learning jargon.
- Maximum 180 words. No markdown. No bullets. No JSON.
"""
        try:
            polished = self.llm.chat(
                "story",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                num_predict=300,
                timeout=35,
            )
            polished = (polished or "").strip()
            if 40 <= len(polished) <= 1400:
                return polished
        except Exception as exc:
            log.warning("Advanced story LLM failed; deterministic story used: %s", exc)
        return base

    # ------------------------------------------------------------------
    # Full advanced pipeline
    # ------------------------------------------------------------------

    def analyze(self, processed_data_path="processed_data.json", output_path="insights.json"):
        warnings: list[str] = []
        try:
            processed_data = self.load_processed_data(processed_data_path)
            df = pd.DataFrame(processed_data.get("data", []))
        except Exception as exc:
            processed_data = {}
            df = pd.DataFrame()
            warnings.append(f"load_processed_data: {exc}")

        print(
            f"Advanced Insight Agent analysing {processed_data_path} "
            f"({processed_data.get('row_count', len(df))} rows)..."
        )

        def safe(name, fn, default):
            try:
                return fn()
            except Exception as exc:
                log.warning("Insight step '%s' failed: %s", name, exc)
                warnings.append(f"{name}: {exc}")
                return default

        numeric_cols, category_cols, datetime_cols = safe(
            "classify_columns",
            lambda: self._classify_columns(df),
            ([], [], []),
        )
        df = safe(
            "coerce_columns",
            lambda: self._coerce_columns(df, numeric_cols, datetime_cols),
            df,
        )

        kpis = safe("compute_kpis", lambda: self.compute_kpis(df, numeric_cols), [])
        trends = safe("detect_trends", lambda: self.detect_trends(df, datetime_cols, numeric_cols), [])
        trends = self._enrich_trends(trends)
        anomalies = safe(
            "detect_anomalies",
            lambda: self.detect_anomalies(df, numeric_cols, category_cols, datetime_cols),
            [],
        )
        anomaly_summary = self.summarize_anomalies(anomalies)

        dashboard = safe(
            "build_dashboard_spec",
            lambda: self.build_dashboard_spec(df, numeric_cols, category_cols, datetime_cols, None),
            [],
        )
        chart_choice = self.chart_recommendation(dashboard)

        hypotheses, ranking_score = safe(
            "generate_hypotheses",
            lambda: self.generate_hypotheses(
                df,
                numeric_cols,
                category_cols,
                datetime_cols,
                kpis,
                anomalies,
            ),
            ([], 0.0),
        )

        drivers = safe("driver_analysis", lambda: self.driver_analysis(df, numeric_cols), {"target": None, "associations": []})
        segments = safe("segment_analysis", lambda: self.segment_analysis(df, numeric_cols, category_cols), [])
        quality = safe("data_quality", lambda: self.data_quality(df), {})

        predictions = safe(
            "time_series_prediction",
            lambda: self.prediction.forecast(
                processed_data,
                datetime_col=datetime_cols[0] if datetime_cols else None,
            ) if numeric_cols else None,
            None,
        )
        encoded_prediction = safe(
            "encoded_tabular_prediction",
            lambda: self.encoded_prediction.predict(processed_data),
            None,
        )

        prescriptions = safe(
            "prescription",
            lambda: self.prescription.prescribe(
                processed_data,
                kpis,
                trends,
                anomalies,
                hypotheses,
                predictions,
            ),
            None,
        )

        confidence = self.insight_confidence(
            quality,
            hypotheses,
            anomalies,
            trends,
            encoded_prediction,
            warnings,
        )
        executive = self.executive_summary(
            processed_data,
            hypotheses,
            trends,
            anomalies,
            segments,
            drivers,
            confidence,
        )

        story = safe(
            "business_story",
            lambda: self.generate_business_story(
                processed_data,
                kpis,
                trends,
                anomalies,
                hypotheses,
                predictions,
                encoded_prediction,
                executive,
                segments,
                drivers,
            ),
            "ARIA completed the analysis. Review the verified KPIs and charts for the available evidence.",
        )

        insights = {
            "analysis_version": self.VERSION,
            "generated_at": datetime.now().isoformat(),
            "source_processed_data": processed_data_path,
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "summary": {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": category_cols,
                "datetime_columns": datetime_cols,
            },
            "data_quality": quality,
            "kpis": kpis,
            "trends": trends,
            "anomalies": anomalies,
            "anomaly_summary": anomaly_summary,
            "hypotheses": hypotheses,
            "hypothesis_ranking_score": round(float(ranking_score), 2),
            "drivers": drivers,
            "segments": segments,
            "predictions": predictions,
            "encoded_prediction": encoded_prediction,
            "prescriptions": prescriptions,
            "chart_recommendation": chart_choice,
            "dashboard": dashboard,
            "insight_confidence": confidence,
            "executive_summary": executive,
            "business_story": story,
            "warning": "; ".join(warnings) if warnings else None,
        }

        output = sanitize_json(insights)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"Advanced Insight Agent done! Saved to {output_path}")
        return output_path


# Compatibility alias for code that wants an InsightAgent name from this module.
InsightAgent = AdvancedInsightAgent


__all__ = ["AdvancedInsightAgent", "InsightAgent", "sanitize_json"]
