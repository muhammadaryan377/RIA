"""Goal-aware decision intelligence for ARIA.

This module is deliberately deterministic. It does not replace the existing
Insight Agent or ask an LLM to invent explanations. Instead it converts the
Goal Agent contract plus verified result rows into evidence-backed business
findings that the advanced Insight Agent can rank and narrate.

Main capabilities
-----------------
- resolve the requested measure/dimensions from ``analysis_plan``;
- period-over-period comparison;
- contribution analysis by segment;
- recursive root-cause drill-down across dimensions;
- retail-specific signals (discount effectiveness, concentration, margin);
- business-impact estimation;
- evidence packets and fact/association/hypothesis separation;
- goal-aware ranking and deduplication of insights.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


_ID_EXACT = {"id", "key", "code", "codeid", "row_number", "rownumber"}
_REVENUE_WORDS = ("revenue", "sales", "amount", "turnover", "gmv", "net_sales", "total_sales")
_PROFIT_WORDS = ("profit", "margin", "gross_profit", "net_profit")
_DISCOUNT_WORDS = ("discount", "disc", "markdown")
_QUANTITY_WORDS = ("quantity", "qty", "units", "volume")
_CUSTOMER_WORDS = ("customer", "client", "buyer")
_PRODUCT_WORDS = ("product", "item", "sku")
_CATEGORY_WORDS = ("category", "segment", "department")
_REGION_WORDS = ("region", "country", "city", "state", "territory", "market")


def _is_id_like(name: str) -> bool:
    low = str(name).strip().lower()
    return (
        low in _ID_EXACT
        or low.endswith("_id")
        or low.endswith("_key")
        or (low.endswith("id") and len(low) <= 18)
    )


def _human(name: str) -> str:
    return str(name).replace("_", " ").replace(".", " ").strip()


def _norm(name: Any) -> str:
    text = str(name or "").lower()
    text = re.sub(r"\bas\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokens(name: Any) -> set[str]:
    return set(_norm(name).split())


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct_change(current: float, previous: float) -> float | None:
    if abs(previous) <= 1e-12:
        return None
    return (current - previous) / abs(previous) * 100.0


def _confidence_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return {"high": 0.9, "medium": 0.68, "low": 0.42}.get(str(value).lower(), 0.5)


def _word_match(name: str, words: tuple[str, ...]) -> bool:
    low = _norm(name).replace(" ", "_")
    return any(word in low for word in words)


class DecisionIntelligence:
    """Turn verified query results into ranked decision evidence."""

    VERSION = "1.0"

    def analyze(
        self,
        df: pd.DataFrame,
        processed_data: dict[str, Any],
        numeric_cols: list[str],
        category_cols: list[str],
        datetime_cols: list[str],
        *,
        hypotheses: list[dict[str, Any]] | None = None,
        anomalies: list[dict[str, Any]] | None = None,
        trends: list[dict[str, Any]] | None = None,
        drivers: dict[str, Any] | None = None,
        segments: list[dict[str, Any]] | None = None,
        encoded_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hypotheses = hypotheses or []
        anomalies = anomalies or []
        trends = trends or []
        drivers = drivers or {"associations": []}
        segments = segments or []

        profile = self.goal_profile(
            processed_data,
            df,
            numeric_cols,
            category_cols,
            datetime_cols,
        )
        metric = profile.get("primary_measure")
        dimensions = profile.get("dimensions") or []
        date_col = profile.get("time_dimension")

        period = self.period_comparison(df, metric, date_col)
        contributions = self.contribution_analysis(df, metric, dimensions, date_col, period)
        drilldown = self.root_cause_drilldown(df, metric, dimensions, date_col, period)
        retail = self.retail_signals(df, metric, numeric_cols, category_cols)
        impact = self.business_impact(metric, period, contributions)

        findings = self.build_ranked_findings(
            profile=profile,
            period=period,
            contributions=contributions,
            drilldown=drilldown,
            retail=retail,
            anomalies=anomalies,
            trends=trends,
            drivers=drivers,
            segments=segments,
            hypotheses=hypotheses,
            encoded_prediction=encoded_prediction,
        )
        evidence = self.evidence_model(
            profile,
            period,
            contributions,
            drilldown,
            retail,
            anomalies,
            drivers,
            hypotheses,
            findings,
        )

        return {
            "version": self.VERSION,
            "goal_profile": profile,
            "period_comparison": period,
            "contribution_analysis": contributions,
            "root_cause_drilldown": drilldown,
            "retail_signals": retail,
            "business_impact": impact,
            "ranked_insights": findings,
            "evidence_model": evidence,
        }

    # ------------------------------------------------------------------
    # Goal awareness
    # ------------------------------------------------------------------

    def goal_profile(
        self,
        processed_data: dict[str, Any],
        df: pd.DataFrame,
        numeric_cols: list[str],
        category_cols: list[str],
        datetime_cols: list[str],
    ) -> dict[str, Any]:
        goal_obj = processed_data.get("goal") or {}
        question = (
            goal_obj.get("original_question")
            or processed_data.get("user_goal")
            or processed_data.get("question")
            or ""
        )
        analysis_plan = processed_data.get("analysis_plan") or {}
        data_selection = processed_data.get("data_selection") or {}

        planned_measures = list(analysis_plan.get("measures") or [])
        planned_dimensions = list(analysis_plan.get("dimensions") or [])
        group_by = list(analysis_plan.get("group_by") or [])
        filters = list(data_selection.get("filters") or [])

        measures = [c for c in numeric_cols if not _is_id_like(c)]
        primary_measure = self._resolve_measure(
            question,
            planned_measures,
            measures,
            list(df.columns),
        )
        resolved_dimensions = self._resolve_dimensions(
            question,
            planned_dimensions + group_by,
            category_cols,
            list(df.columns),
        )
        if not resolved_dimensions:
            resolved_dimensions = [c for c in category_cols if c != primary_measure][:3]

        time_dimension = self._resolve_time_dimension(
            question,
            datetime_cols,
            list(df.columns),
        )

        keywords = _tokens(question)
        focus = []
        if any("discount" in token for token in keywords):
            focus.append("discount_effectiveness")
        if any(token in keywords for token in {"trend", "growth", "decline", "change", "over", "time"}):
            focus.append("trend")
        if any(token in keywords for token in {"why", "reason", "cause", "driver", "contribute", "contribution"}):
            focus.append("drivers")
        if any(token in keywords for token in {"predict", "forecast", "future", "expected"}):
            focus.append("prediction")
        if any(token in keywords for token in {"top", "bottom", "best", "worst", "rank"}):
            focus.append("ranking")

        return {
            "question": question,
            "intent": goal_obj.get("intent"),
            "analysis_type": goal_obj.get("analysis_type"),
            "primary_measure": primary_measure,
            "planned_measures": planned_measures,
            "dimensions": resolved_dimensions[:4],
            "time_dimension": time_dimension,
            "filters": filters,
            "focus": focus,
            "goal_tokens": sorted(keywords),
        }

    def _resolve_measure(
        self,
        question: str,
        planned: list[Any],
        numeric_cols: list[str],
        all_columns: list[str],
    ) -> str | None:
        if not numeric_cols:
            return None
        q_tokens = _tokens(question)
        best: tuple[float, str] | None = None
        plan_tokens = set()
        for item in planned:
            plan_tokens |= _tokens(item)

        for col in numeric_cols:
            col_tokens = _tokens(col)
            score = 0.0
            score += len(col_tokens & q_tokens) * 4.0
            score += len(col_tokens & plan_tokens) * 3.0
            norm_col = _norm(col)
            if norm_col and norm_col in _norm(question):
                score += 6.0
            for item in planned:
                if norm_col and norm_col in _norm(item):
                    score += 5.0
            if _word_match(col, _REVENUE_WORDS) and any(t in q_tokens for t in {"sales", "revenue", "amount"}):
                score += 5.0
            if _word_match(col, _PROFIT_WORDS) and any(t in q_tokens for t in {"profit", "margin"}):
                score += 5.0
            if best is None or score > best[0]:
                best = (score, col)

        if best and best[0] > 0:
            return best[1]
        return numeric_cols[0]

    def _resolve_dimensions(
        self,
        question: str,
        planned: list[Any],
        category_cols: list[str],
        all_columns: list[str],
    ) -> list[str]:
        q_tokens = _tokens(question)
        plan_tokens = set()
        for item in planned:
            plan_tokens |= _tokens(item)

        scored = []
        for col in category_cols:
            col_tokens = _tokens(col)
            score = len(col_tokens & q_tokens) * 4.0 + len(col_tokens & plan_tokens) * 3.0
            if _norm(col) in _norm(question):
                score += 5.0
            scored.append((score, col))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [col for score, col in scored if score > 0]
        return chosen[:4]

    def _resolve_time_dimension(
        self,
        question: str,
        datetime_cols: list[str],
        all_columns: list[str],
    ) -> str | None:
        if not datetime_cols:
            return None
        q = _norm(question)
        for col in datetime_cols:
            if _norm(col) in q:
                return col
        return datetime_cols[0]

    # ------------------------------------------------------------------
    # Time comparison
    # ------------------------------------------------------------------

    def period_comparison(
        self,
        df: pd.DataFrame,
        metric: str | None,
        date_col: str | None,
    ) -> dict[str, Any] | None:
        if not metric or not date_col or metric not in df.columns or date_col not in df.columns:
            return None

        temp = df[[date_col, metric]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        temp = temp.dropna()
        if len(temp) < 2:
            return None

        span_days = int(max(0, (temp[date_col].max() - temp[date_col].min()).days))
        if span_days >= 730:
            granularity = "year"
            temp["_period"] = temp[date_col].dt.to_period("Y").astype(str)
        elif span_days >= 90:
            granularity = "month"
            temp["_period"] = temp[date_col].dt.to_period("M").astype(str)
        elif span_days >= 21:
            granularity = "week"
            temp["_period"] = temp[date_col].dt.to_period("W").astype(str)
        else:
            granularity = "day"
            temp["_period"] = temp[date_col].dt.to_period("D").astype(str)

        grouped = temp.groupby("_period")[metric].sum().sort_index()
        if len(grouped) < 2:
            return None

        previous_period, current_period = grouped.index[-2], grouped.index[-1]
        previous = float(grouped.iloc[-2])
        current = float(grouped.iloc[-1])
        delta = current - previous
        pct = _pct_change(current, previous)
        direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "was unchanged")
        pct_text = f" ({abs(pct):.1f}%)" if pct is not None else ""

        return {
            "metric": metric,
            "date_column": date_col,
            "granularity": granularity,
            "previous_period": str(previous_period),
            "current_period": str(current_period),
            "previous_value": round(previous, 4),
            "current_value": round(current, 4),
            "absolute_change": round(delta, 4),
            "pct_change": round(pct, 2) if pct is not None else None,
            "direction": direction,
            "period_count": int(len(grouped)),
            "series": [
                {"period": str(idx), "value": round(float(val), 4)}
                for idx, val in grouped.tail(24).items()
            ],
            "natural_language": (
                f"{_human(metric)} {direction} from {previous:,.2f} in {previous_period} "
                f"to {current:,.2f} in {current_period}{pct_text}."
            ),
        }

    def _attach_period(self, df: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
        temp = df.copy()
        parsed = pd.to_datetime(temp[date_col], errors="coerce")
        if granularity == "year":
            temp["_period"] = parsed.dt.to_period("Y").astype(str)
        elif granularity == "month":
            temp["_period"] = parsed.dt.to_period("M").astype(str)
        elif granularity == "week":
            temp["_period"] = parsed.dt.to_period("W").astype(str)
        else:
            temp["_period"] = parsed.dt.to_period("D").astype(str)
        return temp

    # ------------------------------------------------------------------
    # Contribution and root-cause drill-down
    # ------------------------------------------------------------------

    def contribution_analysis(
        self,
        df: pd.DataFrame,
        metric: str | None,
        dimensions: list[str],
        date_col: str | None,
        period: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not metric or metric not in df.columns:
            return []
        usable_dims = [d for d in dimensions if d in df.columns and d != metric][:3]
        if not usable_dims:
            return []

        temp = df.copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        results = []

        if period and date_col and date_col in temp.columns:
            temp = self._attach_period(temp, date_col, period["granularity"])
            current_period = period["current_period"]
            previous_period = period["previous_period"]
            current = temp[temp["_period"] == current_period]
            previous = temp[temp["_period"] == previous_period]
            overall_delta = float(period["absolute_change"])

            for dim in usable_dims:
                cur = current.groupby(dim, dropna=False)[metric].sum()
                prev = previous.groupby(dim, dropna=False)[metric].sum()
                labels = cur.index.union(prev.index)
                rows = []
                for label in labels:
                    cv = float(cur.get(label, 0.0))
                    pv = float(prev.get(label, 0.0))
                    delta = cv - pv
                    pct = _pct_change(cv, pv)
                    rows.append(
                        {
                            "segment": str(label),
                            "current": round(cv, 4),
                            "previous": round(pv, 4),
                            "absolute_change": round(delta, 4),
                            "pct_change": round(pct, 2) if pct is not None else None,
                        }
                    )

                aligned = [
                    row
                    for row in rows
                    if (overall_delta >= 0 and row["absolute_change"] > 0)
                    or (overall_delta < 0 and row["absolute_change"] < 0)
                ]
                aligned_total = sum(abs(row["absolute_change"]) for row in aligned) or 1.0
                for row in rows:
                    row["driver_share_pct"] = round(
                        abs(row["absolute_change"]) / aligned_total * 100.0,
                        2,
                    ) if row in aligned else 0.0
                rows.sort(key=lambda r: abs(r["absolute_change"]), reverse=True)
                top = rows[:8]
                if not top:
                    continue
                lead = top[0]
                direction = "increase" if lead["absolute_change"] >= 0 else "decline"
                results.append(
                    {
                        "dimension": dim,
                        "metric": metric,
                        "mode": "period_change",
                        "current_period": current_period,
                        "previous_period": previous_period,
                        "overall_change": round(overall_delta, 4),
                        "contributors": top,
                        "natural_language": (
                            f"The largest {_human(dim)} contributor to the {_human(metric)} {direction} "
                            f"was {lead['segment']}, with a change of {lead['absolute_change']:,.2f}."
                        ),
                    }
                )
            return results

        total = float(temp[metric].sum(skipna=True))
        for dim in usable_dims:
            grouped = temp.groupby(dim, dropna=False)[metric].sum().sort_values(ascending=False)
            if grouped.empty:
                continue
            rows = []
            for label, value in grouped.head(10).items():
                share = float(value) / total * 100.0 if abs(total) > 1e-12 else None
                rows.append(
                    {
                        "segment": str(label),
                        "value": round(float(value), 4),
                        "share_pct": round(share, 2) if share is not None else None,
                    }
                )
            lead = rows[0]
            results.append(
                {
                    "dimension": dim,
                    "metric": metric,
                    "mode": "share_of_total",
                    "contributors": rows,
                    "natural_language": (
                        f"{lead['segment']} is the largest {_human(dim)} contributor to {_human(metric)}"
                        + (f" at about {lead['share_pct']:.1f}% of the total." if lead.get("share_pct") is not None else ".")
                    ),
                }
            )
        return results

    def root_cause_drilldown(
        self,
        df: pd.DataFrame,
        metric: str | None,
        dimensions: list[str],
        date_col: str | None,
        period: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        usable_dims = [d for d in dimensions if d in df.columns][:3]
        if not metric or not usable_dims or not period or not date_col or date_col not in df.columns:
            return None

        temp = df.copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        temp = self._attach_period(temp, date_col, period["granularity"])
        current_period = period["current_period"]
        previous_period = period["previous_period"]
        current = temp[temp["_period"] == current_period].copy()
        previous = temp[temp["_period"] == previous_period].copy()
        target_direction = 1 if float(period["absolute_change"]) >= 0 else -1

        path = []
        current_filter = current
        previous_filter = previous
        for depth, dim in enumerate(usable_dims, start=1):
            cur = current_filter.groupby(dim, dropna=False)[metric].sum()
            prev = previous_filter.groupby(dim, dropna=False)[metric].sum()
            labels = cur.index.union(prev.index)
            candidates = []
            for label in labels:
                cv = float(cur.get(label, 0.0))
                pv = float(prev.get(label, 0.0))
                delta = cv - pv
                aligned = delta * target_direction
                candidates.append((aligned, abs(delta), label, cv, pv, delta))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            _, _, label, cv, pv, delta = candidates[0]
            if abs(delta) <= 1e-12:
                break
            pct = _pct_change(cv, pv)
            path.append(
                {
                    "depth": depth,
                    "dimension": dim,
                    "segment": str(label),
                    "current": round(cv, 4),
                    "previous": round(pv, 4),
                    "absolute_change": round(delta, 4),
                    "pct_change": round(pct, 2) if pct is not None else None,
                }
            )
            current_filter = current_filter[current_filter[dim].astype(str) == str(label)]
            previous_filter = previous_filter[previous_filter[dim].astype(str) == str(label)]

        if not path:
            return None
        chain = " -> ".join(f"{_human(p['dimension'])}: {p['segment']}" for p in path)
        return {
            "metric": metric,
            "current_period": current_period,
            "previous_period": previous_period,
            "path": path,
            "natural_language": (
                f"The strongest drill-down path behind the {_human(metric)} change is {chain}. "
                "This identifies where the change is concentrated; it does not by itself prove causation."
            ),
        }

    # ------------------------------------------------------------------
    # Retail-specific signals
    # ------------------------------------------------------------------

    def retail_signals(
        self,
        df: pd.DataFrame,
        primary_metric: str | None,
        numeric_cols: list[str],
        category_cols: list[str],
    ) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        numeric = [c for c in numeric_cols if c in df.columns and not _is_id_like(c)]
        categorical = [c for c in category_cols if c in df.columns]

        discount_col = next((c for c in numeric if _word_match(c, _DISCOUNT_WORDS)), None)
        revenue_col = next((c for c in numeric if _word_match(c, _REVENUE_WORDS)), None)
        profit_col = next((c for c in numeric if _word_match(c, _PROFIT_WORDS)), None)
        quantity_col = next((c for c in numeric if _word_match(c, _QUANTITY_WORDS)), None)
        customer_col = next((c for c in categorical if _word_match(c, _CUSTOMER_WORDS)), None)
        product_col = next((c for c in categorical if _word_match(c, _PRODUCT_WORDS)), None)
        category_col = next((c for c in categorical if _word_match(c, _CATEGORY_WORDS)), None)
        region_col = next((c for c in categorical if _word_match(c, _REGION_WORDS)), None)

        target_for_discount = profit_col or revenue_col or primary_metric
        if discount_col and target_for_discount and target_for_discount in df.columns:
            d = pd.to_numeric(df[discount_col], errors="coerce")
            m = pd.to_numeric(df[target_for_discount], errors="coerce")
            pair = pd.DataFrame({"discount": d, "metric": m}).dropna()
            if len(pair) >= 8 and pair["discount"].nunique() >= 3 and pair["metric"].nunique() >= 2:
                corr = float(pair["discount"].corr(pair["metric"]))
                if math.isfinite(corr):
                    direction = "moves up with" if corr > 0 else "moves down as"
                    strength = "strong" if abs(corr) >= 0.6 else ("moderate" if abs(corr) >= 0.35 else "weak")
                    signals.append(
                        {
                            "type": "discount_effectiveness",
                            "metric": target_for_discount,
                            "discount_column": discount_col,
                            "correlation": round(corr, 4),
                            "strength": strength,
                            "natural_language": (
                                f"{_human(target_for_discount)} {direction} {_human(discount_col)} changes in this result, "
                                f"with a {strength} association. This does not prove that discount caused the change."
                            ),
                        }
                    )

        if revenue_col and profit_col:
            revenue = pd.to_numeric(df[revenue_col], errors="coerce").sum()
            profit = pd.to_numeric(df[profit_col], errors="coerce").sum()
            if abs(revenue) > 1e-12:
                margin = float(profit / revenue * 100.0)
                signals.append(
                    {
                        "type": "profit_margin",
                        "revenue_metric": revenue_col,
                        "profit_metric": profit_col,
                        "margin_pct": round(margin, 2),
                        "natural_language": f"Overall profit is about {margin:.1f}% of {_human(revenue_col)} in this result.",
                    }
                )

        metric = primary_metric or revenue_col or profit_col or quantity_col
        for dim, signal_type in (
            (customer_col, "customer_concentration"),
            (product_col, "product_concentration"),
            (category_col, "category_concentration"),
            (region_col, "regional_concentration"),
        ):
            if not dim or not metric or metric not in df.columns:
                continue
            temp = df[[dim, metric]].copy()
            temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
            temp = temp.dropna()
            grouped = temp.groupby(dim)[metric].sum().sort_values(ascending=False)
            total = float(grouped.sum()) if len(grouped) else 0.0
            if len(grouped) < 2 or abs(total) <= 1e-12:
                continue
            top_share = float(grouped.iloc[0]) / total * 100.0
            top3_share = float(grouped.head(min(3, len(grouped))).sum()) / total * 100.0
            signals.append(
                {
                    "type": signal_type,
                    "dimension": dim,
                    "metric": metric,
                    "top_segment": str(grouped.index[0]),
                    "top_share_pct": round(top_share, 2),
                    "top3_share_pct": round(top3_share, 2),
                    "natural_language": (
                        f"The leading {_human(dim)} group, {grouped.index[0]}, contributes about {top_share:.1f}% "
                        f"of {_human(metric)}; the top three contribute about {top3_share:.1f}%."
                    ),
                }
            )

        if quantity_col and revenue_col:
            qty = pd.to_numeric(df[quantity_col], errors="coerce")
            rev = pd.to_numeric(df[revenue_col], errors="coerce")
            pair = pd.DataFrame({"qty": qty, "rev": rev}).dropna()
            if len(pair) >= 5 and pair["qty"].sum() > 0:
                per_unit = float(pair["rev"].sum() / pair["qty"].sum())
                signals.append(
                    {
                        "type": "revenue_per_unit",
                        "revenue_metric": revenue_col,
                        "quantity_metric": quantity_col,
                        "value": round(per_unit, 4),
                        "natural_language": (
                            f"{_human(revenue_col)} averages about {per_unit:,.2f} per unit of {_human(quantity_col)} "
                            "across the returned data."
                        ),
                    }
                )

        return signals[:10]

    # ------------------------------------------------------------------
    # Business impact
    # ------------------------------------------------------------------

    def business_impact(
        self,
        metric: str | None,
        period: dict[str, Any] | None,
        contributions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not metric or not period:
            return None
        delta = float(period.get("absolute_change") or 0.0)
        pct = period.get("pct_change")
        direction = "uplift" if delta > 0 else ("gap" if delta < 0 else "no_change")
        lead = None
        if contributions and contributions[0].get("contributors"):
            lead = contributions[0]["contributors"][0]
        return {
            "metric": metric,
            "type": direction,
            "absolute_change": round(delta, 4),
            "pct_change": pct,
            "leading_driver": lead,
            "natural_language": (
                f"The latest period shows an absolute {_human(metric)} change of {delta:,.2f}"
                + (f" ({pct:+.1f}%)." if pct is not None else ".")
                + (
                    f" The largest visible contributor is {lead.get('segment')}."
                    if lead else ""
                )
            ),
        }

    # ------------------------------------------------------------------
    # Ranking and evidence
    # ------------------------------------------------------------------

    def build_ranked_findings(
        self,
        *,
        profile: dict[str, Any],
        period: dict[str, Any] | None,
        contributions: list[dict[str, Any]],
        drilldown: dict[str, Any] | None,
        retail: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        trends: list[dict[str, Any]],
        drivers: dict[str, Any],
        segments: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        encoded_prediction: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        goal_tokens = set(profile.get("goal_tokens") or [])
        primary = profile.get("primary_measure")

        def add(
            kind: str,
            statement: str | None,
            *,
            metric: str | None = None,
            magnitude_pct: float | None = None,
            confidence: Any = "medium",
            evidence: dict[str, Any] | None = None,
            source: str,
            business_relevance: float = 0.7,
        ):
            if not statement:
                return
            metric_tokens = _tokens(metric or "")
            goal_alignment = 1.0 if metric and metric == primary else 0.55
            if metric_tokens & goal_tokens:
                goal_alignment = max(goal_alignment, 0.9)
            statement_tokens = _tokens(statement)
            if statement_tokens & goal_tokens:
                goal_alignment = max(goal_alignment, 0.78)
            magnitude_score = min(1.0, abs(float(magnitude_pct or 0.0)) / 50.0)
            confidence_score = _confidence_number(confidence)
            evidence_score = 1.0 if evidence else 0.55
            score = (
                magnitude_score * 30.0
                + confidence_score * 25.0
                + goal_alignment * 25.0
                + max(0.0, min(1.0, business_relevance)) * 10.0
                + evidence_score * 10.0
            )
            candidates.append(
                {
                    "type": kind,
                    "statement": statement,
                    "metric": metric,
                    "magnitude_pct": round(float(magnitude_pct), 2) if magnitude_pct is not None else None,
                    "confidence": round(confidence_score, 2),
                    "goal_alignment": round(goal_alignment, 2),
                    "score": round(score, 2),
                    "source": source,
                    "evidence": evidence or {},
                }
            )

        if period:
            add(
                "period_change",
                period.get("natural_language"),
                metric=period.get("metric"),
                magnitude_pct=period.get("pct_change"),
                confidence="high" if period.get("period_count", 0) >= 4 else "medium",
                evidence={
                    "previous_period": period.get("previous_period"),
                    "current_period": period.get("current_period"),
                    "previous_value": period.get("previous_value"),
                    "current_value": period.get("current_value"),
                },
                source="period_comparison",
                business_relevance=0.95,
            )

        for item in contributions[:2]:
            lead = (item.get("contributors") or [{}])[0]
            add(
                "contribution",
                item.get("natural_language"),
                metric=item.get("metric"),
                magnitude_pct=lead.get("pct_change") or lead.get("share_pct"),
                confidence="high" if len(item.get("contributors") or []) >= 3 else "medium",
                evidence={"dimension": item.get("dimension"), "leading_contributor": lead},
                source="contribution_analysis",
                business_relevance=0.95,
            )

        if drilldown:
            leaf = (drilldown.get("path") or [{}])[-1]
            add(
                "root_cause_location",
                drilldown.get("natural_language"),
                metric=drilldown.get("metric"),
                magnitude_pct=leaf.get("pct_change"),
                confidence="medium",
                evidence={"path": drilldown.get("path")},
                source="root_cause_drilldown",
                business_relevance=1.0,
            )

        for item in retail[:4]:
            mag = item.get("top_share_pct") or item.get("margin_pct")
            if item.get("correlation") is not None:
                mag = abs(float(item["correlation"])) * 100.0
            add(
                item.get("type", "retail_signal"),
                item.get("natural_language"),
                metric=item.get("metric") or item.get("profit_metric") or item.get("revenue_metric"),
                magnitude_pct=mag,
                confidence="medium",
                evidence=item,
                source="retail_signals",
                business_relevance=0.9,
            )

        for item in anomalies[:3]:
            add(
                "anomaly",
                item.get("natural_language") or item.get("note"),
                metric=item.get("column"),
                magnitude_pct=item.get("difference_pct_from_typical"),
                confidence="high" if item.get("severity") == "high" else "medium",
                evidence=item,
                source="anomaly_detection",
                business_relevance=0.82,
            )

        for item in (drivers.get("associations") or [])[:2]:
            add(
                "association",
                item.get("natural_language"),
                metric=drivers.get("target"),
                magnitude_pct=abs(float(item.get("correlation") or 0.0)) * 100.0,
                confidence="medium",
                evidence=item,
                source="driver_analysis",
                business_relevance=0.78,
            )

        for item in hypotheses[:3]:
            add(
                "hypothesis",
                item.get("hypothesis"),
                metric=primary,
                confidence=item.get("confidence", "low"),
                evidence={"basis": item.get("basis")},
                source="existing_hypothesis_engine",
                business_relevance=0.72,
            )

        if encoded_prediction and encoded_prediction.get("valid"):
            metrics = encoded_prediction.get("metrics") or {}
            prediction_strength = metrics.get("accuracy_like_pct") or metrics.get("accuracy_pct")
            add(
                "prediction",
                encoded_prediction.get("natural_language"),
                metric=encoded_prediction.get("target"),
                magnitude_pct=prediction_strength,
                confidence=encoded_prediction.get("reliability", "medium"),
                evidence={
                    "model": encoded_prediction.get("model"),
                    "metrics": metrics,
                    "top_features": (encoded_prediction.get("feature_importance") or [])[:5],
                },
                source="encoded_prediction",
                business_relevance=0.82,
            )

        return self._dedupe(candidates)[:7]

    def _dedupe(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)
        kept = []
        seen_keys = set()
        for item in candidates:
            tokens = [t for t in _tokens(item.get("statement")) if len(t) > 2]
            signature = tuple(sorted(Counter(tokens).keys())[:8])
            simple_key = (item.get("type"), item.get("metric"), signature)
            if simple_key in seen_keys:
                continue
            duplicate = False
            item_tokens = set(tokens)
            for existing in kept:
                existing_tokens = _tokens(existing.get("statement"))
                union = item_tokens | existing_tokens
                similarity = len(item_tokens & existing_tokens) / len(union) if union else 0.0
                if similarity >= 0.72 and item.get("metric") == existing.get("metric"):
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_keys.add(simple_key)
            kept.append(item)
        return kept

    def evidence_model(
        self,
        profile: dict[str, Any],
        period: dict[str, Any] | None,
        contributions: list[dict[str, Any]],
        drilldown: dict[str, Any] | None,
        retail: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        drivers: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        facts = []
        associations = []
        hypothesis_items = []
        next_checks = []

        if period:
            facts.append(
                {
                    "statement": period.get("natural_language"),
                    "evidence": {
                        "previous": period.get("previous_value"),
                        "current": period.get("current_value"),
                        "pct_change": period.get("pct_change"),
                    },
                }
            )
        for item in contributions[:2]:
            facts.append({"statement": item.get("natural_language"), "evidence": item.get("contributors", [])[:3]})
        if drilldown:
            facts.append({"statement": drilldown.get("natural_language"), "evidence": drilldown.get("path")})
        for item in anomalies[:2]:
            facts.append({"statement": item.get("natural_language"), "evidence": item})

        for item in (drivers.get("associations") or [])[:3]:
            associations.append(
                {
                    "statement": item.get("natural_language"),
                    "evidence": {
                        "correlation": item.get("correlation"),
                        "feature": item.get("feature"),
                    },
                    "causation_warning": "Association does not prove causation.",
                }
            )
        for item in retail:
            if item.get("type") == "discount_effectiveness":
                associations.append(
                    {
                        "statement": item.get("natural_language"),
                        "evidence": item,
                        "causation_warning": "Discount association does not prove that discount caused the outcome.",
                    }
                )

        for item in hypotheses[:3]:
            hypothesis_items.append(
                {
                    "statement": item.get("hypothesis"),
                    "basis": item.get("basis"),
                    "confidence": item.get("confidence"),
                }
            )

        if anomalies:
            next_checks.append("Verify the highest-severity unusual rows against the source records and recent business events.")
        if drilldown:
            next_checks.append("Compare the highlighted drill-down path with an unaffected segment before treating it as a root cause.")
        if associations:
            next_checks.append("Test the strongest association with a controlled segment or period comparison before making a causal claim.")
        if not next_checks:
            next_checks.append("Collect another comparable period or segment before making a high-impact decision.")

        return {
            "facts": facts[:6],
            "associations": associations[:5],
            "hypotheses": hypothesis_items,
            "next_checks": next_checks[:4],
            "top_ranked_insights": findings[:5],
            "source_context": {
                "question": profile.get("question"),
                "measure": profile.get("primary_measure"),
                "dimensions": profile.get("dimensions"),
                "filters": profile.get("filters"),
            },
        }
