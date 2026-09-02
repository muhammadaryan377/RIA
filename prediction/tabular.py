"""Deterministic mixed-table prediction for ARIA's Insight Agent.

The Goal Agent remains responsible for selecting/joining data. This module only
works on the resulting ``processed_data`` table inside the Insight layer.

Capabilities
------------
- regression and classification target inference,
- numeric median imputation + standard scaling,
- categorical most-frequent imputation + one-hot encoding,
- date expansion to year/month/day/day-of-week,
- identifier/high-cardinality leakage protection,
- held-out evaluation, reliability grading and feature importance.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("aria.prediction.tabular")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - graceful runtime fallback
    SKLEARN_AVAILABLE = False


_ID_EXACT = {"id", "key", "code", "row_number", "rownumber"}
_DATE_HINTS = ("date", "time", "timestamp", "year", "month", "day", "week", "quarter")
_PREDICTION_WORDS = ("predict", "prediction", "forecast", "estimate", "expected", "future")


def _is_id_like(name: str) -> bool:
    low = str(name).strip().lower()
    return (
        low in _ID_EXACT
        or low.endswith("_id")
        or low.endswith("_key")
        or (low.endswith("id") and len(low) <= 18)
    )


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _make_one_hot_encoder():
    """Support both new and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class PreparedFeatures:
    frame: pd.DataFrame
    numeric: list[str]
    categorical: list[str]
    derived_from_dates: list[str]
    dropped: list[dict[str, str]]


class TabularPredictor:
    """Automatic supervised prediction with safe mixed-table encoding."""

    def __init__(self, min_rows: int = 20, test_size: float = 0.25, random_state: int = 42):
        self.min_rows = int(min_rows)
        self.test_size = float(test_size)
        self.random_state = int(random_state)

    def predict(self, processed_data: dict[str, Any] | None, target: str | None = None) -> dict[str, Any]:
        processed_data = processed_data or {}
        rows = processed_data.get("data") or []

        if not SKLEARN_AVAILABLE:
            return self._empty(
                "scikit-learn is not installed; run `pip install -r requirements.txt`.",
                code="dependency_missing",
            )
        if len(rows) < self.min_rows:
            return self._empty(
                f"encoded prediction needs at least {self.min_rows} rows; received {len(rows)}.",
                code="insufficient_data",
            )

        df = pd.DataFrame(rows)
        if df.empty or len(df.columns) < 2:
            return self._empty("prediction needs at least two usable columns.", code="insufficient_features")

        target, target_source = self._choose_target(df, processed_data, target)
        if not target:
            return self._empty("no suitable prediction target could be identified.", code="no_target")

        task = self._infer_task(df[target], processed_data)
        work = df.copy()
        if task == "regression":
            work[target] = pd.to_numeric(work[target], errors="coerce")
        work = work[work[target].notna()].reset_index(drop=True)

        if len(work) < self.min_rows:
            return self._empty(
                f"only {len(work)} rows have a usable target value for '{target}'.",
                code="insufficient_target_rows",
                target=target,
            )

        y = work[target]
        if y.nunique(dropna=True) < 2:
            return self._empty(
                f"target '{target}' has no variation, so a model cannot learn from it.",
                code="constant_target",
                target=target,
            )

        prepared = self._prepare_features(work.drop(columns=[target]))
        X = prepared.frame
        if X.empty or not (prepared.numeric or prepared.categorical):
            return self._empty(
                "no safe predictive features remained after removing IDs, constants and unsuitable text columns.",
                code="no_features",
                target=target,
            )

        test_count = max(4, int(round(len(X) * self.test_size)))
        test_count = min(test_count, max(4, len(X) // 3))
        if len(X) - test_count < 8:
            return self._empty(
                "not enough rows remain for a reliable train/test split.",
                code="insufficient_split",
                target=target,
            )

        stratify = None
        if task == "classification":
            counts = y.value_counts()
            if 2 <= len(counts) <= 20 and counts.min() >= 2:
                stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_count,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_count,
                random_state=self.random_state,
                stratify=None,
            )

        preprocessor = self._build_preprocessor(prepared.numeric, prepared.categorical)
        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=220,
                random_state=self.random_state,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=220,
                random_state=self.random_state,
                min_samples_leaf=1,
                class_weight="balanced" if y.nunique() <= 20 else None,
                n_jobs=-1,
            )

        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        except Exception as exc:
            log.warning("Encoded prediction model failed: %s", exc)
            return self._empty(f"model training failed: {exc}", code="model_failure", target=target)

        metrics, valid, reliability = self._evaluate(task, y_test, y_pred)
        feature_importance = self._feature_importance(pipe, limit=12)
        samples = self._prediction_samples(task, y_test, y_pred, limit=12)

        return {
            "valid": bool(valid),
            "mode": "encoded_tabular",
            "task": task,
            "target": target,
            "target_source": target_source,
            "model": model.__class__.__name__,
            "reliability": reliability,
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "metrics": metrics,
            "encoding": {
                "method": "ColumnTransformer",
                "numeric": {
                    "columns": prepared.numeric,
                    "steps": ["median imputation", "standard scaling"],
                },
                "categorical": {
                    "columns": prepared.categorical,
                    "steps": ["most-frequent imputation", "one-hot encoding"],
                    "unknown_categories": "ignored safely at prediction time",
                },
                "date_features_created": prepared.derived_from_dates,
                "dropped_columns": prepared.dropped,
            },
            "feature_importance": feature_importance,
            "prediction_samples": samples,
            "natural_language": self._plain_summary(task, target, metrics, reliability, feature_importance),
            "warning": None if valid else "Model quality is weak; treat this as exploratory rather than a confident forecast.",
        }

    def _choose_target(self, df: pd.DataFrame, processed_data: dict[str, Any], target: str | None):
        explicit = target or processed_data.get("prediction_target")
        if explicit in df.columns and not _is_id_like(explicit):
            return str(explicit), "explicit"

        goal = str(processed_data.get("user_goal") or "").strip().lower()
        normalized_goal = goal.replace("_", " ")
        prediction_goal = any(word in goal for word in _PREDICTION_WORDS)

        measure_cols: list[str] = []
        for col in df.columns:
            if _is_id_like(col):
                continue
            sample = df[col].dropna()
            if sample.empty:
                continue
            if pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.8:
                measure_cols.append(str(col))

        mentioned = [c for c in measure_cols if c.lower().replace("_", " ") in normalized_goal]
        if mentioned:
            return mentioned[-1], "mentioned_in_goal"

        if prediction_goal:
            for col in df.columns:
                if _is_id_like(col):
                    continue
                name = str(col).lower().replace("_", " ")
                if name and name in normalized_goal:
                    return str(col), "mentioned_in_goal"

        if measure_cols:
            return measure_cols[-1], "automatic_primary_measure"
        return None, "none"

    def _infer_task(self, series: pd.Series, processed_data: dict[str, Any]) -> str:
        explicit = str(processed_data.get("prediction_task") or "").lower()
        if explicit in {"classification", "regression"}:
            return explicit
        sample = series.dropna()
        numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean() if len(sample) else 0.0
        return "regression" if numeric_ratio >= 0.9 else "classification"

    def _looks_datetime(self, name: str, series: pd.Series) -> bool:
        low = str(name).lower()
        hinted = any(h in low for h in _DATE_HINTS)
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not hinted:
            text = series.dropna().astype(str).head(25)
            hinted = bool(len(text)) and text.str.contains(r"[-/:T]", regex=True).mean() >= 0.6
        if not hinted:
            return False
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.notna().mean() >= 0.8

    def _prepare_features(self, X: pd.DataFrame) -> PreparedFeatures:
        frame = X.copy()
        dropped: list[dict[str, str]] = []
        derived: list[str] = []

        for col in list(frame.columns):
            if _is_id_like(col):
                frame = frame.drop(columns=[col])
                dropped.append({"column": str(col), "reason": "identifier/key leakage risk"})
                continue
            if frame[col].nunique(dropna=True) <= 1:
                frame = frame.drop(columns=[col])
                dropped.append({"column": str(col), "reason": "constant column"})

        for col in list(frame.columns):
            if self._looks_datetime(col, frame[col]):
                parsed = pd.to_datetime(frame[col], errors="coerce")
                for suffix, values in (
                    ("year", parsed.dt.year),
                    ("month", parsed.dt.month),
                    ("day", parsed.dt.day),
                    ("dayofweek", parsed.dt.dayofweek),
                ):
                    new_col = f"{col}__{suffix}"
                    frame[new_col] = values
                    derived.append(new_col)
                frame = frame.drop(columns=[col])
                dropped.append({"column": str(col), "reason": "replaced by date components"})

        numeric: list[str] = []
        categorical: list[str] = []
        for col in list(frame.columns):
            sample = frame[col].dropna()
            if sample.empty:
                frame = frame.drop(columns=[col])
                dropped.append({"column": str(col), "reason": "all values missing"})
                continue

            coerced = pd.to_numeric(sample, errors="coerce")
            if coerced.notna().mean() >= 0.8:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
                numeric.append(str(col))
                continue

            nunique = int(sample.nunique())
            unique_ratio = nunique / max(1, len(sample))
            if nunique > 100 or (nunique > 30 and unique_ratio > 0.55):
                frame = frame.drop(columns=[col])
                dropped.append({"column": str(col), "reason": "very high-cardinality text"})
                continue

            frame[col] = frame[col].astype("object")
            categorical.append(str(col))

        return PreparedFeatures(
            frame=frame,
            numeric=numeric,
            categorical=categorical,
            derived_from_dates=derived,
            dropped=dropped,
        )

    def _build_preprocessor(self, numeric: list[str], categorical: list[str]):
        transformers = []
        if numeric:
            numeric_pipe = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            transformers.append(("num", numeric_pipe, numeric))
        if categorical:
            categorical_pipe = Pipeline(
                [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot_encoder())]
            )
            transformers.append(("cat", categorical_pipe, categorical))
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)

    def _evaluate(self, task: str, actual, predicted):
        if task == "regression":
            actual_f = np.asarray(actual, dtype=float)
            pred_f = np.asarray(predicted, dtype=float)
            mae = float(mean_absolute_error(actual_f, pred_f))
            rmse = float(np.sqrt(mean_squared_error(actual_f, pred_f)))
            r2 = float(r2_score(actual_f, pred_f)) if len(actual_f) >= 2 else float("nan")
            nonzero = np.abs(actual_f) > 1e-12
            mape = (
                float(np.mean(np.abs((actual_f[nonzero] - pred_f[nonzero]) / actual_f[nonzero])) * 100)
                if nonzero.any()
                else None
            )
            accuracy_like = max(0.0, 100.0 - mape) if mape is not None else None
            valid = (math.isfinite(r2) and r2 >= 0.30) or (accuracy_like is not None and accuracy_like >= 70.0)
            reliability = "high" if math.isfinite(r2) and r2 >= 0.65 else ("medium" if valid else "low")
            return {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4) if math.isfinite(r2) else None,
                "mape_pct": round(mape, 2) if mape is not None and math.isfinite(mape) else None,
                "accuracy_like_pct": round(accuracy_like, 2) if accuracy_like is not None else None,
            }, valid, reliability

        acc = float(accuracy_score(actual, predicted))
        f1 = float(f1_score(actual, predicted, average="weighted", zero_division=0))
        valid = acc >= 0.65
        reliability = "high" if acc >= 0.85 else ("medium" if valid else "low")
        return {
            "accuracy_pct": round(acc * 100.0, 2),
            "weighted_f1_pct": round(f1 * 100.0, 2),
        }, valid, reliability

    def _feature_importance(self, pipe: Pipeline, limit: int = 12):
        try:
            pre = pipe.named_steps["preprocess"]
            model = pipe.named_steps["model"]
            names = list(pre.get_feature_names_out())
            values = list(model.feature_importances_)
            pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:limit]
            return [
                {
                    "feature": re.sub(r"^(num|cat)__", "", str(name)),
                    "importance": round(float(value), 4),
                }
                for name, value in pairs
            ]
        except Exception:
            return []

    def _prediction_samples(self, task: str, actual, predicted, limit: int = 12):
        out = []
        for a, p in zip(list(actual)[:limit], list(predicted)[:limit]):
            if task == "regression":
                actual_f = _safe_float(a)
                predicted_f = _safe_float(p)
                out.append(
                    {
                        "actual": round(actual_f, 4) if actual_f is not None else None,
                        "predicted": round(predicted_f, 4) if predicted_f is not None else None,
                    }
                )
            else:
                out.append({"actual": str(a), "predicted": str(p)})
        return out

    def _plain_summary(
        self,
        task: str,
        target: str,
        metrics: dict[str, Any],
        reliability: str,
        importance: list[dict[str, Any]],
    ) -> str:
        top = importance[0]["feature"] if importance else None
        driver = f" The strongest predictive signal is {top}." if top else ""
        if task == "regression":
            accuracy = metrics.get("accuracy_like_pct")
            r2 = metrics.get("r2")
            score_text = (
                f" Estimated held-out accuracy is about {accuracy:.1f}%."
                if accuracy is not None
                else (f" The held-out R2 score is {r2:.2f}." if r2 is not None else "")
            )
            return (
                f"ARIA prepared the mixed table and trained a model to estimate {target}. "
                f"Reliability is {reliability}.{score_text}{driver}"
            ).strip()
        acc = metrics.get("accuracy_pct")
        acc_text = f" Held-out accuracy is {acc:.1f}%." if acc is not None else ""
        return (
            f"ARIA prepared the mixed table and trained a classifier for {target}. "
            f"Reliability is {reliability}.{acc_text}{driver}"
        ).strip()

    def _empty(self, reason: str, code: str, target: str | None = None):
        return {
            "valid": False,
            "mode": "encoded_tabular",
            "task": None,
            "target": target,
            "target_source": None,
            "model": None,
            "reliability": "not_available",
            "rows_used": 0,
            "train_rows": 0,
            "test_rows": 0,
            "metrics": {},
            "encoding": {},
            "feature_importance": [],
            "prediction_samples": [],
            "natural_language": reason,
            "warning": reason,
            "reason_code": code,
        }
