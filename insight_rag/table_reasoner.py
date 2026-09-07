"""Deterministic helpers for numeric reasoning over retrieved Markdown tables."""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from langchain_core.documents import Document

_NUMERIC_INTENT = (
    "total", "sum", "average", "avg", "mean", "highest", "lowest", "maximum", "minimum",
    "max", "min", "top", "bottom", "compare", "difference", "percent", "percentage",
    "how many", "count", "above", "below", "over", "under", "greater than", "less than",
)
_THRESHOLD_RE = re.compile(
    r"\b(above|over|greater\s+than|more\s+than|at\s+least|below|under|less\s+than|at\s+most)\s*"
    r"([$£€₹]?\s*[-+]?\d[\d,]*(?:\.\d+)?%?)",
    re.IGNORECASE,
)


def has_numeric_table_intent(question: str) -> bool:
    lowered = (question or "").lower()
    return any(word in lowered for word in _NUMERIC_INTENT)


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    placeholder = "__ARIA_PIPE__"
    line = line.replace("\\|", placeholder)
    return [cell.strip().replace(placeholder, "|") for cell in line.split("|")]


def markdown_to_frame(text: str) -> pd.DataFrame | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = _split_markdown_row(lines[0])
    if not header:
        return None
    rows = []
    for line in lines[2:]:
        row = _split_markdown_row(line)
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        rows.append(row[: len(header)])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def _to_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"[$£€₹,%]", "", text)
    text = text.replace(",", "").strip()
    try:
        number = float(text)
        return -number if negative else number
    except ValueError:
        return None


def _normalise_column(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1
    }


def _numeric_series(frame: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for column in frame.columns:
        numeric = frame[column].map(_to_number)
        valid = numeric.dropna()
        if valid.empty:
            continue
        # Treat a column as numeric only when at least half its non-empty rows can
        # be parsed as numbers. This avoids summing IDs/descriptions accidentally.
        non_empty = frame[column].astype(str).str.strip().ne("").sum()
        if len(valid) < max(1, int(non_empty * 0.5)):
            continue
        result[str(column)] = numeric
    return result


def _relevant_numeric_columns(question: str, numeric: dict[str, pd.Series]) -> list[str]:
    if not numeric:
        return []
    q_tokens = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    ranked = []
    for column in numeric:
        tokens = _normalise_column(column)
        overlap = len(tokens & q_tokens)
        ranked.append((overlap, column))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked and ranked[0][0] > 0:
        best = ranked[0][0]
        return [column for score, column in ranked if score == best]
    return list(numeric)


def _label_columns(frame: pd.DataFrame, numeric_columns: set[str]) -> list[str]:
    labels = [str(column) for column in frame.columns if str(column) not in numeric_columns]
    return labels[:3]


def _row_descriptor(frame: pd.DataFrame, index, label_columns: list[str]) -> str:
    parts = []
    for column in label_columns:
        value = str(frame.loc[index, column]).strip()
        if value and value.lower() != "nan":
            parts.append(f"{column}={value}")
    return ", ".join(parts) if parts else f"row {int(index) + 1}"


def _threshold_matches(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    operator: str,
    threshold: float,
    label_columns: list[str],
    column: str,
) -> list[str]:
    op = operator.lower()
    if op in {"above", "over", "greater than", "more than"}:
        mask = series > threshold
    elif op == "at least":
        mask = series >= threshold
    elif op in {"below", "under", "less than"}:
        mask = series < threshold
    else:
        mask = series <= threshold

    matches = []
    for index in frame.index[mask.fillna(False)]:
        descriptor = _row_descriptor(frame, index, label_columns)
        value = series.loc[index]
        matches.append(f"{descriptor}; {column}={float(value):g}")
        if len(matches) >= 10:
            break
    return matches


def build_table_facts(question: str, docs: list[Document]) -> str:
    """Create deterministic facts from retrieved table rows.

    The LLM receives these facts as supplemental evidence. Calculations are done
    locally with pandas so totals, averages, extrema, threshold filters, and row
    labels do not depend on model arithmetic.
    """
    if not has_numeric_table_intent(question):
        return ""

    grouped: dict[tuple, list[Document]] = defaultdict(list)
    for doc in docs:
        meta = doc.metadata
        if meta.get("content_type") != "table":
            continue
        key = (
            meta.get("document_id"),
            meta.get("filename"),
            meta.get("page"),
            meta.get("table_index"),
        )
        grouped[key].append(doc)

    lowered = (question or "").lower()
    wants_sum = any(word in lowered for word in ("total", "sum"))
    wants_mean = any(word in lowered for word in ("average", "avg", "mean"))
    wants_max = any(word in lowered for word in ("highest", "maximum", "max", "top"))
    wants_min = any(word in lowered for word in ("lowest", "minimum", "min", "bottom"))
    wants_count = "how many" in lowered or "count" in lowered
    wants_compare = any(word in lowered for word in ("compare", "difference", "versus", " vs "))
    threshold_match = _THRESHOLD_RE.search(question or "")

    sections = []
    for key, table_docs in grouped.items():
        frames = []
        for doc in table_docs:
            frame = markdown_to_frame(doc.page_content)
            if frame is not None:
                frames.append(frame)
        if not frames:
            continue

        frame = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        numeric = _numeric_series(frame)
        relevant_columns = _relevant_numeric_columns(question, numeric)
        if not relevant_columns:
            continue

        label_columns = _label_columns(frame, set(numeric))
        filename, page, table_index = key[1], key[2], key[3]
        facts = []

        for column in relevant_columns:
            series = numeric[column]
            valid = series.dropna().astype(float)
            if valid.empty:
                continue

            # For broad comparison questions provide compact descriptive facts;
            # otherwise emit only the operations the user actually requested.
            if wants_sum or wants_compare:
                facts.append(f"{column} sum={float(valid.sum()):g}")
            if wants_mean or wants_compare:
                facts.append(f"{column} mean={float(valid.mean()):g}")
            if wants_count:
                facts.append(f"{column} count={int(valid.count())}")

            if wants_max or wants_compare:
                max_index = valid.idxmax()
                facts.append(
                    f"{column} max={float(valid.loc[max_index]):g} at "
                    f"{_row_descriptor(frame, max_index, label_columns)}"
                )
            if wants_min or wants_compare:
                min_index = valid.idxmin()
                facts.append(
                    f"{column} min={float(valid.loc[min_index]):g} at "
                    f"{_row_descriptor(frame, min_index, label_columns)}"
                )

            if threshold_match:
                operator = " ".join(threshold_match.group(1).lower().split())
                threshold = _to_number(threshold_match.group(2))
                if threshold is not None:
                    matches = _threshold_matches(
                        frame,
                        series,
                        operator=operator,
                        threshold=float(threshold),
                        label_columns=label_columns,
                        column=column,
                    )
                    if matches:
                        facts.append(
                            f"{column} rows {operator} {float(threshold):g}: " + " | ".join(matches)
                        )
                    else:
                        facts.append(f"{column} rows {operator} {float(threshold):g}: none")

        if facts:
            sections.append(
                f"Computed deterministically from {filename}, page {page}, table {table_index}:\n"
                + "\n".join(f"- {fact}" for fact in facts)
            )

    return "\n\n".join(sections)
