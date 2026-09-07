"""Deterministic numeric reasoning over retrieved Markdown tables.

Natural-language intent is resolved by the semantic router. This module receives
explicit operations and performs only auditable pandas calculations.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from langchain_core.documents import Document


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
    return [str(column) for column in frame.columns if str(column) not in numeric_columns][:3]


def _row_descriptor(frame: pd.DataFrame, index, label_columns: list[str]) -> str:
    parts = []
    for column in label_columns:
        value = str(frame.loc[index, column]).strip()
        if value and value.lower() != "nan":
            parts.append(f"{column}={value}")
    return ", ".join(parts) if parts else f"row {int(index) + 1}"


def _filter_matches(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    operator: str,
    threshold: float,
    label_columns: list[str],
    column: str,
) -> list[str]:
    operator = operator.upper()
    if operator == "GT":
        mask = series > threshold
    elif operator == "GTE":
        mask = series >= threshold
    elif operator == "LT":
        mask = series < threshold
    elif operator == "LTE":
        mask = series <= threshold
    else:
        mask = series == threshold

    matches = []
    for index in frame.index[mask.fillna(False)]:
        descriptor = _row_descriptor(frame, index, label_columns)
        value = series.loc[index]
        matches.append(f"{descriptor}; {column}={float(value):g}")
        if len(matches) >= 20:
            break
    return matches


def _rank_rows(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    label_columns: list[str],
    column: str,
    top_n: int,
) -> list[str]:
    valid = series.dropna().astype(float).sort_values(ascending=False).head(max(1, min(top_n, 20)))
    return [
        f"{_row_descriptor(frame, index, label_columns)}; {column}={float(value):g}"
        for index, value in valid.items()
    ]


def build_table_facts(
    question: str,
    docs: list[Document],
    *,
    operations: tuple[str, ...] | list[str] = (),
    filter_operator: str = "NONE",
    filter_value: float | None = None,
    top_n: int | None = None,
) -> str:
    """Create deterministic facts from retrieved table rows.

    ``operations`` comes from the validated semantic route, so this function
    never guesses numeric intent from hard-coded user phrases.
    """
    op_set = {str(operation).upper() for operation in operations}
    if not op_set and filter_operator == "NONE":
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

            if "SUM" in op_set or "COMPARE" in op_set:
                facts.append(f"{column} sum={float(valid.sum()):g}")
            if "MEAN" in op_set or "COMPARE" in op_set:
                facts.append(f"{column} mean={float(valid.mean()):g}")
            if "COUNT" in op_set:
                facts.append(f"{column} count={int(valid.count())}")
            if "MAX" in op_set or "COMPARE" in op_set:
                max_index = valid.idxmax()
                facts.append(
                    f"{column} max={float(valid.loc[max_index]):g} at "
                    f"{_row_descriptor(frame, max_index, label_columns)}"
                )
            if "MIN" in op_set or "COMPARE" in op_set:
                min_index = valid.idxmin()
                facts.append(
                    f"{column} min={float(valid.loc[min_index]):g} at "
                    f"{_row_descriptor(frame, min_index, label_columns)}"
                )

            if "FILTER" in op_set and filter_operator != "NONE" and filter_value is not None:
                matches = _filter_matches(
                    frame,
                    series,
                    operator=filter_operator,
                    threshold=float(filter_value),
                    label_columns=label_columns,
                    column=column,
                )
                facts.append(
                    f"{column} filter {filter_operator} {float(filter_value):g}: "
                    + (" | ".join(matches) if matches else "none")
                )

            if "RANK" in op_set:
                ranked = _rank_rows(
                    frame,
                    series,
                    label_columns=label_columns,
                    column=column,
                    top_n=int(top_n or 5),
                )
                if ranked:
                    facts.append(f"{column} ranked rows: " + " | ".join(ranked))

        if facts:
            sections.append(
                f"Computed deterministically from {filename}, page {page}, table {table_index}:\n"
                + "\n".join(f"- {fact}" for fact in facts)
            )

    return "\n\n".join(sections)
