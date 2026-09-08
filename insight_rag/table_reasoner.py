"""Deterministic numeric reasoning over retrieved Markdown tables.

Natural-language intent is resolved by the semantic router. This module receives
explicit operations and performs only auditable pandas calculations.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict

import pandas as pd
from langchain_core.documents import Document

from .retrieval import deduplicate_chunks


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
    if not header or len(set(header)) != len(header):
        return None
    separator = _split_markdown_row(lines[1])
    if len(separator) != len(header) or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        return None
    rows = []
    for line in lines[2:]:
        row = _split_markdown_row(line)
        if len(row) != len(header):
            return None
        rows.append(row)
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
    text = re.sub(r"[$£€₹%]", "", text).strip()
    if "," in text and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        return None
    text = text.replace(",", "").strip()
    try:
        number = float(text)
        return (-number if negative else number) if math.isfinite(number) else None
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
    sources: list[dict] | None = None,
) -> str:
    """Create deterministic facts from retrieved table rows.

    ``operations`` comes from the validated semantic route, so this function
    never guesses numeric intent from hard-coded user phrases.
    """
    op_set = {str(operation).upper() for operation in operations}
    if not op_set and filter_operator == "NONE":
        return ""

    grouped: dict[tuple, list[Document]] = defaultdict(list)
    for doc in deduplicate_chunks(docs):
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
        table_docs.sort(key=lambda doc: doc.metadata.get("table_chunk_index", 0))
        expected_counts = {doc.metadata.get("table_chunk_count") for doc in table_docs
                           if doc.metadata.get("table_chunk_count") is not None}
        complete = not expected_counts or (len(expected_counts) == 1 and
                    {doc.metadata.get("table_chunk_index") for doc in table_docs} == set(range(next(iter(expected_counts)))))
        if not complete:
            sections.append(f"No calculation: incomplete table from {key[1]}, page {key[2]}, table {key[3]}.")
            continue
        malformed = False
        for doc in table_docs:
            frame = markdown_to_frame(doc.page_content)
            if frame is not None:
                frames.append(frame)
            else:
                malformed = True
        if malformed or not frames or any(list(f.columns) != list(frames[0].columns) for f in frames):
            continue

        frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
        numeric = _numeric_series(frame)
        relevant_columns = _relevant_numeric_columns(question, numeric)
        if not relevant_columns:
            continue

        label_columns = _label_columns(frame, set(numeric))
        filename, page, table_index = key[1], key[2], key[3]
        facts = []
        has_total_row = any(
            frame[column].astype(str).str.strip().str.casefold().isin(
                {"total", "subtotal", "grand total", "sub-total"}
            ).any() for column in label_columns
        )
        if has_total_row and op_set & {"SUM", "MEAN", "COUNT", "COMPARE"}:
            sections.append(f"No aggregate calculation: {filename}, page {page}, table {table_index} contains total/subtotal rows; isolate detail rows first.")
            continue

        for column in relevant_columns:
            series = numeric[column]
            valid = series.dropna().astype(float)
            if valid.empty:
                continue
            missing = int(series.isna().sum())
            if missing:
                facts.append(f"{column}: {missing} missing/unparseable cells; aggregates withheld")
                continue
            raw_values = frame[column].astype(str).str.strip()
            percent = raw_values.str.endswith("%")
            if percent.any() and not percent.all():
                facts.append(f"{column}: mixed percent and plain values; aggregates withheld")
                continue
            currencies = {symbol for value in raw_values for symbol in "$£€₹" if symbol in value}
            if len(currencies) > 1:
                facts.append(f"{column}: mixed currencies; aggregates withheld")
                continue
            if percent.all() and "SUM" in op_set:
                facts.append(f"{column}: percentage sum withheld (denominator/weight required)")
            if ("SUM" in op_set or "COMPARE" in op_set) and not percent.all():
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
            labels = [str(source["source_id"]) for source in (sources or [])
                      if (source.get("document_id"), source.get("filename"), source.get("page"), source.get("table_index")) == key]
            provenance = " ".join(f"[{label}]" for label in labels)
            facts.insert(0, f"Calculation scope: {len(frame)} extracted rows from this table only. {provenance}")
            if any(frame[column].astype(str).str.strip().str.endswith("%").all() for column in relevant_columns):
                facts.insert(1, "Percentage values use percentage-point units; MEAN is an unweighted row mean, not an overall rate.")
            sections.append(
                f"Computed deterministically from {filename}, page {page}, table {table_index}:\n"
                + "\n".join(f"- {fact}" for fact in facts)
            )

    return "\n\n".join(sections)
