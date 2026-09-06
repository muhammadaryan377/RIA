"""Deterministic helpers for numeric reasoning over retrieved Markdown tables."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import pandas as pd
from langchain_core.documents import Document

_NUMERIC_INTENT = (
    "total", "sum", "average", "avg", "mean", "highest", "lowest", "maximum", "minimum",
    "max", "min", "compare", "difference", "percent", "percentage", "how many", "count",
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
    # PDF table cells rarely contain literal pipes. The parser escapes them as
    # \|; protect those before splitting and restore afterward.
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
    data_lines = lines[2:]
    rows = []
    for line in data_lines:
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
        if negative:
            number = -number
        return number
    except ValueError:
        return None


def build_table_facts(question: str, docs: list[Document]) -> str:
    """Create deterministic numeric facts for retrieved table evidence.

    Facts are supplementary evidence for GPT-OSS; they never replace citations.
    Only generated for clearly numeric/comparative user questions.
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

    sections = []
    for key, table_docs in grouped.items():
        frames = []
        for doc in table_docs:
            frame = markdown_to_frame(doc.page_content)
            if frame is not None:
                frames.append(frame)
        if not frames:
            continue

        # Chunked tables repeat their header, but DataFrames share columns; concat
        # reconstructs the visible rows for deterministic calculations.
        frame = pd.concat(frames, ignore_index=True).drop_duplicates()
        filename, page, table_index = key[1], key[2], key[3]
        facts = []
        for column in frame.columns:
            numeric = frame[column].map(_to_number)
            valid = numeric.dropna()
            if valid.empty or len(valid) < max(1, int(len(frame) * 0.5)):
                continue
            values = valid.astype(float)
            total = float(values.sum())
            mean = float(values.mean())
            minimum = float(values.min())
            maximum = float(values.max())
            facts.append(
                f"{column}: count={len(values)}, sum={total:g}, mean={mean:g}, min={minimum:g}, max={maximum:g}"
            )

        if facts:
            sections.append(
                f"Computed from {filename}, page {page}, table {table_index}:\n" + "\n".join(f"- {fact}" for fact in facts)
            )

    return "\n\n".join(sections)
