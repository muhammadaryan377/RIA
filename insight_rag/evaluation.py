"""Offline evaluation primitives for ARIA PDF-RAG.

The evaluator scores captured chat results against a small human-authored gold
contract.  It does not call an LLM, so benchmark results are reproducible and can
run in CI or locally without API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


_ABSTENTION_STATUSES = {
    "insufficient",
    "verification_failed",
    "verification_unavailable",
    "no_documents",
    "rewrite_unavailable",
    "scope_changed",
    "incomplete_comparison",
    "service_unavailable",
    "model_unavailable",
}


@dataclass(frozen=True)
class EvalExpectation:
    should_answer: bool = True
    expected_pages: tuple[int, ...] = ()
    expected_document_ids: tuple[str, ...] = ()
    expected_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()


def _normalise(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _source_pairs(result: dict) -> set[tuple[str, int]]:
    values: set[tuple[str, int]] = set()
    for source in result.get("sources") or []:
        document_id = str(source.get("document_id") or "")
        try:
            page = int(source.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if document_id and page > 0:
            values.add((document_id, page))
    return values


def _page_set(result: dict) -> set[int]:
    pages = set()
    for source in result.get("sources") or []:
        try:
            page = int(source.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            pages.add(page)
    return pages


def is_abstention(result: dict) -> bool:
    status = str(result.get("evidence_status") or "")
    if status in _ABSTENTION_STATUSES:
        return True
    return not bool((result.get("answer") or "").strip())


def score_case(result: dict, expected: EvalExpectation | dict) -> dict:
    """Score one captured result against deterministic expectations."""
    if isinstance(expected, dict):
        expected = EvalExpectation(
            should_answer=bool(expected.get("should_answer", True)),
            expected_pages=tuple(int(v) for v in expected.get("expected_pages", []) or []),
            expected_document_ids=tuple(str(v) for v in expected.get("expected_document_ids", []) or []),
            expected_phrases=tuple(str(v) for v in expected.get("expected_phrases", []) or []),
            forbidden_phrases=tuple(str(v) for v in expected.get("forbidden_phrases", []) or []),
        )

    answer = _normalise(str(result.get("answer") or ""))
    abstained = is_abstention(result)
    answered = not abstained
    behavior_correct = answered == expected.should_answer

    expected_pages = set(expected.expected_pages)
    actual_pages = _page_set(result)
    if expected_pages:
        page_recall = len(expected_pages & actual_pages) / len(expected_pages)
        page_precision = len(expected_pages & actual_pages) / max(1, len(actual_pages))
    else:
        page_recall = None
        page_precision = None

    expected_docs = set(expected.expected_document_ids)
    actual_docs = {doc_id for doc_id, _ in _source_pairs(result)}
    if expected_docs:
        document_recall = len(expected_docs & actual_docs) / len(expected_docs)
    else:
        document_recall = None

    phrase_hits = [phrase for phrase in expected.expected_phrases if _normalise(phrase) in answer]
    phrase_recall = (
        len(phrase_hits) / len(expected.expected_phrases)
        if expected.expected_phrases else None
    )
    forbidden_hits = [phrase for phrase in expected.forbidden_phrases if _normalise(phrase) in answer]

    citation_status = str(result.get("citation_status") or "")
    citation_ok = (
        citation_status == "cited"
        if answered and bool(result.get("sources"))
        else citation_status in {"not_applicable", "cited", ""}
    )

    return {
        "behavior_correct": behavior_correct,
        "answered": answered,
        "abstained": abstained,
        "citation_ok": citation_ok,
        "page_recall": None if page_recall is None else round(page_recall, 4),
        "page_precision": None if page_precision is None else round(page_precision, 4),
        "document_recall": None if document_recall is None else round(document_recall, 4),
        "phrase_recall": None if phrase_recall is None else round(phrase_recall, 4),
        "forbidden_phrase_hits": forbidden_hits,
        "latency_ms": float(result.get("latency_ms") or 0.0),
        "evidence_status": result.get("evidence_status"),
    }


def _mean_present(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), 4) if values else None


def aggregate_scores(rows: list[dict]) -> dict:
    """Aggregate per-case scores into a compact benchmark report."""
    rows = list(rows or [])
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "behavior_accuracy": round(mean(1.0 if row.get("behavior_correct") else 0.0 for row in rows), 4),
        "citation_accuracy": round(mean(1.0 if row.get("citation_ok") else 0.0 for row in rows), 4),
        "page_recall": _mean_present(rows, "page_recall"),
        "page_precision": _mean_present(rows, "page_precision"),
        "document_recall": _mean_present(rows, "document_recall"),
        "phrase_recall": _mean_present(rows, "phrase_recall"),
        "forbidden_failure_rate": round(
            mean(1.0 if row.get("forbidden_phrase_hits") else 0.0 for row in rows), 4
        ),
        "average_latency_ms": round(mean(float(row.get("latency_ms") or 0.0) for row in rows), 2),
    }
