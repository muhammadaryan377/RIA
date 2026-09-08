"""Deterministic enterprise quality signals for ARIA PDF-RAG.

These helpers deliberately avoid model self-ratings.  They expose auditable
signals derived from ingestion, retrieval, citations and verification so the UI,
benchmarks and operations tooling can reason about answer quality without
pretending a heuristic score is a calibrated probability.
"""

from __future__ import annotations

import re
from collections import Counter


_INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior)\s+instructions?\b", re.I)),
    ("system_prompt_request", re.compile(r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b", re.I)),
    ("role_override", re.compile(r"\b(?:act|behave|pretend)\s+as\b", re.I)),
    ("prompt_exfiltration", re.compile(r"\b(?:reveal|show|print|repeat)\b.{0,40}\b(?:prompt|instructions?)\b", re.I)),
    ("instruction_override", re.compile(r"\b(?:follow|obey)\s+(?:these|the following)\s+instructions?\b", re.I)),
)


def scan_untrusted_instructions(text: str) -> list[str]:
    """Return stable labels for instruction-like text found inside a PDF chunk.

    A hit is a *risk signal*, not proof of malicious content.  PDF text remains
    evidence and is never executed as an instruction by the generation prompts.
    """
    value = text or ""
    return [label for label, pattern in _INSTRUCTION_PATTERNS if pattern.search(value)]


def assess_ingestion_quality(
    *,
    pages: int,
    text_chars: int,
    page_text_chars: list[int],
    tables: int,
    instruction_like_chunks: int,
) -> dict:
    """Create an auditable extraction-quality report for one ingested PDF."""
    pages = max(0, int(pages or 0))
    text_chars = max(0, int(text_chars or 0))
    page_text_chars = [max(0, int(value or 0)) for value in page_text_chars]
    empty_pages = sum(1 for value in page_text_chars if value < 20)
    text_pages = max(0, pages - empty_pages)
    coverage = (text_pages / pages) if pages else 0.0
    chars_per_page = (text_chars / pages) if pages else 0.0

    warnings: list[str] = []
    if coverage < 0.70:
        warnings.append("A significant share of pages has little extractable text; answers may miss image/scanned content.")
    elif coverage < 0.90:
        warnings.append("Some pages have little extractable text; page-level coverage is incomplete.")
    if chars_per_page < 120:
        warnings.append("Average extracted text per page is low.")
    if instruction_like_chunks:
        warnings.append(
            f"Detected instruction-like text in {instruction_like_chunks} chunk(s); it is treated as untrusted evidence, never as commands."
        )

    if coverage >= 0.95 and chars_per_page >= 300:
        grade = "high"
    elif coverage >= 0.80 and chars_per_page >= 120:
        grade = "medium"
    else:
        grade = "low"

    return {
        "grade": grade,
        "text_page_coverage": round(coverage, 4),
        "average_text_chars_per_page": round(chars_per_page, 1),
        "empty_or_low_text_pages": empty_pages,
        "table_count": max(0, int(tables or 0)),
        "instruction_like_chunks": max(0, int(instruction_like_chunks or 0)),
        "warnings": warnings,
    }


def _retrieval_consensus(sources: list[dict]) -> tuple[float | None, int]:
    votes = []
    with_metadata = 0
    for source in sources:
        value = source.get("retrieval_votes")
        if value is None:
            continue
        with_metadata += 1
        try:
            votes.append(max(0, int(value)))
        except (TypeError, ValueError):
            continue
    if not votes:
        return None, with_metadata
    # Two independent retrieval paths/subqueries agreeing is a strong signal;
    # cap at three so query decomposition cannot inflate the score indefinitely.
    normalised = [min(value, 3) / 3.0 for value in votes]
    return sum(normalised) / len(normalised), with_metadata


def grounding_quality(
    *,
    evidence_status: str,
    citation_status: str,
    sources: list[dict],
    requested_document_ids: list[str] | None = None,
    cross_document: bool = False,
) -> dict:
    """Return a deterministic quality index for an already-produced response.

    The score is an operational heuristic (0..1), not a probability that the
    answer is true.  A fail-closed verification result always scores zero.
    """
    evidence_status = str(evidence_status or "")
    citation_status = str(citation_status or "")
    sources = list(sources or [])

    if evidence_status not in {"supported", "carried_forward"}:
        return {
            "score": 0.0,
            "level": "none",
            "calibrated_probability": False,
            "signals": {"verified_grounding": False},
        }

    score = 0.45
    signals: dict[str, object] = {"verified_grounding": True}

    citation_ok = citation_status == "cited"
    if citation_ok:
        score += 0.20
    signals["citation_integrity"] = citation_ok

    source_count = len(sources)
    score += min(source_count, 4) / 4.0 * 0.10
    signals["source_count"] = source_count

    consensus, consensus_sources = _retrieval_consensus(sources)
    if consensus is None:
        # Direct page/summary paths intentionally bypass ranked retrieval. Give a
        # neutral partial credit rather than penalising deterministic local scope.
        score += 0.075 if sources else 0.0
        signals["retrieval_consensus"] = "not_applicable"
    else:
        score += 0.15 * consensus
        signals["retrieval_consensus"] = round(consensus, 4)
        signals["consensus_sources"] = consensus_sources

    requested = {str(value) for value in (requested_document_ids or []) if str(value)}
    represented = {str(source.get("document_id")) for source in sources if source.get("document_id")}
    if cross_document and requested:
        coverage = len(requested & represented) / len(requested)
        score += 0.10 * coverage
        signals["document_coverage"] = round(coverage, 4)
    else:
        score += 0.10 if sources else 0.0
        signals["document_coverage"] = "not_applicable"

    score = min(1.0, round(score, 4))
    level = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"
    return {
        "score": score,
        "level": level,
        "calibrated_probability": False,
        "signals": signals,
    }


def summarize_security_flags(chunks: list[dict]) -> dict:
    """Aggregate per-chunk risk labels for diagnostics or admin tooling."""
    counter: Counter[str] = Counter()
    for chunk in chunks:
        for label in chunk.get("security_flags") or []:
            counter[str(label)] += 1
    return {"instruction_like_chunks": sum(counter.values()), "labels": dict(counter)}
