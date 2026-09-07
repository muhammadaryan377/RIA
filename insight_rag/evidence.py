"""Deterministic evidence-scoring primitives for ARIA PDF RAG."""

from __future__ import annotations

import re

# Linguistic stopwords are retrieval noise filters, not intent phrases.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and",
    "or", "with", "what", "which", "who", "when", "where", "why", "how", "did", "does",
    "do", "me", "show", "tell", "from", "according", "please", "can", "could", "would",
    "about", "this", "that", "these", "those",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.%-]*", (text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def lexical_evidence_score(question: str, evidence_texts: list[str]) -> float:
    """Return a conservative 0..1 lexical support score.

    This is only a cheap first-pass support signal. Semantic verification remains
    authoritative for paraphrases and non-lexical matches.
    """
    q_tokens = _tokens(question)
    if not q_tokens or not evidence_texts:
        return 0.0

    q_numbers = {token for token in q_tokens if any(ch.isdigit() for ch in token)}
    long_tokens = {token for token in q_tokens if len(token) >= 4}
    best = 0.0

    for text in evidence_texts:
        d_tokens = _tokens(text)
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        long_overlap = len(long_tokens & d_tokens) / max(1, len(long_tokens)) if long_tokens else 0.0
        number_overlap = len(q_numbers & d_tokens) / max(1, len(q_numbers)) if q_numbers else 0.0
        score = (0.65 * overlap) + (0.25 * long_overlap) + (0.10 * number_overlap)
        best = max(best, score)

    return min(1.0, best)
