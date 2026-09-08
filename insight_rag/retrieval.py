"""Dependency-free lexical retrieval, RRF fusion and provenance signals."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from langchain_core.documents import Document


def chunk_key(doc: Document) -> str:
    meta = doc.metadata
    identity = (meta.get("document_id"), meta.get("chunk_id"))
    if identity[1]:
        return str(identity)
    raw = repr((meta.get("document_id"), meta.get("page"),
                meta.get("table_index"), doc.page_content))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate_chunks(documents: list[Document]) -> list[Document]:
    unique = {}
    for doc in documents:
        unique.setdefault(chunk_key(doc), doc)
    return list(unique.values())


def _annotate_retrieval(doc: Document, **values) -> Document:
    """Attach request-local provenance without replacing the Document object.

    Existing orchestration and tests rely on stable chunk object identity. The
    retrieval Documents are request-local/load-local objects, so metadata can be
    safely enriched in place while preserving that identity contract.
    """
    doc.metadata.update(values)
    return doc


def fuse_ranked_lists(rankings: list[list[Document]]) -> list[Document]:
    """RRF across subqueries with auditable consensus metadata.

    ``retrieval_votes`` counts independent ranked lists that returned a chunk and
    ``retrieval_query_count`` is the denominator. This allows monitoring to
    compare single-query and decomposed-query requests without inflating scores.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    ranks: dict[str, list[int]] = {}
    active_rankings = [ranking for ranking in rankings if ranking]
    query_count = max(1, len(active_rankings))
    for ranking in active_rankings:
        seen_in_ranking: set[str] = set()
        for rank, doc in enumerate(deduplicate_chunks(ranking), start=1):
            key = chunk_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            if key not in seen_in_ranking:
                ranks.setdefault(key, []).append(rank)
                seen_in_ranking.add(key)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    result: list[Document] = []
    for key in ordered:
        result.append(
            _annotate_retrieval(
                documents[key],
                retrieval_votes=len(ranks.get(key, [])),
                retrieval_query_count=query_count,
                retrieval_rrf_score=round(scores[key], 8),
                retrieval_subquery_ranks=list(ranks.get(key, [])),
            )
        )
    return result


def terms(text: str) -> list[str]:
    # Unicode letters/digits retain Roman Urdu, Arabic and exact identifiers.
    return re.findall(r"\w+(?:[.-]\w+)*", text.casefold(), flags=re.UNICODE)


def bm25_search(query: str, documents: list[Document], *, limit: int) -> list[Document]:
    """Okapi BM25 with document frequency and length normalisation.

    The corpus must already be restricted to the authenticated user's active
    documents. Index construction is request-local; no stale cross-user cache.
    Returned chunks include their BM25 score/rank for diagnostics and source
    provenance; the score is not treated as a calibrated relevance probability.
    """
    documents = deduplicate_chunks(documents)
    query_terms = terms(query)
    if not documents or not query_terms:
        return []
    counts = [Counter(terms(doc.page_content)) for doc in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1.0
    frequency = Counter(term for count in counts for term in count)
    scores = []
    for index, count in enumerate(counts):
        score = 0.0
        for term in set(query_terms):
            tf = count.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(counts) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / average))
        if score > 0:
            scores.append((score, index))
    scores.sort(key=lambda item: (-item[0], item[1]))
    result: list[Document] = []
    for rank, (score, index) in enumerate(scores[:max(0, limit)], start=1):
        result.append(
            _annotate_retrieval(
                documents[index],
                retrieval_bm25_score=round(score, 8),
                retrieval_bm25_rank=rank,
            )
        )
    return result
