"""Dependency-free BM25 and stable chunk identity for bounded PDF retrieval."""

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


def fuse_ranked_lists(rankings: list[list[Document]]) -> list[Document]:
    """RRF across subqueries so duplicate hits cannot consume reranker slots."""
    scores, documents = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(deduplicate_chunks(ranking), start=1):
            key = chunk_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)]


def terms(text: str) -> list[str]:
    # Unicode letters/digits retain Roman Urdu, Arabic and exact identifiers.
    return re.findall(r"\w+(?:[.-]\w+)*", text.casefold(), flags=re.UNICODE)


def bm25_search(query: str, documents: list[Document], *, limit: int) -> list[Document]:
    """Okapi BM25 with document frequency and length normalisation.

    The corpus must already be restricted to the authenticated user's active
    documents. Index construction is request-local; no stale cross-user cache.
    """
    documents = deduplicate_chunks(documents)
    if not documents or not terms(query):
        return []
    counts = [Counter(terms(doc.page_content)) for doc in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1.0
    frequency = Counter(term for count in counts for term in count)
    scores = []
    for index, count in enumerate(counts):
        score = 0.0
        for term in set(terms(query)):
            tf = count.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(counts) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / average))
        if score > 0:
            scores.append((score, index))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [documents[index] for _, index in scores[:max(0, limit)]]
