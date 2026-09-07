"""Fast conversational routing helpers for ARIA Insight PDF-RAG.

Obvious conversation is handled locally so greetings and capability questions do
not waste embeddings, vector search, or cloud tokens. Everything factual that
may require document evidence continues into the context-engineering layer.
"""

from __future__ import annotations

import re


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hiya|yo|salam|salaam|assalam(?:u)?\s*alaikum|good\s+(morning|afternoon|evening))"
    r"(?:\s+(there|aria|friend))?[!. ]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^(thanks|thank\s+you|thank\s+u|thx|jazakallah|jazak\s+allah|shukriya|appreciate\s+it)[!. ]*$",
    re.IGNORECASE,
)
_BYE_RE = re.compile(
    r"^(bye|goodbye|see\s+you|see\s+ya|take\s+care|talk\s+later)[!. ]*$",
    re.IGNORECASE,
)
_HOW_ARE_YOU_RE = re.compile(
    r"^(how\s+are\s+you|how\s+are\s+u|how's\s+it\s+going|how\s+is\s+it\s+going|what's\s+up|whats\s+up)[?!. ]*$",
    re.IGNORECASE,
)
_ACK_RE = re.compile(
    r"^(ok|okay|alright|great|nice|cool|perfect|done|got\s+it|understood|fine)[!. ]*$",
    re.IGNORECASE,
)

_CAPABILITY_PHRASES = (
    "what can you do",
    "what you can do",
    "what do you do",
    "what are you able to do",
    "how can you help",
    "how you can help",
    "can you help me",
    "who are you",
    "what are you",
    "what can i ask",
    "what should i ask",
    "how do i use this",
    "how does this work",
    "what can you find",
    "what you can find",
)

_BROAD_DOCUMENT_PHRASES = (
    "summarize this document",
    "summarise this document",
    "summarize the document",
    "summarise the document",
    "summarize this pdf",
    "summarise this pdf",
    "what is this document about",
    "what is this pdf about",
    "what are the topics in this pdf",
    "what topics are in this pdf",
    "give me an overview",
    "give me a summary",
    "main points",
    "key points",
    "explain this document",
    "explain this pdf",
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and",
    "or", "with", "what", "which", "who", "when", "where", "why", "how", "did", "does",
    "do", "me", "show", "tell", "from", "according", "report", "document", "pdf", "please",
    "can", "could", "would", "about", "this", "that", "these", "those",
}


def _normalise(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def route_message(text: str) -> str:
    """Return ``smalltalk``, ``capability`` or ``document_query``.

    Messages that mix a greeting with a real question deliberately continue to
    document routing, e.g. ``hello, what was revenue in 2025?``.
    """
    normalised = _normalise(text)
    if not normalised:
        return "document_query"

    if any(phrase in normalised for phrase in _CAPABILITY_PHRASES) and len(normalised.split()) <= 16:
        # "what can you find in this pdf about revenue" is a document query, not
        # a generic capability question.
        if not re.search(r"\b(?:in|inside|from)\s+(?:this|the|my)\s+(?:pdf|document|file)\b", normalised):
            return "capability"

    if (
        _GREETING_RE.fullmatch(normalised)
        or _THANKS_RE.fullmatch(normalised)
        or _BYE_RE.fullmatch(normalised)
        or _HOW_ARE_YOU_RE.fullmatch(normalised)
        or _ACK_RE.fullmatch(normalised)
    ):
        return "smalltalk"

    return "document_query"


def conversational_reply(text: str, intent: str) -> str:
    """Return a friendly zero-retrieval response for conversational messages."""
    normalised = _normalise(text)

    if intent == "capability":
        return (
            "I'm ARIA's Insight Agent. I can chat normally and work with uploaded text-based PDFs: "
            "find exact text, summarize sections, answer page questions, read tables, compare values and PDFs, "
            "and understand follow-up questions. If the evidence is not in the PDF, I won't guess."
        )

    if _THANKS_RE.fullmatch(normalised):
        return "You're welcome!"
    if _BYE_RE.fullmatch(normalised):
        return "Goodbye! Come back anytime you want to explore your PDFs."
    if _HOW_ARE_YOU_RE.fullmatch(normalised):
        return "I'm doing well and ready to help. You can chat with me normally or ask about your PDFs."
    if _ACK_RE.fullmatch(normalised):
        return "Great. I'm ready whenever you are."
    return "Hi! How can I help you today? You can chat with me or ask anything about your uploaded PDFs."


def is_broad_document_query(text: str) -> bool:
    normalised = _normalise(text)
    return any(phrase in normalised for phrase in _BROAD_DOCUMENT_PHRASES)


def lexical_evidence_score(question: str, evidence_texts: list[str]) -> float:
    """Return a conservative 0..1 lexical support score.

    This is only a first-pass gate. Weak lexical matches can still be sent to a
    semantic verifier so paraphrases are not incorrectly rejected.
    """
    q_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.%-]*", _normalise(question))
        if len(token) > 1 and token not in _STOPWORDS
    }
    if not q_tokens or not evidence_texts:
        return 0.0

    q_numbers = {token for token in q_tokens if any(ch.isdigit() for ch in token)}
    long_tokens = {token for token in q_tokens if len(token) >= 4}
    best = 0.0

    for text in evidence_texts:
        lowered = (text or "").lower()
        d_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.%-]*", lowered))
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        long_overlap = len(long_tokens & d_tokens) / max(1, len(long_tokens)) if long_tokens else 0.0
        number_overlap = len(q_numbers & d_tokens) / max(1, len(q_numbers)) if q_numbers else 0.0
        score = (0.65 * overlap) + (0.25 * long_overlap) + (0.10 * number_overlap)
        best = max(best, score)

    return min(1.0, best)
