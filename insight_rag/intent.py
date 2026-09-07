"""Fast conversational routing helpers for the Insight Agent PDF-RAG UI.

The goal is to keep obvious conversation (hello, thanks, capability questions)
out of the retrieval path while still routing document questions to RAG.
"""

from __future__ import annotations

import re


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hiya|salam|salaam|assalam(?:u)?\s*alaikum|good\s+(morning|afternoon|evening))"
    r"(?:\s+(there|aria|friend))?[!. ]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^(thanks|thank\s+you|thank\s+u|thx|jazakallah|jazak\s+allah|shukriya)[!. ]*$",
    re.IGNORECASE,
)
_BYE_RE = re.compile(
    r"^(bye|goodbye|see\s+you|see\s+ya|take\s+care)[!. ]*$",
    re.IGNORECASE,
)
_HOW_ARE_YOU_RE = re.compile(
    r"^(how\s+are\s+you|how\s+are\s+u|how's\s+it\s+going|what's\s+up|whats\s+up)[?!. ]*$",
    re.IGNORECASE,
)
_ACK_RE = re.compile(
    r"^(ok|okay|alright|great|nice|cool|perfect|done|got\s+it)[!. ]*$",
    re.IGNORECASE,
)

_CAPABILITY_PHRASES = (
    "what can you do",
    "what do you do",
    "how can you help",
    "can you help me",
    "who are you",
    "what are you",
    "what can i ask",
    "how do i use this",
    "how does this work",
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

    Routing is intentionally deterministic and local so greetings never trigger
    embeddings/vector search or an unnecessary cloud LLM request.
    """
    normalised = _normalise(text)
    if not normalised:
        return "document_query"

    if any(phrase in normalised for phrase in _CAPABILITY_PHRASES) and len(normalised.split()) <= 12:
        return "capability"

    if (
        _GREETING_RE.fullmatch(normalised)
        or _THANKS_RE.fullmatch(normalised)
        or _BYE_RE.fullmatch(normalised)
        or _HOW_ARE_YOU_RE.fullmatch(normalised)
        or _ACK_RE.fullmatch(normalised)
    ):
        return "smalltalk"

    # Important: messages like "hello, what was revenue in 2025?" are not
    # swallowed as small talk; they continue to the document retrieval path.
    return "document_query"


def conversational_reply(text: str, intent: str) -> str:
    """Return a friendly zero-retrieval response for conversational messages."""
    normalised = _normalise(text)

    if intent == "capability":
        return (
            "I'm ARIA's Insight Agent. I can answer questions about your uploaded text-based PDFs, "
            "including summaries, tables, comparisons, calculations, and follow-up questions. "
            "If the answer is not supported by the PDF, I'll tell you that I couldn't find it."
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

    This is only a fast first-pass gate. Weak lexical matches can still be sent
    to the semantic verifier so paraphrased questions are not incorrectly rejected.
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
