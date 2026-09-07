"""Context engineering for ARIA's Insight Agent PDF-RAG capability.

This module is deliberately deterministic and cheap. It handles conversation
routing, document references, metadata questions, ambiguity, page targeting,
text-search intent, and evidence selection before any expensive retrieval or
LLM call is made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .intent import is_broad_document_query, route_message


_COMPARE_WORDS = (
    "compare", "comparison", "difference", "different", "versus", " vs ",
    "both pdf", "both document", "across pdf", "across document", "between the pdf",
)
_TABLE_WORDS = (
    "table", "row", "column", "total", "sum", "average", "avg", "highest", "lowest",
    "maximum", "minimum", "revenue", "sales", "profit", "amount", "count", "percentage",
)
_INVENTORY_PATTERNS = (
    r"\bhow\s+many\s+(?:pdfs?|documents?|files?)\b",
    r"\b(?:what|which|list|show)\s+(?:pdfs?|documents?|files?)\b",
    r"\b(?:pdfs?|documents?|files?)\s+(?:do\s+i|do\s+you|are)\s+(?:have|uploaded|using|selected)\b",
    r"\bwhat\s+(?:do\s+you|i)\s+have\s+uploaded\b",
    r"\buploaded\s+(?:pdfs?|documents?|files?)\b",
)
_METADATA_PATTERNS = {
    "pages": (
        r"\bhow\s+many\s+pages\b",
        r"\bpage\s+count\b",
    ),
    "tables": (
        r"\bhow\s+many\s+tables\b",
        r"\btable\s+count\b",
    ),
    "chunks": (
        r"\bhow\s+many\s+chunks\b",
        r"\bchunk\s+count\b",
    ),
    "name": (
        r"\b(?:what|which)\s+(?:is\s+)?(?:the\s+)?(?:pdf|document|file)\s+name\b",
        r"\bname\s+of\s+(?:this|the)\s+(?:pdf|document|file)\b",
    ),
}
_VAGUE_SEARCH_RE = re.compile(
    r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:find|search|locate|look\s+for)"
    r"(?:\s+(?:the\s+)?(?:text|information|info|content|it|that|this))?[?!. ]*$",
    re.IGNORECASE,
)
_FOLLOWUP_VAGUE_RE = re.compile(
    r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:find|search(?:\s+for)?|locate|look\s+for)\s+(?:it|that|this)[?!. ]*$",
    re.IGNORECASE,
)
_FIND_PREFIX_RE = re.compile(
    r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:find|search(?:\s+for)?|locate|look\s+for)\s+(.+)$",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"\bpages?\s+(\d{1,4})(?:\s*(?:-|to)\s*(\d{1,4}))?\b", re.IGNORECASE)
_P_DOT_RE = re.compile(r"\bp\.?\s*(\d{1,4})\b", re.IGNORECASE)
_SINGULAR_DOC_REF_RE = re.compile(
    r"\b(?:this|that|the|current|same)\s+(?:pdf|document|file)\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _doc_tokens(filename: str) -> set[str]:
    stem = Path(filename or "").stem.lower()
    return {token for token in re.findall(r"[a-z0-9]+", stem) if len(token) > 1}


def _history_source_ids(history: list[dict]) -> list[str]:
    seen: list[str] = []
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        for source in message.get("sources") or []:
            doc_id = str(source.get("document_id") or "")
            if doc_id and doc_id not in seen:
                seen.append(doc_id)
        if seen:
            break
    return seen


@dataclass(frozen=True)
class ContextPlan:
    intent: str
    question: str
    document_ids: tuple[str, ...] = ()
    page_numbers: tuple[int, ...] = ()
    text_search_term: str | None = None
    broad_query: bool = False
    cross_document: bool = False
    clarification: str | None = None
    reason: str = ""


class ContextEngineer:
    """Build a compact execution plan from user text + conversation + metadata."""

    def plan(
        self,
        question: str,
        *,
        history: list[dict],
        documents: list[dict],
        selected_document_ids: list[str] | None,
    ) -> ContextPlan:
        text = (question or "").strip()
        normalised = _norm(text)
        selected = self.resolve_document_scope(
            text,
            history=history,
            documents=documents,
            selected_document_ids=selected_document_ids,
        )

        basic = route_message(text)
        if basic in {"smalltalk", "capability"}:
            return ContextPlan(
                intent=basic,
                question=text,
                document_ids=tuple(selected),
                reason="fast conversational route",
            )

        if self.is_inventory_query(normalised):
            return ContextPlan(
                intent="document_inventory",
                question=text,
                document_ids=tuple(selected),
                reason="document inventory can be answered from manifest metadata",
            )

        if self.needs_scope_clarification(
            text,
            history=history,
            documents=documents,
            selected_document_ids=selected_document_ids,
        ):
            names = self._active_document_names(documents, selected_document_ids)
            return ContextPlan(
                intent="clarification",
                question=text,
                document_ids=tuple(selected),
                clarification=(
                    "You have multiple PDFs selected. Which one do you mean? "
                    + ", ".join(names[:5])
                    + ("." if len(names) <= 5 else ", or another uploaded PDF?")
                ),
                reason="singular document reference is ambiguous",
            )

        metadata_kind = self.metadata_query_kind(normalised)
        if metadata_kind:
            return ContextPlan(
                intent=f"document_metadata:{metadata_kind}",
                question=text,
                document_ids=tuple(selected),
                reason="document metadata can be answered without retrieval",
            )

        if _VAGUE_SEARCH_RE.fullmatch(normalised):
            # "find it/that/this" can be a real follow-up only when the previous
            # assistant answer was grounded in a concrete source. Generic phrases
            # like "can you find the text" always require clarification.
            recent_sources = _history_source_ids(history)
            if not (_FOLLOWUP_VAGUE_RE.fullmatch(normalised) and recent_sources):
                return ContextPlan(
                    intent="clarification",
                    question=text,
                    document_ids=tuple(selected),
                    clarification="Sure. What exact text, phrase, topic, or information should I look for?",
                    reason="search request has no concrete target",
                )

        term = self.extract_text_search_term(text)
        pages = self.extract_page_numbers(text)
        broad = is_broad_document_query(text)
        cross = self.is_cross_document_query(normalised, selected)

        return ContextPlan(
            intent="text_search" if term else "document_query",
            question=text,
            document_ids=tuple(selected),
            page_numbers=tuple(pages),
            text_search_term=term,
            broad_query=broad,
            cross_document=cross,
            reason="document evidence required",
        )

    @staticmethod
    def is_inventory_query(normalised: str) -> bool:
        return any(re.search(pattern, normalised, re.IGNORECASE) for pattern in _INVENTORY_PATTERNS)

    @staticmethod
    def metadata_query_kind(normalised: str) -> str | None:
        for kind, patterns in _METADATA_PATTERNS.items():
            if any(re.search(pattern, normalised, re.IGNORECASE) for pattern in patterns):
                return kind
        return None

    @staticmethod
    def _active_document_names(documents: list[dict], selected_document_ids: list[str] | None) -> list[str]:
        valid = {str(doc.get("document_id")): doc for doc in documents if doc.get("document_id")}
        ids = [str(i) for i in (selected_document_ids or []) if str(i) in valid]
        if not ids:
            ids = list(valid)
        return [str(valid[doc_id].get("filename") or "Untitled PDF") for doc_id in ids]

    def needs_scope_clarification(
        self,
        text: str,
        *,
        history: list[dict],
        documents: list[dict],
        selected_document_ids: list[str] | None,
    ) -> bool:
        """Return True when a singular PDF reference could mean several PDFs."""
        if not _SINGULAR_DOC_REF_RE.search(text or ""):
            return False
        docs = [doc for doc in documents if doc.get("document_id")]
        valid = {str(doc["document_id"]) for doc in docs}
        active = [str(i) for i in (selected_document_ids or []) if str(i) in valid]
        if not active:
            active = [str(doc["document_id"]) for doc in docs]
        if len(active) <= 1:
            return False

        q = _norm(text)
        if re.search(r"\b(?:all|both)\s+(?:pdfs?|documents?|files?)\b", q):
            return False
        if _history_source_ids(history):
            return False

        # Explicit filenames or ordinals remove the ambiguity.
        for doc in docs:
            filename = str(doc.get("filename") or "")
            stem = Path(filename).stem.lower()
            if filename.lower() in q or (len(stem) >= 4 and stem in q):
                return False
        if any(re.search(rf"\b{re.escape(label)}\s+(?:pdf|document|file)\b", q) for label in _ORDINALS):
            return False
        if re.search(r"\blast\s+(?:pdf|document|file)\b", q):
            return False
        return True

    @staticmethod
    def extract_page_numbers(text: str) -> list[int]:
        pages: list[int] = []
        for match in _PAGE_RE.finditer(text or ""):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            if end < start:
                start, end = end, start
            # Avoid accidentally expanding a huge typo into thousands of pages.
            for page in range(start, min(end, start + 20) + 1):
                if page not in pages:
                    pages.append(page)
        for match in _P_DOT_RE.finditer(text or ""):
            page = int(match.group(1))
            if page not in pages:
                pages.append(page)
        return pages

    @staticmethod
    def extract_text_search_term(text: str) -> str | None:
        raw = (text or "").strip()
        quoted = re.search(r"[\"']([^\"']{2,200})[\"']", raw)
        if quoted and re.search(r"\b(find|search|locate|look\s+for)\b", raw, re.IGNORECASE):
            return quoted.group(1).strip()

        match = _FIND_PREFIX_RE.match(raw)
        if not match:
            return None
        term = match.group(1).strip(" ?!.")
        term = re.sub(
            r"^(?:the\s+)?(?:text|phrase|word|words|topic|information|info)\s+(?:about\s+)?",
            "",
            term,
            flags=re.IGNORECASE,
        )
        if _norm(term) in {"text", "the text", "information", "info", "content", "it", "that", "this"}:
            return None
        return term if len(term) >= 2 else None

    @staticmethod
    def is_cross_document_query(normalised: str, selected_ids: list[str]) -> bool:
        if len(selected_ids) < 2:
            return False
        return any(word in normalised for word in _COMPARE_WORDS)

    def resolve_document_scope(
        self,
        question: str,
        *,
        history: list[dict],
        documents: list[dict],
        selected_document_ids: list[str] | None,
    ) -> list[str]:
        """Resolve explicit filenames, pronouns and ordinals into owned document IDs."""
        docs = [doc for doc in documents if doc.get("document_id")]
        all_ids = [str(doc["document_id"]) for doc in docs]
        valid = set(all_ids)
        selected = [str(i) for i in (selected_document_ids or []) if str(i) in valid]
        base_ids = selected or list(all_ids)
        q = _norm(question)

        if not docs:
            return []

        # Explicit "all uploaded PDFs" ignores UI selection; "both" respects the
        # active selection when present because it usually means both checked docs.
        if re.search(r"\ball\s+(?:uploaded\s+)?(?:pdfs?|documents?|files?)\b", q):
            return all_ids
        if re.search(r"\bboth\s+(?:pdfs?|documents?|files?)\b", q) and len(base_ids) >= 2:
            return base_ids

        # Exact/near filename mention. Require at least two filename tokens when
        # possible so ordinary words don't accidentally select a document.
        mentioned: list[str] = []
        for doc in docs:
            filename = str(doc.get("filename") or "")
            stem = Path(filename).stem.lower()
            if filename.lower() in q or (len(stem) >= 4 and stem in q):
                mentioned.append(str(doc["document_id"]))
                continue
            tokens = _doc_tokens(filename)
            if tokens:
                overlap = sum(1 for token in tokens if token in q)
                needed = 1 if len(tokens) == 1 else 2
                if overlap >= needed:
                    mentioned.append(str(doc["document_id"]))
        if mentioned:
            return mentioned

        # Ordinal references use the order shown by list_documents() (newest first).
        for label, index in _ORDINALS.items():
            if re.search(rf"\b{re.escape(label)}\s+(?:pdf|document|file)\b", q) and index < len(docs):
                return [str(docs[index]["document_id"])]
        if re.search(r"\blast\s+(?:pdf|document|file)\b", q):
            return [str(docs[-1]["document_id"])]

        recent = [doc_id for doc_id in _history_source_ids(history) if doc_id in valid]
        if re.search(r"\b(?:this|same|current|that)\s+(?:pdf|document|file)\b", q) and recent:
            return recent[:1]
        if re.search(r"\b(?:other|another)\s+(?:pdf|document|file)\b", q) and recent:
            pool = [doc_id for doc_id in base_ids if doc_id not in recent]
            if pool:
                return pool[:1]

        return base_ids

    @staticmethod
    def inventory_answer(
        question: str,
        *,
        documents: list[dict],
        selected_document_ids: Iterable[str] | None = None,
    ) -> str:
        q = _norm(question)
        docs = list(documents)
        selected_set = {str(i) for i in (selected_document_ids or [])}
        if "selected" in q or "using" in q or "active" in q:
            docs = [doc for doc in docs if str(doc.get("document_id")) in selected_set]

        if not docs:
            return "You don't have any PDFs uploaded yet."

        count = len(docs)
        noun = "PDF" if count == 1 else "PDFs"
        if re.search(r"\bhow\s+many\b", q):
            names = ", ".join(str(doc.get("filename") or "Untitled PDF") for doc in docs[:5])
            suffix = "" if count <= 5 else f", and {count - 5} more"
            return f"You currently have {count} {noun}: {names}{suffix}."

        lines = [f"You currently have {count} {noun}:"]
        for index, doc in enumerate(docs, start=1):
            lines.append(
                f"{index}. {doc.get('filename') or 'Untitled PDF'} "
                f"({doc.get('pages', 0)} pages, {doc.get('tables', 0)} tables)"
            )
        return "\n".join(lines)

    @staticmethod
    def metadata_answer(kind: str, *, documents: list[dict]) -> str:
        docs = list(documents)
        if not docs:
            return "I don't have an uploaded PDF to inspect yet."

        if kind == "name":
            if len(docs) == 1:
                return f"The PDF is {docs[0].get('filename') or 'Untitled PDF'}."
            return "The selected PDFs are: " + ", ".join(str(d.get("filename") or "Untitled PDF") for d in docs) + "."

        field = {"pages": "pages", "tables": "tables", "chunks": "total_chunks"}.get(kind)
        label = {"pages": "pages", "tables": "tables", "chunks": "indexed chunks"}.get(kind, kind)
        if not field:
            return "I couldn't determine that document property."
        if len(docs) == 1:
            doc = docs[0]
            return f"{doc.get('filename') or 'This PDF'} has {doc.get(field, 0)} {label}."
        return "\n".join(
            f"- {doc.get('filename') or 'Untitled PDF'}: {doc.get(field, 0)} {label}"
            for doc in docs
        )

    @staticmethod
    def select_evidence(
        question: str,
        docs: list[Document],
        *,
        selected_document_ids: list[str],
        page_numbers: list[int] | None = None,
        cross_document: bool = False,
    ) -> list[Document]:
        """Deduplicate and diversify evidence before it enters the model context."""
        if not docs:
            return []
        page_set = set(page_numbers or [])
        q = _norm(question)
        table_intent = any(word in q for word in _TABLE_WORDS)
        max_docs = 6 if cross_document else (6 if table_intent else 5)

        unique: list[Document] = []
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        for doc in docs:
            chunk_id = str(doc.metadata.get("chunk_id") or "")
            fingerprint = re.sub(r"\s+", " ", doc.page_content.strip().lower())[:500]
            if chunk_id and chunk_id in seen_ids:
                continue
            if fingerprint and fingerprint in seen_content:
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            if fingerprint:
                seen_content.add(fingerprint)
            unique.append(doc)

        # Explicit page requests get priority.
        if page_set:
            unique.sort(key=lambda d: (0 if int(d.metadata.get("page", 0) or 0) in page_set else 1))

        chosen: list[Document] = []
        chosen_ids: set[str] = set()

        # For comparisons, guarantee at least one evidence block per selected PDF
        # when retrieval found one, avoiding a one-sided comparison.
        if cross_document:
            for document_id in selected_document_ids:
                candidate = next((d for d in unique if str(d.metadata.get("document_id")) == document_id), None)
                if candidate is not None:
                    cid = str(candidate.metadata.get("chunk_id") or id(candidate))
                    chosen.append(candidate)
                    chosen_ids.add(cid)

        for doc in unique:
            cid = str(doc.metadata.get("chunk_id") or id(doc))
            if cid in chosen_ids:
                continue
            chosen.append(doc)
            chosen_ids.add(cid)
            if len(chosen) >= max_docs:
                break

        return chosen[:max_docs]
