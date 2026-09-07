"""Context engineering for ARIA's Insight Agent PDF-RAG capability.

High-level language understanding is supplied by ``SemanticRouter``. This module
is intentionally deterministic: it validates owned document scope, converts the
router decision into an execution plan, answers local metadata questions, and
selects diversified evidence. It does not contain phrase-based user intents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from .semantic_router import RouteDecision


@dataclass(frozen=True)
class ContextPlan:
    intent: str
    question: str
    task: str = "DOCUMENT_QA"
    document_ids: tuple[str, ...] = ()
    page_numbers: tuple[int, ...] = ()
    text_search_term: str | None = None
    exact_search: bool = False
    broad_query: bool = False
    cross_document: bool = False
    prefer_tables: bool = False
    needs_rewrite: bool = False
    complex_query: bool = False
    clarification: str | None = None
    metadata_kind: str = "NONE"
    table_operations: tuple[str, ...] = ()
    table_filter_operator: str = "NONE"
    table_filter_value: float | None = None
    table_top_n: int | None = None
    reason: str = ""


class ContextEngineer:
    """Translate a validated semantic route into a deterministic RAG plan."""

    @staticmethod
    def _valid_document_ids(
        documents: list[dict],
        selected_document_ids: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        all_ids = [str(doc.get("document_id")) for doc in documents if doc.get("document_id")]
        valid = set(all_ids)
        selected = [str(value) for value in (selected_document_ids or []) if str(value) in valid]
        return all_ids, selected

    def plan(
        self,
        question: str,
        *,
        history: list[dict],
        documents: list[dict],
        selected_document_ids: list[str] | None,
        route_decision: RouteDecision | None = None,
    ) -> ContextPlan:
        del history  # conversation semantics were already resolved by the router
        text = (question or "").strip()
        all_ids, selected = self._valid_document_ids(documents, selected_document_ids)
        default_ids = tuple(selected or all_ids)

        # This fallback is intentionally not an intent classifier. Direct callers
        # that bypass the enterprise router get a conservative generic document
        # plan over the active owned PDFs.
        if route_decision is None:
            return ContextPlan(
                intent="document_query",
                question=text,
                document_ids=default_ids,
                reason="no semantic route supplied; generic grounded document plan",
            )

        decision = route_decision
        owned = set(all_ids)
        active = set(selected) if selected else owned
        resolved = tuple(
            document_id
            for document_id in decision.document_ids
            if document_id in owned and (not selected or document_id in active)
        )

        # Inventory metadata may refer to all owned PDFs even when a UI selection
        # exists; content retrieval remains constrained to selected documents.
        if decision.task == "DOCUMENT_METADATA" and decision.metadata_kind in {
            "INVENTORY_COUNT",
            "INVENTORY_LIST",
        }:
            resolved = tuple(document_id for document_id in decision.document_ids if document_id in owned)
            if not resolved:
                resolved = tuple(all_ids)
        elif not resolved:
            resolved = default_ids

        if decision.scope == "CLARIFICATION" or decision.needs_clarification:
            return ContextPlan(
                intent="clarification",
                question=text,
                task="CLARIFY",
                document_ids=resolved,
                clarification=decision.clarification_question
                or "Could you clarify what you want me to use or find?",
                reason=decision.reason,
            )

        if decision.task == "DOCUMENT_METADATA":
            metadata = decision.metadata_kind
            if metadata in {"INVENTORY_COUNT", "INVENTORY_LIST"}:
                intent = "document_inventory"
            else:
                intent = f"document_metadata:{metadata.lower()}"
            return ContextPlan(
                intent=intent,
                question=text,
                task=decision.task,
                document_ids=resolved,
                metadata_kind=metadata,
                reason=decision.reason,
            )

        if decision.task == "DOCUMENT_SEARCH":
            if not decision.search_term:
                return ContextPlan(
                    intent="clarification",
                    question=text,
                    task="CLARIFY",
                    document_ids=resolved,
                    clarification=decision.clarification_question
                    or "What exact text, topic, or information should I look for?",
                    reason="document search requires a concrete target",
                )
            intent = "text_search" if decision.exact_search else "document_query"
        else:
            intent = "document_query"

        return ContextPlan(
            intent=intent,
            question=text,
            task=decision.task,
            document_ids=resolved,
            page_numbers=tuple(decision.target_pages),
            text_search_term=decision.search_term,
            exact_search=decision.exact_search,
            broad_query=decision.broad_query,
            cross_document=decision.cross_document,
            prefer_tables=decision.prefer_tables,
            needs_rewrite=decision.needs_rewrite,
            complex_query=decision.needs_query_decomposition,
            metadata_kind=decision.metadata_kind,
            table_operations=tuple(decision.table_operations),
            table_filter_operator=decision.table_filter_operator,
            table_filter_value=decision.table_filter_value,
            table_top_n=decision.table_top_n,
            reason=decision.reason,
        )

    @staticmethod
    def inventory_answer(kind: str, *, documents: list[dict]) -> str:
        docs = list(documents)
        if not docs:
            return "You don't have any PDFs uploaded yet."

        count = len(docs)
        noun = "PDF" if count == 1 else "PDFs"
        if kind == "INVENTORY_COUNT":
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

        kind = str(kind or "").lower()
        if kind == "name":
            if len(docs) == 1:
                return f"The PDF is {docs[0].get('filename') or 'Untitled PDF'}."
            return "The selected PDFs are: " + ", ".join(
                str(document.get("filename") or "Untitled PDF") for document in docs
            ) + "."

        field = {"pages": "pages", "tables": "tables", "chunks": "total_chunks"}.get(kind)
        label = {"pages": "pages", "tables": "tables", "chunks": "indexed chunks"}.get(kind, kind)
        if not field:
            return "I couldn't determine that document property."
        if len(docs) == 1:
            document = docs[0]
            return f"{document.get('filename') or 'This PDF'} has {document.get(field, 0)} {label}."
        return "\n".join(
            f"- {document.get('filename') or 'Untitled PDF'}: {document.get(field, 0)} {label}"
            for document in docs
        )

    @staticmethod
    def select_evidence(
        question: str,
        docs: list[Document],
        *,
        selected_document_ids: list[str],
        page_numbers: list[int] | None = None,
        cross_document: bool = False,
        broad_query: bool = False,
        prefer_tables: bool = False,
    ) -> list[Document]:
        """Deduplicate and diversify evidence before it enters model context."""
        del question  # relevance is already encoded by retrieval + reranking
        if not docs:
            return []

        page_set = set(page_numbers or [])
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

        if page_set:
            unique.sort(
                key=lambda document: (
                    0 if int(document.metadata.get("page", 0) or 0) in page_set else 1,
                    0 if prefer_tables and document.metadata.get("content_type") == "table" else 1,
                )
            )
        elif prefer_tables:
            unique.sort(key=lambda document: 0 if document.metadata.get("content_type") == "table" else 1)

        max_docs = 8 if broad_query or cross_document else (6 if prefer_tables else 5)
        chosen: list[Document] = []
        chosen_ids: set[str] = set()

        # Broad synthesis needs page coverage rather than many adjacent chunks.
        if broad_query:
            for document_id in selected_document_ids:
                page_seen: set[int] = set()
                candidates = [
                    document
                    for document in unique
                    if str(document.metadata.get("document_id")) == str(document_id)
                ]
                candidates.sort(
                    key=lambda document: (
                        int(document.metadata.get("page", 0) or 0),
                        0 if document.metadata.get("content_type") == "text" else 1,
                    )
                )
                for document in candidates:
                    page = int(document.metadata.get("page", 0) or 0)
                    if page in page_seen:
                        continue
                    page_seen.add(page)
                    cid = str(document.metadata.get("chunk_id") or id(document))
                    chosen.append(document)
                    chosen_ids.add(cid)
                    if len(chosen) >= max_docs:
                        return chosen

        # Cross-document tasks must contain evidence from each document when
        # retrieval found it, preventing one-sided comparisons.
        if cross_document:
            for document_id in selected_document_ids:
                candidate = next(
                    (
                        document
                        for document in unique
                        if str(document.metadata.get("document_id")) == str(document_id)
                    ),
                    None,
                )
                if candidate is not None:
                    cid = str(candidate.metadata.get("chunk_id") or id(candidate))
                    if cid not in chosen_ids:
                        chosen.append(candidate)
                        chosen_ids.add(cid)

        for document in unique:
            cid = str(document.metadata.get("chunk_id") or id(document))
            if cid in chosen_ids:
                continue
            chosen.append(document)
            chosen_ids.add(cid)
            if len(chosen) >= max_docs:
                break

        return chosen[:max_docs]
