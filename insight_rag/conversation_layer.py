"""Final conversational layer for ARIA's Insight Agent PDF-RAG.

This layer handles natural follow-ups about the previous answer (repeat, rephrase,
translate, show sources) without unnecessarily querying the vector store again.
It also improves broad-document context selection with page-balanced evidence.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from langchain_core.documents import Document

from .config import RAG_LLM_MODEL
from .context_engine import ContextEngineer
from .friendly import InsightPDFRAG as ContextAwareInsightPDFRAG
from .intent import is_broad_document_query


_SOURCE_FOLLOWUP_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"show\s+(?:me\s+)?(?:the\s+)?sources?|"
    r"what\s+(?:is|are)\s+(?:the\s+|your\s+)?sources?|"
    r"where\s+did\s+you\s+(?:find|get)\s+(?:that|this|it)|"
    r"where\s+(?:is|was)\s+(?:that|this)\s+from|"
    r"which\s+page\s+(?:is|was)\s+(?:that|this)\s+(?:on|from)|"
    r"give\s+(?:me\s+)?(?:the\s+)?citations?"
    r")[?!. ]*$",
    re.IGNORECASE,
)
_REPEAT_RE = re.compile(
    r"^(?:please\s+)?(?:repeat\s+(?:that|it|your\s+answer)|say\s+(?:that|it)\s+again|"
    r"show\s+(?:me\s+)?(?:your\s+)?previous\s+answer|what\s+did\s+you\s+just\s+say)[?!. ]*$",
    re.IGNORECASE,
)
_TRANSFORM_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"make\s+(?:it|that|your\s+answer)\s+(?:shorter|short|simpler|simple|more\s+concise|clearer)|"
    r"explain\s+(?:it|that|your\s+answer)\s+(?:simply|more\s+simply|in\s+simple\s+words)|"
    r"(?:put|give)\s+(?:it|that|your\s+answer)\s+(?:in|as)\s+(?:bullet\s+points?|bullets?)|"
    r"translate\s+(?:it|that|your\s+answer)\s+(?:to|into)\s+[a-zA-Z -]+|"
    r"(?:write|say)\s+(?:it|that|your\s+answer)\s+in\s+[a-zA-Z -]+"
    r")[?!. ]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationAction:
    kind: str
    instruction: str = ""


def _last_assistant_message(history: list[dict]) -> dict | None:
    return next((item for item in reversed(history or []) if item.get("role") == "assistant"), None)


def detect_conversation_action(text: str, history: list[dict]) -> ConversationAction | None:
    """Detect a follow-up that operates on the previous answer rather than the PDFs."""
    if _last_assistant_message(history) is None:
        return None
    message = " ".join((text or "").strip().split())
    if _SOURCE_FOLLOWUP_RE.fullmatch(message):
        return ConversationAction("show_sources", message)
    if _REPEAT_RE.fullmatch(message):
        return ConversationAction("repeat", message)
    if _TRANSFORM_RE.fullmatch(message):
        return ConversationAction("transform", message)
    return None


class AdvancedContextEngineer(ContextEngineer):
    """Adds page-balanced context for broad summaries while preserving fast routing."""

    @staticmethod
    def select_evidence(
        question: str,
        docs: list[Document],
        *,
        selected_document_ids: list[str],
        page_numbers: list[int] | None = None,
        cross_document: bool = False,
    ) -> list[Document]:
        if not is_broad_document_query(question):
            return ContextEngineer.select_evidence(
                question,
                docs,
                selected_document_ids=selected_document_ids,
                page_numbers=page_numbers,
                cross_document=cross_document,
            )

        # Broad summaries need coverage, not several nearly-identical chunks from
        # one page. Keep one representative text chunk per page first, then fill
        # remaining slots with other unique chunks. Eight chunks stays within the
        # current context budget while covering typical short/medium PDFs well.
        unique: list[Document] = []
        seen_chunks: set[str] = set()
        seen_content: set[str] = set()
        for doc in docs:
            chunk_id = str(doc.metadata.get("chunk_id") or "")
            fingerprint = re.sub(r"\s+", " ", doc.page_content.strip().lower())[:500]
            if chunk_id and chunk_id in seen_chunks:
                continue
            if fingerprint and fingerprint in seen_content:
                continue
            if chunk_id:
                seen_chunks.add(chunk_id)
            if fingerprint:
                seen_content.add(fingerprint)
            unique.append(doc)

        chosen: list[Document] = []
        chosen_ids: set[str] = set()
        max_docs = 8

        for document_id in selected_document_ids:
            page_seen: set[int] = set()
            candidates = [
                doc for doc in unique
                if str(doc.metadata.get("document_id")) == str(document_id)
            ]
            candidates.sort(
                key=lambda doc: (
                    int(doc.metadata.get("page", 0) or 0),
                    0 if doc.metadata.get("content_type") == "text" else 1,
                )
            )
            for doc in candidates:
                page = int(doc.metadata.get("page", 0) or 0)
                if page in page_seen:
                    continue
                page_seen.add(page)
                cid = str(doc.metadata.get("chunk_id") or id(doc))
                chosen.append(doc)
                chosen_ids.add(cid)
                if len(chosen) >= max_docs:
                    return chosen

        for doc in unique:
            cid = str(doc.metadata.get("chunk_id") or id(doc))
            if cid in chosen_ids:
                continue
            chosen.append(doc)
            if len(chosen) >= max_docs:
                break
        return chosen


class InsightPDFRAG(ContextAwareInsightPDFRAG):
    """Final exported RAG with previous-answer operations and richer context engineering."""

    def __init__(self, *, insight_agent, user_id: str | int):
        super().__init__(insight_agent=insight_agent, user_id=user_id)
        self.context_engine = AdvancedContextEngineer()

    @staticmethod
    def _source_summary(sources: list[dict]) -> str:
        if not sources:
            return "My previous answer did not use PDF evidence, so there are no PDF citations to show."
        lines = ["I used these PDF sources for my previous answer:"]
        seen = set()
        for source in sources:
            key = (
                source.get("filename"),
                source.get("page"),
                source.get("table_index"),
            )
            if key in seen:
                continue
            seen.add(key)
            detail = f"- {source.get('filename') or 'PDF'}, page {source.get('page')}"
            if source.get("content_type") == "table" and source.get("table_index"):
                detail += f", table {source.get('table_index')}"
            lines.append(detail)
            if len(lines) >= 6:
                break
        return "\n".join(lines)

    def _transform_previous_answer(self, instruction: str, previous: dict) -> str:
        previous_text = str(previous.get("content") or "").strip()
        if not previous_text:
            return "I don't have a previous answer to transform yet."
        messages = [
            {
                "role": "system",
                "content": (
                    "Transform the previous assistant answer exactly as the user requests. "
                    "Do not add new facts, outside knowledge, new claims, or new citations. "
                    "Preserve source markers like [S1] exactly when they appear. "
                    "If translating, translate the prose but keep filenames, numbers, and source labels accurate. "
                    "Return only the transformed answer."
                ),
            },
            {
                "role": "user",
                "content": f"User instruction: {instruction}\n\nPrevious answer:\n{previous_text}",
            },
        ]
        return self.llm.chat(
            "rag",
            messages,
            temperature=0.0,
            num_predict=700,
            timeout=20,
        ).strip()

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        question = (question or "").strip()
        if len(question) < 2:
            raise ValueError("Please type a message or a question.")
        if len(question) > 4000:
            raise ValueError("Question is too long.")

        if conversation_id:
            history = self.conversations.load(conversation_id).get("messages", [])
            action = detect_conversation_action(question, history)
        else:
            history = []
            action = None

        if not action:
            return super().chat(
                question,
                conversation_id=conversation_id,
                document_ids=document_ids,
            )

        conversation_id = conversation_id or uuid.uuid4().hex
        previous = _last_assistant_message(history) or {}
        previous_sources = list(previous.get("sources") or [])
        previous_doc_ids = []
        for source in previous_sources:
            doc_id = str(source.get("document_id") or "")
            if doc_id and doc_id not in previous_doc_ids:
                previous_doc_ids.append(doc_id)

        if action.kind == "show_sources":
            answer = self._source_summary(previous_sources)
            model = None
        elif action.kind == "repeat":
            answer = str(previous.get("content") or "I don't have a previous answer to repeat yet.")
            model = None
        else:
            try:
                answer = self._transform_previous_answer(action.instruction, previous)
                model = RAG_LLM_MODEL
            except Exception:
                answer = (
                    "I'm having trouble rephrasing the previous answer right now. "
                    "The original answer is still available just above."
                )
                model = RAG_LLM_MODEL

        intent = f"conversation_{action.kind}"
        self._remember(
            conversation_id,
            question,
            answer,
            intent=intent,
            sources=previous_sources,
            document_ids=previous_doc_ids,
        )
        return self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent=intent,
            document_ids=previous_doc_ids,
            sources=previous_sources,
            retrieval="not_used",
            evidence_status="carried_forward",
            model=model,
            context_plan={"conversation_action": action.kind},
        )
