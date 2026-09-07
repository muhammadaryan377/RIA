"""Assistant/system-awareness layer for ARIA Insight PDF-RAG.

Questions about ARIA itself must never be sent to PDF retrieval.  This layer
recognises natural/typo-tolerant questions such as "which model you are" or
"what kind of agent uou are" and answers from runtime configuration.
"""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher

from .config import EMBEDDING_MODEL, RAG_LLM_MODEL
from .final_layer import InsightPDFRAG as FinalInsightPDFRAG


_SYSTEM_TERMS = {
    "agent", "model", "llm", "version", "provider", "engine", "architecture",
    "stack", "technology", "technologies", "embedding", "embeddings", "vector",
    "database", "pgvector", "retrieval", "name", "identity",
}
_DOCUMENT_TERMS = {"pdf", "document", "file", "page", "table", "text"}
_DOCUMENT_EVIDENCE_CUES = {
    "in", "inside", "from", "according", "mentioned", "written", "says", "stated",
}


def _normalise(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.+/-]+", " ", text)
    return " ".join(text.split())


def _looks_like_self_reference(token: str) -> bool:
    """Allow common mobile typos of 'you' while staying conservative."""
    token = token.lower().strip()
    if token in {"you", "your", "yours", "u", "ur", "uou", "yuo", "yoou", "aria"}:
        return True
    if 2 <= len(token) <= 5:
        return SequenceMatcher(None, token, "you").ratio() >= 0.72
    return False


def detect_system_question(text: str) -> str | None:
    """Return the assistant-info category, or ``None`` for normal PDF queries.

    The classifier is deterministic so asking about the assistant never burns a
    vector search or LLM call.  Explicit document-evidence phrasing still wins,
    e.g. "what model is mentioned in the PDF?" remains a PDF question.
    """
    q = _normalise(text)
    if not q:
        return None
    tokens = q.split()
    token_set = set(tokens)

    has_system_term = bool(token_set & _SYSTEM_TERMS)
    has_self_ref = any(_looks_like_self_reference(token) for token in tokens)
    has_question_shape = bool(
        re.search(r"\b(what|which|who|whose|tell|give|are|do|use|using|kind|type)\b", q)
    )

    # "What model is mentioned in the PDF?" is evidence-seeking, not a question
    # about ARIA.  A clear self-reference ("what model do you use for PDF chat")
    # overrides this document cue and remains assistant metadata.
    has_doc = bool(token_set & _DOCUMENT_TERMS)
    has_doc_cue = bool(token_set & _DOCUMENT_EVIDENCE_CUES)
    if has_doc and has_doc_cue and not has_self_ref:
        return None

    if not has_system_term:
        # Natural identity variants not containing the word "agent".
        if re.fullmatch(r"(?:well )?(?:who|what) (?:are|is) (?:you|u|uou|yuo|aria)[?!. ]*", q):
            return "identity"
        return None

    # Keep long content questions out of this fast route unless they clearly ask
    # about the assistant itself.
    if not has_self_ref and (len(tokens) > 12 or not has_question_shape):
        return None

    if "agent" in token_set:
        return "agent"
    if token_set & {"model", "llm", "engine"}:
        return "model"
    if "version" in token_set:
        return "version"
    if token_set & {"embedding", "embeddings"}:
        return "embedding"
    if token_set & {"vector", "database", "pgvector"}:
        return "vector_store"
    if "provider" in token_set:
        return "provider"
    if token_set & {"architecture", "stack", "technology", "technologies", "retrieval"}:
        return "stack"
    if token_set & {"name", "identity"}:
        return "identity"
    return "identity"


class InsightPDFRAG(FinalInsightPDFRAG):
    """Production export with assistant identity/model awareness."""

    def _system_answer(self, kind: str) -> str:
        agent_name = self.insight_agent.__class__.__name__
        version = getattr(self.insight_agent, "VERSION", None)
        version_text = f" v{version}" if version else ""
        provider = str(getattr(self.llm, "provider", "cloud") or "cloud")

        if kind == "agent":
            return (
                f"I'm ARIA's {agent_name}{version_text}. The PDF RAG is one capability inside the "
                "Insight Agent, not a separate fifth agent."
            )
        if kind == "model":
            return (
                f"For PDF chat, I use {RAG_LLM_MODEL} through the {provider} LLM provider for answering, "
                f"query rewriting, planning and evidence checks. Retrieval embeddings use {EMBEDDING_MODEL} locally."
            )
        if kind == "version":
            return f"The active ARIA Insight Agent is {agent_name}{version_text}."
        if kind == "embedding":
            return f"PDF retrieval uses the local embedding model {EMBEDDING_MODEL}."
        if kind == "vector_store":
            return "PDF vectors are stored and searched in PostgreSQL with pgvector through LangChain PGVector."
        if kind == "provider":
            return f"The PDF RAG LLM provider is {provider}; the configured RAG model is {RAG_LLM_MODEL}."
        if kind == "stack":
            return (
                f"My PDF stack uses LangChain, {EMBEDDING_MODEL} local embeddings, PostgreSQL + pgvector, "
                f"hybrid lexical/vector retrieval with RRF, deterministic table reasoning, and {RAG_LLM_MODEL} "
                "for grounded generation."
            )
        return (
            f"I'm ARIA's {agent_name}{version_text}, with conversational PDF RAG built into the Insight Agent."
        )

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

        kind = detect_system_question(question)
        if not kind:
            return super().chat(
                question,
                conversation_id=conversation_id,
                document_ids=document_ids,
            )

        conversation_id = conversation_id or uuid.uuid4().hex
        answer = self._system_answer(kind)
        # Use the existing capability intent so the current UI correctly marks
        # this as a no-retrieval conversational turn.
        self._remember(
            conversation_id,
            question,
            answer,
            intent="capability",
            document_ids=[],
        )
        result = self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent="capability",
            retrieval="not_used",
            evidence_status="not_applicable",
            model=None,
            context_plan={"system_question": kind, "reason": "assistant metadata; PDF retrieval skipped"},
        )
        result["citation_status"] = "not_applicable"
        result["grounding"] = {
            "intent": "capability",
            "evidence_status": "not_applicable",
            "citation_status": "not_applicable",
            "used_pdf_evidence": False,
            "retrieval": "not_used",
        }
        return result
