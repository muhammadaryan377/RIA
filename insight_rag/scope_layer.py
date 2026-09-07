"""Bounded assistant policy for ARIA's Insight Agent PDF-RAG.

ARIA should feel conversational, but it is not a general-purpose world-knowledge
assistant. This layer separates four kinds of user turns:

- CONVERSATION: greetings, thanks, casual interaction
- SYSTEM: questions about ARIA, the Insight Agent, its RAG stack or behavior
- DOCUMENT: questions that must be answered from uploaded PDF evidence
- OUT_OF_SCOPE: unrelated general knowledge/current-affairs/trivia requests

Only DOCUMENT turns are allowed to enter PDF retrieval. SYSTEM turns are answered
from a compact runtime profile. OUT_OF_SCOPE turns are never answered from model
world knowledge.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .config import EMBEDDING_MODEL, RAG_LLM_MODEL
from .intent import route_message
from .system_layer import InsightPDFRAG as SystemAwareInsightPDFRAG, detect_system_question


_SCOPE_LABELS = {"CONVERSATION", "SYSTEM", "DOCUMENT", "OUT_OF_SCOPE"}

# Strong signals that a question is intentionally asking about uploaded evidence.
_DOCUMENT_CUES = (
    "pdf", "document", "file", "report", "page", "section", "chapter", "table", "row", "column",
    "uploaded", "source", "citation", "according to", "mentioned in", "written in", "from the report",
    "from this", "in this", "summarize", "summarise", "extract", "find", "locate", "search for",
    "compare both", "compare the", "what does it say", "what does this say",
)
_DOCUMENT_ANALYTICS_TERMS = {
    "revenue", "sales", "profit", "margin", "cost", "expense", "expenses", "growth", "decline",
    "increase", "decrease", "trend", "kpi", "average", "total", "sum", "highest", "lowest",
    "maximum", "minimum", "product", "category", "customer", "customers", "orders", "order",
    "quarter", "monthly", "yearly", "annual", "forecast", "variance", "percentage",
}
_SYSTEM_CUES = {
    "aria", "agent", "insight agent", "rag", "retrieval", "embedding", "embeddings", "pgvector",
    "vector database", "langchain", "groq", "gpt-oss", "model", "llm", "provider", "architecture",
    "pipeline", "context engineering", "hybrid retrieval", "rrf", "citation", "citations",
}
_GENERAL_KNOWLEDGE_PATTERNS = (
    r"\bcapital\s+of\b",
    r"\bwho\s+(?:is|was)\s+(?:the\s+)?president\b",
    r"\bwho\s+(?:is|was)\s+(?:the\s+)?prime\s+minister\b",
    r"\bweather\s+(?:in|for|today|tomorrow)\b",
    r"\bcurrent\s+(?:weather|time|news|price|score)\b",
    r"\b(?:football|cricket|basketball|tennis)\s+(?:score|match|result)\b",
    r"\brecipe\s+for\b",
    r"\bhow\s+far\s+is\b",
)
_CASUAL_PATTERNS = (
    r"^(?:that'?s|this\s+is)\s+(?:good|great|nice|interesting|helpful|cool)[!?. ]*$",
    r"^(?:good|great|nice|amazing|helpful|interesting|cool)\s+(?:job|work|answer)[!?. ]*$",
    r"^(?:i\s+(?:understand|got\s+it)|makes\s+sense|sounds\s+good)[!?. ]*$",
    r"^(?:i\s+am|i'm)\s+(?:confused|not\s+sure)[!?. ]*$",
)


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _recent_assistant_intent(history: list[dict]) -> str:
    for item in reversed(history or []):
        if item.get("role") != "assistant":
            continue
        metadata = item.get("metadata") or {}
        intent = str(metadata.get("intent") or "")
        if intent:
            return intent
    return ""


def _looks_like_followup(text: str) -> bool:
    q = _norm(text)
    if not q:
        return False
    if len(q.split()) <= 5 and re.match(
        r"^(and|also|what\s+about|how\s+about|why|how|which|where|when|what|that|this|it|same|then)\b",
        q,
    ):
        return True
    return bool(re.search(r"\b(that|this|it|same|previous|earlier|above)\b", q))


def quick_scope_route(text: str, history: list[dict] | None = None) -> str | None:
    """Return an obvious scope label, otherwise ``None`` for semantic routing.

    This cheap layer keeps common turns fast and avoids an LLM classification call
    when the intent is clear.
    """
    history = history or []
    q = _norm(text)
    if not q:
        return None

    basic = route_message(text)
    if basic == "smalltalk":
        return "CONVERSATION"
    if basic == "capability" or detect_system_question(text):
        return "SYSTEM"

    if any(re.search(pattern, q, re.IGNORECASE) for pattern in _CASUAL_PATTERNS):
        return "CONVERSATION"

    recent_intent = _recent_assistant_intent(history)
    if _looks_like_followup(text):
        if recent_intent.startswith("document") or recent_intent in {"text_search", "clarification"}:
            return "DOCUMENT"
        if recent_intent in {"capability", "system"}:
            return "SYSTEM"

    if any(pattern in q for pattern in _DOCUMENT_CUES):
        return "DOCUMENT"

    tokens = set(re.findall(r"[a-z0-9]+", q))
    if tokens & _DOCUMENT_ANALYTICS_TERMS:
        return "DOCUMENT"

    # Explicit assistant/RAG vocabulary is system scope unless the user clearly
    # asks what a document says about that term.
    if any(cue in q for cue in _SYSTEM_CUES):
        if not re.search(r"\b(?:in|inside|from|mentioned|described|written)\s+(?:this|the|my)?\s*(?:pdf|document|file|report)\b", q):
            return "SYSTEM"

    if any(re.search(pattern, q, re.IGNORECASE) for pattern in _GENERAL_KNOWLEDGE_PATTERNS):
        return "OUT_OF_SCOPE"

    return None


@dataclass(frozen=True)
class ScopeDecision:
    label: str
    reason: str
    classifier_used: bool = False


class InsightPDFRAG(SystemAwareInsightPDFRAG):
    """Final bounded assistant: conversational + system-aware + document-grounded."""

    def __init__(self, *, insight_agent, user_id: str | int):
        super().__init__(insight_agent=insight_agent, user_id=user_id)
        self.llm.models["rag_scope"] = RAG_LLM_MODEL

    def _runtime_profile(self) -> str:
        agent_name = self.insight_agent.__class__.__name__
        version = getattr(self.insight_agent, "VERSION", None)
        version_text = f" v{version}" if version else ""
        provider = str(getattr(self.llm, "provider", "cloud") or "cloud")
        return (
            f"Assistant: ARIA {agent_name}{version_text}.\n"
            "Role: the existing Insight Agent with PDF RAG as an internal capability, not a fifth agent.\n"
            f"RAG LLM: {RAG_LLM_MODEL} via {provider}.\n"
            f"Embeddings: {EMBEDDING_MODEL}, local through FastEmbed.\n"
            "Vector store: PostgreSQL + pgvector through LangChain PGVector.\n"
            "Retrieval: hybrid vector + lexical retrieval with reciprocal-rank fusion (RRF), adaptive context selection, evidence gating and citations.\n"
            "Tables: extracted separately and supported by deterministic pandas-based calculations.\n"
            "Conversation: supports greetings, system questions, document follow-ups, source requests, rephrasing and translation of previous answers.\n"
            "Grounding policy: document facts must come from uploaded PDF evidence; if evidence is insufficient, ARIA says it could not find the information.\n"
            "Scope policy: ARIA does not answer unrelated general world-knowledge/current-affairs/trivia questions.\n"
            "Current document scope: digitally generated/text PDFs and extractable tables; scanned/image-only PDFs, handwriting and image/chart understanding are outside the current scope."
        )

    def _history_for_scope(self, history: list[dict]) -> str:
        lines = []
        for item in history[-6:]:
            role = str(item.get("role") or "user").upper()
            content = str(item.get("content") or "")[:500]
            metadata = item.get("metadata") or {}
            intent = metadata.get("intent")
            suffix = f" [intent={intent}]" if intent else ""
            lines.append(f"{role}: {content}{suffix}")
        return "\n".join(lines) or "(none)"

    def _semantic_scope_route(self, question: str, history: list[dict], has_documents: bool) -> ScopeDecision:
        """Use the LLM only as a classifier; it is never allowed to answer here."""
        prompt = [
            {
                "role": "system",
                "content": (
                    "Classify the latest message for a bounded ARIA PDF assistant. Return exactly one label: "
                    "CONVERSATION, SYSTEM, DOCUMENT, or OUT_OF_SCOPE.\n"
                    "CONVERSATION = social interaction only (greeting, thanks, acknowledgement, user is confused).\n"
                    "SYSTEM = asks about ARIA, its Insight Agent, RAG, model, embeddings, vector DB, architecture, behavior, limits, citations, or how the system works.\n"
                    "DOCUMENT = the answer should come from uploaded PDF evidence, including summaries, facts, numbers, tables, comparisons, extraction, or a follow-up to a prior PDF answer.\n"
                    "OUT_OF_SCOPE = unrelated general knowledge, trivia, geography, politics/current affairs, weather, sports, recipes, or other world facts.\n"
                    "Important: do not answer the question. Do not use world knowledge to provide facts. "
                    "Use recent conversation intent to resolve short follow-ups. If a business/data question clearly refers to a report or prior PDF discussion, choose DOCUMENT."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Uploaded PDFs available: {'yes' if has_documents else 'no'}\n"
                    f"Recent conversation:\n{self._history_for_scope(history)}\n\n"
                    f"Latest message:\n{question}\n\nLabel:"
                ),
            },
        ]
        try:
            raw = self.llm.chat(
                "rag_scope",
                prompt,
                temperature=0.0,
                num_predict=12,
                timeout=8,
            ).strip().upper()
            label = next((candidate for candidate in _SCOPE_LABELS if raw.startswith(candidate)), None)
            if label:
                return ScopeDecision(label=label, reason="semantic scope classifier", classifier_used=True)
        except Exception:
            pass

        # Conservative failure policy: preserve grounded follow-ups, otherwise do
        # not accidentally turn the model into a general-knowledge assistant.
        recent_intent = _recent_assistant_intent(history)
        if recent_intent.startswith("document") or recent_intent == "text_search":
            return ScopeDecision("DOCUMENT", "classifier unavailable; continuing grounded document context")
        if recent_intent in {"capability", "system"} and _looks_like_followup(question):
            return ScopeDecision("SYSTEM", "classifier unavailable; continuing system context")
        return ScopeDecision("OUT_OF_SCOPE", "classifier unavailable; conservative scope boundary")

    def _scope_decision(self, question: str, history: list[dict], documents: list[dict]) -> ScopeDecision:
        quick = quick_scope_route(question, history)
        if quick:
            return ScopeDecision(quick, "deterministic fast route")
        return self._semantic_scope_route(question, history, bool(documents))

    @staticmethod
    def _out_of_scope_answer() -> str:
        return (
            "I’m focused on ARIA itself and your uploaded documents, so I don’t answer unrelated general-knowledge questions. "
            "You can ask me about how ARIA works, the Insight Agent/RAG system, or ask a question that should be answered from your PDFs."
        )

    def _system_context_answer(self, question: str, history: list[dict]) -> str:
        """Answer broader ARIA questions only from the runtime system profile."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ARIA's Insight Agent speaking about your own implementation. Answer ONLY from the supplied ARIA runtime profile. "
                    "Do not add outside facts, generic product claims, or undocumented capabilities. If the profile does not support the answer, say that clearly. "
                    "Be concise and conversational."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ARIA runtime profile:\n{self._runtime_profile()}\n\n"
                    f"Recent conversation (context only):\n{self._history_for_scope(history)}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        try:
            return self.llm.chat(
                "rag",
                messages,
                temperature=0.05,
                num_predict=450,
                timeout=20,
            ).strip()
        except Exception:
            return "I can explain ARIA and how this Insight Agent works, but I’m having trouble reaching the language model right now."

    def _conversation_answer(self, question: str) -> str:
        q = _norm(question)
        if "confused" in q or "not sure" in q:
            return "No problem. Tell me which part is unclear, and I’ll explain that part of ARIA or your document workflow more simply."
        if any(word in q for word in ("interesting", "helpful", "great", "nice", "cool", "good job", "great job")):
            return "Glad that helped. You can keep chatting with me, ask about ARIA, or ask something from your uploaded PDFs."
        return "I’m here and ready. You can ask about ARIA or anything that should be answered from your uploaded PDFs."

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

        conversation_id = conversation_id or uuid.uuid4().hex
        history = self.conversations.load(conversation_id).get("messages", [])
        documents = self.metadata.list_documents()
        decision = self._scope_decision(question, history, documents)

        # Existing layers are richer for their known deterministic routes (hello,
        # direct model/agent questions, document metadata, retrieval, etc.).
        if decision.label == "DOCUMENT":
            return super().chat(question, conversation_id=conversation_id, document_ids=document_ids)

        if decision.label == "SYSTEM":
            # Let the existing system-aware layer answer direct known questions
            # without an LLM; use the profile-grounded LLM only for broader system
            # questions/follow-ups.
            if detect_system_question(question) or route_message(question) == "capability":
                result = super().chat(question, conversation_id=conversation_id, document_ids=document_ids)
                result.setdefault("scope", decision.label)
                return result
            answer = self._system_context_answer(question, history)
            intent = "system"
        elif decision.label == "CONVERSATION":
            # Existing fast replies are best for greetings/thanks/how-are-you.
            if route_message(question) == "smalltalk":
                result = super().chat(question, conversation_id=conversation_id, document_ids=document_ids)
                result.setdefault("scope", decision.label)
                return result
            answer = self._conversation_answer(question)
            intent = "smalltalk"
        else:
            answer = self._out_of_scope_answer()
            intent = "out_of_scope"

        self._remember(
            conversation_id,
            question,
            answer,
            intent=intent,
            document_ids=[],
        )
        result = self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent=intent,
            retrieval="not_used",
            evidence_status="not_applicable",
            model=RAG_LLM_MODEL if decision.label == "SYSTEM" and not detect_system_question(question) else None,
            context_plan={
                "scope": decision.label,
                "scope_reason": decision.reason,
                "scope_classifier_used": decision.classifier_used,
            },
        )
        result["scope"] = decision.label
        result["citation_status"] = "not_applicable"
        result["grounding"] = {
            "intent": intent,
            "evidence_status": "not_applicable",
            "citation_status": "not_applicable",
            "used_pdf_evidence": False,
            "retrieval": "not_used",
        }
        return result
