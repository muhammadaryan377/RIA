"""Enterprise bounded-assistant orchestration for ARIA Insight PDF-RAG.

All high-level language understanding is performed by ``SemanticRouter`` using a
validated schema. This layer enforces the product boundary, tenant/document
scope, request limits, privacy-safe observability and final response persistence.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

from .config import (
    MAX_SELECTED_DOCUMENTS,
    RAG_LLM_MODEL,
    RAG_PIPELINE_VERSION,
    validate_rag_config,
)
from .context_engine import ContextPlan
from .diagnostics import TRACE, record, stage
from .final_layer import InsightPDFRAG as FinalInsightPDFRAG
from .runtime_profile import build_runtime_profile
from .semantic_router import RouteDecision, SemanticRouter

_ACTIVE_ROUTE = ContextVar("aria_active_route", default=None)


class InsightPDFRAG(FinalInsightPDFRAG):
    """Production export: semantic routing + grounded document intelligence."""

    def __init__(self, *, insight_agent, user_id: str | int):
        validate_rag_config()
        super().__init__(insight_agent=insight_agent, user_id=user_id)
        self.semantic_router = SemanticRouter(self.llm)
        self.runtime_profile = build_runtime_profile(insight_agent)

    def _build_context_plan(
        self,
        question: str,
        *,
        history: list[dict],
        documents: list[dict],
        selected_document_ids: list[str] | None,
    ) -> ContextPlan:
        return self.context_engine.plan(
            question,
            history=history,
            documents=documents,
            selected_document_ids=selected_document_ids,
            route_decision=_ACTIVE_ROUTE.get(),
        )

    @staticmethod
    def _routing_payload(decision: RouteDecision) -> dict:
        return decision.as_dict()

    @staticmethod
    def _out_of_scope_answer() -> str:
        return (
            "I'm focused on ARIA itself and your uploaded documents, so I don't answer unrelated general-knowledge or current-affairs questions. "
            "You can ask about ARIA, the Insight Agent/RAG system, or something that should be answered from your PDFs."
        )

    @staticmethod
    def _document_inventory_context(documents: list[dict]) -> str:
        """Authoritative upload metadata safe to expose in non-document turns."""
        if not documents:
            return "No PDFs are currently uploaded/indexed for this user."
        lines = [f"Uploaded/indexed PDFs: {len(documents)}"]
        for index, document in enumerate(documents[:20], start=1):
            quality = (document.get("ingestion_quality") or {}).get("grade")
            quality_note = f", extraction quality {quality}" if quality else ""
            lines.append(
                f"{index}. {document.get('filename') or 'Untitled PDF'} "
                f"({int(document.get('pages', 0) or 0)} pages, "
                f"{int(document.get('tables', 0) or 0)} tables, "
                f"{int(document.get('total_chunks', 0) or 0)} chunks{quality_note})"
            )
        if len(documents) > 20:
            lines.append(f"...and {len(documents) - 20} more PDFs")
        return "\n".join(lines)

    def _system_context_answer(self, question: str, history: list[dict]) -> str:
        profile = self.runtime_profile.render()
        conversation = self.semantic_router._history_text(history)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ARIA's Insight Agent answering a question about your own implementation. "
                    "Use only the supplied runtime profile as factual authority. Do not add generic AI/product claims, outside facts or undocumented capabilities. "
                    "If the runtime profile does not support a claim, say that clearly. Keep the answer natural, concise and technically accurate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ARIA runtime profile:\n{profile}\n\n"
                    f"Recent conversation (context only):\n{conversation}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        try:
            with stage("system_answer"):
                return self.llm.chat(
                    "rag",
                    messages,
                    temperature=0.03,
                    num_predict=600,
                    timeout=20,
                ).strip()
        except Exception:
            return (
                "I can explain ARIA and this Insight Agent from its runtime configuration, "
                "but the language-model service is temporarily unavailable."
            )

    def _conversation_answer(
        self,
        question: str,
        history: list[dict],
        documents: list[dict],
    ) -> str:
        conversation = self.semantic_router._history_text(history)
        inventory = self._document_inventory_context(documents)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ARIA's bounded Insight Agent in a normal conversational turn. "
                    "Respond naturally to social interaction, acknowledgements, greetings, thanks or conversational clarification. "
                    "Do not provide world-knowledge facts, current affairs, trivia, medical/legal/financial advice, or claims about PDF contents. "
                    "The supplied PDF inventory is authoritative metadata: you may accurately acknowledge whether PDFs are uploaded/indexed and may name/count them, but do not claim to have read or searched their contents in this conversational route. "
                    "Never say that no PDF is uploaded when the inventory shows one or more PDFs. If the user's conversational wording appears to refer to an uploaded PDF, acknowledge the available PDF metadata and invite a concrete document question rather than inventing content. "
                    "Keep the response brief and friendly, and keep ARIA's scope clear when useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Authoritative PDF inventory:\n{inventory}\n\n"
                    f"Recent conversation:\n{conversation}\n\n"
                    f"Latest message:\n{question}"
                ),
            },
        ]
        try:
            with stage("conversation_answer"):
                return self.llm.chat(
                    "rag",
                    messages,
                    temperature=0.20,
                    num_predict=180,
                    timeout=12,
                ).strip()
        except Exception:
            if documents:
                count = len(documents)
                noun = "PDF" if count == 1 else "PDFs"
                return f"I'm here and ready. You currently have {count} uploaded {noun}; ask me what you'd like to find or summarize."
            return "I'm here and ready. You can ask about ARIA or upload a PDF and ask me about it."

    @staticmethod
    def _decorate_result(result: dict, decision: RouteDecision, request_id: str) -> dict:
        result["scope"] = decision.scope
        result["routing"] = decision.as_dict()
        result["request_id"] = request_id
        result["pipeline_version"] = RAG_PIPELINE_VERSION
        grounding = result.setdefault("grounding", {})
        grounding.setdefault("scope", decision.scope)
        grounding.setdefault("route_task", decision.task)
        return result

    def _chat(
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
        request_id = uuid.uuid4().hex
        record("request_id", request_id)
        history = self.conversations.load_recent(conversation_id).get("messages", [])
        documents = self.metadata.list_documents()
        with stage("semantic_router"):
            decision = self.semantic_router.decide(
                question,
                history=history,
                documents=documents,
                selected_document_ids=document_ids,
            )
        record("route_scope", decision.scope)
        record("route_task", decision.task)
        record("router_confidence", round(float(decision.confidence or 0.0), 4))

        if decision.scope == "DOCUMENT":
            token = _ACTIVE_ROUTE.set(decision)
            try:
                with stage("document_execution"):
                    result = super().chat(
                        question,
                        conversation_id=conversation_id,
                        document_ids=document_ids,
                    )
            finally:
                _ACTIVE_ROUTE.reset(token)
            return self._decorate_result(result, decision, request_id)

        if decision.scope == "CONVERSATION" and decision.task == "PREVIOUS_ANSWER":
            previous = self._last_assistant_message(history) or {}
            prior_ids = {str(source.get("document_id")) for source in previous.get("sources", [])}
            owned = {str(doc.get("document_id")) for doc in documents}
            permitted = owned & set(document_ids) if document_ids else owned
            if prior_ids - permitted:
                result = self._base_result(
                    conversation_id=conversation_id, question=question,
                    answer="The previous answer used a PDF that is no longer available or selected. Please select the relevant PDF and ask the question again.",
                    intent="clarification", evidence_status="scope_changed",
                )
                return self._decorate_result(result, decision, request_id)
            with stage("previous_answer_action"):
                result = self.handle_previous_answer(
                    question,
                    conversation_id=conversation_id,
                    history=history,
                    action=decision.previous_action,
                    transform_instruction=decision.transform_instruction,
                )
            return self._decorate_result(result, decision, request_id)

        if decision.scope == "SYSTEM":
            answer = self._system_context_answer(question, history)
            intent = "system"
            model = RAG_LLM_MODEL
        elif decision.scope == "CONVERSATION":
            answer = self._conversation_answer(question, history, documents)
            intent = "smalltalk"
            model = RAG_LLM_MODEL
        elif decision.scope == "CLARIFICATION":
            answer = decision.clarification_question or (
                "Could you clarify whether you're asking about ARIA itself or information from an uploaded PDF?"
            )
            intent = "clarification"
            model = None
        else:
            answer = self._out_of_scope_answer()
            intent = "out_of_scope"
            model = None

        result = self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent=intent,
            retrieval="not_used",
            evidence_status="not_applicable",
            model=model,
            context_plan={
                "scope": decision.scope,
                "task": decision.task,
                "router_confidence": decision.confidence,
                "router_latency_ms": decision.latency_ms,
                "router_reason": decision.reason,
            },
        )
        result["citation_status"] = "not_applicable"
        result["grounding"] = {
            "intent": intent,
            "scope": decision.scope,
            "route_task": decision.task,
            "evidence_status": "not_applicable",
            "citation_status": "not_applicable",
            "used_pdf_evidence": False,
            "retrieval": "not_used",
        }
        return self._decorate_result(result, decision, request_id)

    def chat(self, question: str, *, conversation_id: str | None = None,
             document_ids: list[str] | None = None) -> dict:
        """Serialize one conversation, enforce scope limits, and persist one visible response."""
        conversation_id = conversation_id or uuid.uuid4().hex
        if len(conversation_id) > 128:
            raise ValueError("Conversation id is too long.")

        if document_ids:
            document_ids = list(dict.fromkeys(str(value) for value in document_ids if str(value)))
            if len(document_ids) > MAX_SELECTED_DOCUMENTS:
                raise ValueError(
                    f"Select at most {MAX_SELECTED_DOCUMENTS} PDFs for one request. Narrow the selection and try again."
                )

        started = time.perf_counter()
        token = TRACE.set({"pipeline_version": RAG_PIPELINE_VERSION})
        try:
            with self.conversations.turn_lock(conversation_id):
                if document_ids:
                    owned = {str(doc.get("document_id")) for doc in self.metadata.list_documents()}
                    if any(str(did) not in owned for did in document_ids):
                        raise ValueError("One or more selected PDFs are unavailable. Refresh the document list.")

                with stage("orchestration"):
                    raw_result = self._chat(
                        question, conversation_id=conversation_id, document_ids=document_ids,
                    )
                with stage("finalize_response"):
                    result = self.finalize_result(raw_result)

                owned_now = {str(doc.get("document_id")) for doc in self.metadata.list_documents()}
                if any(str(source.get("document_id")) not in owned_now for source in result.get("sources", [])):
                    result.update(
                        answer="A source PDF was removed during this request. Please ask again using the available PDFs.",
                        sources=[], evidence_status="scope_changed",
                    )
                    with stage("finalize_scope_change"):
                        result = self.finalize_result(result)

                with stage("conversation_persist"):
                    self._remember(
                        conversation_id, result["question"], result["answer"],
                        intent=result["intent"], sources=result.get("sources", []),
                        document_ids=result.get("document_ids", []),
                        search_query=result.get("search_query"), routing=result.get("routing"),
                    )

                result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                trace = dict(TRACE.get() or {})
                trace["total_latency_ms"] = result["latency_ms"]
                result["diagnostics"] = trace
                result["pipeline_version"] = RAG_PIPELINE_VERSION
                return result
        finally:
            TRACE.reset(token)
