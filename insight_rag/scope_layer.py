"""Enterprise bounded-assistant orchestration for ARIA Insight PDF-RAG.

All high-level language understanding is performed by ``SemanticRouter`` using a
validated schema. This layer enforces the product boundary:
- normal conversation is allowed but cannot become factual world-knowledge QA;
- ARIA/system questions are answered only from the runtime profile;
- document questions execute grounded PDF plans;
- unrelated factual requests are declined;
- ambiguous requests ask for clarification instead of guessing.
"""

from __future__ import annotations

import uuid

from .config import RAG_LLM_MODEL
from .context_engine import ContextPlan
from .final_layer import InsightPDFRAG as FinalInsightPDFRAG
from .runtime_profile import build_runtime_profile
from .semantic_router import RouteDecision, SemanticRouter


class InsightPDFRAG(FinalInsightPDFRAG):
    """Production export: semantic routing + grounded document intelligence."""

    def __init__(self, *, insight_agent, user_id: str | int):
        super().__init__(insight_agent=insight_agent, user_id=user_id)
        self.semantic_router = SemanticRouter(self.llm)
        self.runtime_profile = build_runtime_profile(insight_agent)
        self._active_route_decision: RouteDecision | None = None

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
            route_decision=self._active_route_decision,
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

    def _conversation_answer(self, question: str, history: list[dict]) -> str:
        conversation = self.semantic_router._history_text(history)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ARIA's bounded Insight Agent in a normal conversational turn. "
                    "Respond naturally to social interaction, acknowledgements, greetings, thanks or conversational clarification. "
                    "Do not provide world-knowledge facts, current affairs, trivia, medical/legal/financial advice, or document claims. "
                    "Do not pretend to have searched a PDF. Keep the response brief and friendly, and keep ARIA's scope clear when useful."
                ),
            },
            {
                "role": "user",
                "content": f"Recent conversation:\n{conversation}\n\nLatest message:\n{question}",
            },
        ]
        try:
            return self.llm.chat(
                "rag",
                messages,
                temperature=0.25,
                num_predict=180,
                timeout=12,
            ).strip()
        except Exception:
            return "I'm here and ready. You can ask about ARIA or anything that should be answered from your uploaded PDFs."

    @staticmethod
    def _decorate_result(result: dict, decision: RouteDecision, request_id: str) -> dict:
        result["scope"] = decision.scope
        result["routing"] = decision.as_dict()
        result["request_id"] = request_id
        grounding = result.setdefault("grounding", {})
        grounding.setdefault("scope", decision.scope)
        grounding.setdefault("route_task", decision.task)
        return result

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
        request_id = uuid.uuid4().hex
        history = self.conversations.load(conversation_id).get("messages", [])
        documents = self.metadata.list_documents()
        decision = self.semantic_router.decide(
            question,
            history=history,
            documents=documents,
            selected_document_ids=document_ids,
        )
        routing = self._routing_payload(decision)

        if decision.scope == "DOCUMENT":
            self._active_route_decision = decision
            try:
                result = super().chat(
                    question,
                    conversation_id=conversation_id,
                    document_ids=document_ids,
                )
            finally:
                self._active_route_decision = None
            return self._decorate_result(result, decision, request_id)

        if decision.scope == "CONVERSATION" and decision.task == "PREVIOUS_ANSWER":
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
            answer = self._conversation_answer(question, history)
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

        self._remember(
            conversation_id,
            question,
            answer,
            intent=intent,
            document_ids=[],
            routing=routing,
        )
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
