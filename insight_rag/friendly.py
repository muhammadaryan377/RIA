"""User-friendly conversation layer for ARIA's Insight Agent PDF-RAG capability."""

from __future__ import annotations

import uuid

from .config import RECENT_HISTORY_TURNS, RAG_LLM_MODEL
from .intent import (
    conversational_reply,
    is_broad_document_query,
    lexical_evidence_score,
    route_message,
)
from .service import InsightPDFRAG as BaseInsightPDFRAG
from .table_reasoner import build_table_facts


class InsightPDFRAG(BaseInsightPDFRAG):
    """Adds small-talk routing and an evidence gate without changing the RAG core.

    Obvious conversation is handled locally and never touches embeddings,
    pgvector, or the cloud LLM. Document questions still use the existing hybrid
    retrieval path. Weak evidence is checked by a tiny semantic verifier before
    answer generation to reduce hallucinations.
    """

    def __init__(self, *, insight_agent, user_id: str | int):
        super().__init__(insight_agent=insight_agent, user_id=user_id)
        self.llm.models["rag_verify"] = RAG_LLM_MODEL

    @staticmethod
    def _base_result(
        *,
        conversation_id: str,
        question: str,
        answer: str,
        intent: str,
        document_ids: list[str] | None = None,
        sources: list[dict] | None = None,
        search_query: str | None = None,
        retrieval: str = "not_used",
        evidence_status: str = "not_applicable",
        model: str | None = None,
    ) -> dict:
        sources = sources or []
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "question": question,
            "search_query": search_query,
            "answer": answer,
            "sources": sources,
            "document_ids": document_ids or [],
            "retrieved_chunks": len(sources),
            "model": model,
            "retrieval": retrieval,
            "intent": intent,
            "evidence_status": evidence_status,
        }

    def _remember(self, conversation_id: str, question: str, answer: str, sources=None) -> None:
        self.conversations.append(conversation_id, "user", question)
        self.conversations.append(conversation_id, "assistant", answer, sources=sources or [])

    def _semantic_evidence_check(self, question: str, docs) -> bool:
        """Verify weak lexical matches with a very small cloud call.

        Strong lexical support skips this call entirely. The verifier returns a
        single token-like decision and never answers the user's question.
        """
        if not docs:
            return False
        if is_broad_document_query(question):
            return True

        evidence_texts = [doc.page_content for doc in docs[:8]]
        score = lexical_evidence_score(question, evidence_texts)
        if score >= 0.22:
            return True

        snippets = []
        used = 0
        for doc in docs[:6]:
            meta = doc.metadata
            block = (
                f"{meta.get('filename')} p.{meta.get('page')} "
                f"({meta.get('content_type', 'text')}):\n{doc.page_content.strip()}"
            )
            if used + len(block) > 6500 and snippets:
                break
            snippets.append(block)
            used += len(block)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence gate for a PDF question-answering system. "
                    "Decide whether the supplied PDF evidence contains enough information to answer the user's question. "
                    "Treat clear paraphrases as support, but do not use outside knowledge or guess. "
                    "Return exactly SUPPORTED or UNSUPPORTED and nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nPDF evidence:\n" + "\n\n".join(snippets),
            },
        ]
        try:
            decision = self.llm.chat(
                "rag_verify",
                messages,
                temperature=0.0,
                num_predict=12,
                timeout=10,
            ).strip().upper()
            return decision.startswith("SUPPORTED")
        except Exception:
            # Conservative fallback: accept only a moderate lexical match if the
            # verifier is temporarily unavailable.
            return score >= 0.10

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
        intent = route_message(question)

        # Friendly conversation is handled before document validation so a user
        # can say hello, thanks, or ask what ARIA can do even with no PDF loaded.
        if intent in {"smalltalk", "capability"}:
            answer = conversational_reply(question, intent)
            self._remember(conversation_id, question, answer)
            return self._base_result(
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                intent=intent,
            )

        available = {item["document_id"] for item in self.metadata.list_documents()}
        if not available:
            answer = "Please upload a PDF first, then ask me a question about it."
            self._remember(conversation_id, question, answer)
            return self._base_result(
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                intent="document_query",
                evidence_status="no_documents",
            )

        if document_ids:
            unknown = [doc_id for doc_id in document_ids if doc_id not in available]
            if unknown:
                raise ValueError("One or more selected documents do not belong to this user.")
            selected_ids = document_ids
        else:
            selected_ids = sorted(available)

        payload = self.conversations.load(conversation_id)
        history = payload.get("messages", [])
        search_query = self._rewrite_question(question, history)
        docs = self._retrieve(search_query, selected_ids)

        if not docs or not self._semantic_evidence_check(search_query, docs):
            answer = "I couldn't find that information in the uploaded PDF."
            self._remember(conversation_id, question, answer)
            return self._base_result(
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                intent="document_query",
                document_ids=selected_ids,
                search_query=search_query,
                retrieval="hybrid_rrf",
                evidence_status="insufficient",
                model=RAG_LLM_MODEL,
            )

        context, sources = self._build_context(docs)
        table_facts = build_table_facts(question, docs)
        answer = self._generate_answer(
            question=question,
            context=context,
            table_facts=table_facts,
            history=history[-RECENT_HISTORY_TURNS * 2 :],
        )
        self._remember(conversation_id, question, answer, sources=sources)

        return self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent="document_query",
            document_ids=selected_ids,
            sources=sources,
            search_query=search_query,
            retrieval="hybrid_rrf",
            evidence_status="supported",
            model=RAG_LLM_MODEL,
        )
