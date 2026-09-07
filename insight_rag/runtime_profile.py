"""Structured ARIA runtime profile used for self/system questions.

The profile contains implementation facts and policy, not user-language intent
phrases. System answers are generated only from this profile so the assistant
cannot invent undocumented ARIA capabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import EMBEDDING_MODEL, RAG_LLM_MODEL, RERANKER_ENABLED, RERANKER_MODEL


@dataclass(frozen=True)
class RuntimeProfile:
    assistant: str
    agent_class: str
    agent_version: str | None
    llm_provider: str
    rag_model: str
    embedding_model: str
    vector_store: str
    retrieval_strategy: str
    reranker: str | None
    table_reasoning: str
    grounding_policy: str
    assistant_scope: str
    document_scope: str

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        fields = self.as_dict()
        return "\n".join(
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in fields.items()
            if value not in {None, ""}
        )


def build_runtime_profile(insight_agent) -> RuntimeProfile:
    llm = insight_agent.llm
    provider = str(getattr(llm, "provider", "cloud") or "cloud")
    agent_class = insight_agent.__class__.__name__
    version = getattr(insight_agent, "VERSION", None)
    return RuntimeProfile(
        assistant="ARIA Insight Agent",
        agent_class=agent_class,
        agent_version=str(version) if version is not None else None,
        llm_provider=provider,
        rag_model=RAG_LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        vector_store="PostgreSQL + pgvector through LangChain PGVector",
        retrieval_strategy="hybrid dense + lexical retrieval with reciprocal-rank fusion and adaptive evidence selection",
        reranker=RERANKER_MODEL if RERANKER_ENABLED else None,
        table_reasoning="structured table extraction with deterministic pandas-based calculations",
        grounding_policy=(
            "Document facts must be supported by uploaded PDF evidence. Missing evidence is reported as not found; "
            "the assistant must not fill document gaps with outside knowledge."
        ),
        assistant_scope=(
            "Conversational interaction, questions about ARIA and its implementation, and evidence-grounded work over owned PDFs. "
            "Unrelated general world-knowledge/current-affairs requests are outside scope."
        ),
        document_scope=(
            "Digitally generated/text PDFs and extractable tables. Scanned/image-only PDFs, handwriting, and image/chart understanding "
            "are not part of the current document pipeline."
        ),
    )
