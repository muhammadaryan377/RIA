"""Unified Insight RAG endpoints for ARIA documents + structured data.

Existing /api/insight/pdf/* endpoints remain backwards compatible.  These routes
add one conversational surface that can automatically route to PDF RAG,
PostgreSQL/MySQL/CSV retrieval, or combine both evidence families.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.config import get_session
from core.deps import require_writable
from insight_agent_industry import InsightAgent
from insight_rag import InsightPDFRAG
from insight_rag.config import RAG_LLM_MODEL
from insight_rag.multisource import (
    MULTISOURCE_PIPELINE_VERSION,
    InsightMultiSourceRAG,
)
from insight_rag.storage import RAGMetadataStore
from insight_rag.structured_retrieval import (
    StructuredDataRetriever,
    StructuredSourceConfig,
)
from llm_provider import create_provider


router = APIRouter(prefix="/api/insight/data", tags=["Insight Multi-Source RAG"])


class MultiSourceChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=80)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    source_mode: str = Field(default="auto", max_length=20)


def _structured_config(user_id: str | int) -> tuple[StructuredSourceConfig | None, object | None]:
    session = get_session(user_id)
    db_uri = str(session.get("db_uri") or "")
    schema_path = session.get("schema_path")
    if not db_uri or not schema_path:
        return None, session.get("provider")

    config = StructuredSourceConfig(
        source_type=str(session.get("source_type") or "relational"),
        dialect=str(session.get("dialect") or session.get("db_type") or "postgresql"),
        database=str(session.get("db_name") or "connected_data"),
        schema_path=str(schema_path),
        db_uri=db_uri,
        processed_path=str(session.get("processed_path") or ""),
    )
    return config, session.get("provider")


def _source_snapshot(user_id: str | int) -> dict:
    documents = RAGMetadataStore(user_id).list_documents()
    config, provider = _structured_config(user_id)
    structured = (
        config.public_metadata()
        if config is not None
        else {"available": False, "source_type": None, "dialect": None, "database": None}
    )
    # A schema/DB connection without its session LLM provider cannot execute the
    # Goal Agent yet, so report it as connected-but-not-ready rather than usable.
    if structured.get("available") and provider is None:
        structured["available"] = False
        structured["connected"] = True
        structured["reason"] = "LLM provider session is not active"
    else:
        structured["connected"] = bool(config is not None and config.available)

    public_documents = [
        {
            key: value
            for key, value in document.items()
            if key not in {"chunk_ids", "sha256"}
        }
        for document in documents
    ]
    return {
        "ok": True,
        "documents": public_documents,
        "structured": structured,
        "capabilities": {
            "documents": bool(public_documents),
            "structured": bool(structured.get("available")),
            "hybrid": bool(public_documents and structured.get("available")),
        },
        "pipeline_version": MULTISOURCE_PIPELINE_VERSION,
    }


def _capability(user_id: str | int) -> InsightMultiSourceRAG:
    rag_provider = create_provider(
        "cloud",
        models={
            "rag": RAG_LLM_MODEL,
            "rag_rewrite": RAG_LLM_MODEL,
            "rag_verify": RAG_LLM_MODEL,
            "rag_plan": RAG_LLM_MODEL,
            "rag_scope": RAG_LLM_MODEL,
            "rag_source": RAG_LLM_MODEL,
        },
    )
    insight_agent = InsightAgent(provider=rag_provider)
    document_rag = InsightPDFRAG(insight_agent=insight_agent, user_id=user_id)

    config, structured_provider = _structured_config(user_id)
    structured = None
    if config is not None and config.available and structured_provider is not None:
        structured = StructuredDataRetriever(
            provider=structured_provider,
            config=config,
        )
    return InsightMultiSourceRAG(
        document_rag=document_rag,
        structured_retriever=structured,
    )


@router.get("/sources")
def list_sources(user: dict = Depends(require_writable)):
    """Return safe metadata for all evidence sources available to Insight RAG."""
    return _source_snapshot(user["user_id"])


@router.get("/health")
def multisource_health(user: dict = Depends(require_writable)):
    """Readiness snapshot without exposing credentials or data values."""
    snapshot = _source_snapshot(user["user_id"])
    capabilities = snapshot["capabilities"]
    ready = bool(capabilities["documents"] or capabilities["structured"])
    return {
        "ok": True,
        "ready": ready,
        "status": "ready" if ready else "waiting_for_source",
        "capabilities": capabilities,
        "structured": snapshot["structured"],
        "documents": len(snapshot["documents"]),
        "pipeline_version": MULTISOURCE_PIPELINE_VERSION,
    }


@router.post("/chat")
def chat_with_data(
    request: MultiSourceChatRequest,
    user: dict = Depends(require_writable),
):
    mode = (request.source_mode or "auto").strip().lower()
    if mode not in {"auto", "documents", "structured", "hybrid"}:
        raise HTTPException(
            status_code=400,
            detail="source_mode must be one of: auto, documents, structured, hybrid.",
        )
    try:
        return _capability(user["user_id"]).chat(
            request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids,
            source_mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "I'm having trouble with the multi-source Insight assistant right now. "
                "Your connected data has not been modified; please try again."
            ),
        )
