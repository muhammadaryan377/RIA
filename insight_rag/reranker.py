"""Optional local cross-encoder reranking for hybrid PDF retrieval."""

from __future__ import annotations

from threading import Lock

from langchain_core.documents import Document

from .config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_K
from .diagnostics import record, stage
from .retrieval import deduplicate_chunks


class LocalCrossEncoderReranker:
    """Lazy FastEmbed reranker with fail-open retrieval behaviour.

    The first call downloads/loads the configured ONNX reranker. If loading or
    inference fails, retrieval continues in the original hybrid-RRF order.
    Successful inference attaches score/rank metadata for auditability.
    """

    _model = None
    _lock = Lock()
    _load_failed = False

    @classmethod
    def _get_model(cls):
        if not RERANKER_ENABLED or cls._load_failed:
            return None
        if cls._model is not None:
            return cls._model
        with cls._lock:
            if cls._model is not None:
                return cls._model
            try:
                with stage("reranker_model_load"):
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    cls._model = TextCrossEncoder(model_name=RERANKER_MODEL)
            except Exception:
                cls._load_failed = True
                return None
        return cls._model

    @staticmethod
    def _scored_copy(doc: Document, *, score: float, rank: int) -> Document:
        metadata = dict(doc.metadata)
        metadata.update(
            retrieval_reranker_score=round(float(score), 8),
            retrieval_reranker_rank=int(rank),
        )
        return Document(page_content=doc.page_content, metadata=metadata)

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: list[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        documents = deduplicate_chunks(documents)
        if not documents:
            return []
        model = cls._get_model()
        limit = max(1, int(top_k or RERANKER_TOP_K))
        record("reranker_candidate_count", len(documents))
        if model is None:
            record("reranker_status", "unavailable" if RERANKER_ENABLED else "disabled")
            return documents[:limit]

        try:
            texts = [doc.page_content for doc in documents]
            with stage("cross_encoder_rerank"):
                scores = list(model.rerank(query, texts))
            if len(scores) != len(documents):
                record("reranker_status", "invalid_scores")
                return documents[:limit]
            ranked = sorted(
                zip(scores, documents),
                key=lambda item: float(item[0]),
                reverse=True,
            )
            record("reranker_status", "applied")
            selected = ranked[:limit]
            record("reranker_selected_count", len(selected))
            return [
                cls._scored_copy(doc, score=float(score), rank=rank)
                for rank, (score, doc) in enumerate(selected, start=1)
            ]
        except Exception:
            record("reranker_status", "failed")
            return documents[:limit]
