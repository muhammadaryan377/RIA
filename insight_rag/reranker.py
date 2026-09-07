"""Optional local cross-encoder reranking for hybrid PDF retrieval."""

from __future__ import annotations

from threading import Lock

from langchain_core.documents import Document

from .config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_K


class LocalCrossEncoderReranker:
    """Lazy FastEmbed reranker with fail-open retrieval behaviour.

    The first call downloads/loads the configured ONNX reranker. If loading or
    inference fails, retrieval continues in the original hybrid-RRF order.
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
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                cls._model = TextCrossEncoder(model_name=RERANKER_MODEL)
            except Exception:
                cls._load_failed = True
                return None
        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: list[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []
        model = cls._get_model()
        limit = max(1, int(top_k or RERANKER_TOP_K))
        if model is None:
            return documents[:limit]

        try:
            texts = [doc.page_content for doc in documents]
            scores = list(model.rerank(query, texts))
            if len(scores) != len(documents):
                return documents[:limit]
            ranked = sorted(
                zip(scores, documents),
                key=lambda item: float(item[0]),
                reverse=True,
            )
            return [doc for _, doc in ranked[:limit]]
        except Exception:
            return documents[:limit]
