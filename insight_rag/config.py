"""Configuration for ARIA Insight Agent's conversational PDF-RAG capability."""

import os
from pathlib import Path

from core.config import DATA_DIR

RAG_PIPELINE_VERSION = "2.0-enterprise"

RAG_ROOT = DATA_DIR / "insight_rag"
RAG_USERS_DIR = RAG_ROOT / "users"
RAG_USERS_DIR.mkdir(parents=True, exist_ok=True)

RAG_DATABASE_URL = (
    os.getenv("ARIA_RAG_DATABASE_URL")
    or os.getenv("RAG_DATABASE_URL")
    or ""
).strip()

EMBEDDING_MODEL = os.getenv("ARIA_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# During the Groq -> DeepSeek migration an existing local .env may still contain
# ARIA_RAG_LLM_MODEL=openai/gpt-oss-20b. Do not send that legacy Groq model name
# to DeepSeek. Explicit DeepSeek model overrides remain supported.
_requested_rag_model = os.getenv("ARIA_RAG_LLM_MODEL", "").strip()
RAG_LLM_MODEL = (
    _requested_rag_model
    if _requested_rag_model.startswith("deepseek-")
    else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"
)
RAG_COLLECTION_PREFIX = os.getenv("ARIA_RAG_COLLECTION_PREFIX", "aria_insight_pdf")

# Semantic routing is model-driven. User utterances are not mapped through phrase
# lists or regex intents; the router returns a validated schema.
ROUTER_MIN_CONFIDENCE = float(os.getenv("ARIA_RAG_ROUTER_MIN_CONFIDENCE", "0.62"))
ROUTER_TIMEOUT_SECONDS = int(os.getenv("ARIA_RAG_ROUTER_TIMEOUT_SECONDS", "10"))

# Local cross-encoder reranking improves precision after hybrid retrieval while
# preserving a graceful fallback when the model is unavailable.
RERANKER_ENABLED = os.getenv("ARIA_RAG_RERANKER_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RERANKER_MODEL = os.getenv(
    "ARIA_RAG_RERANKER_MODEL",
    "Xenova/ms-marco-MiniLM-L-6-v2",
)
RERANKER_TOP_K = int(os.getenv("ARIA_RAG_RERANKER_TOP_K", "10"))

MAX_PDF_MB = int(os.getenv("ARIA_RAG_MAX_PDF_MB", "50"))
MAX_SELECTED_DOCUMENTS = int(os.getenv("ARIA_RAG_MAX_SELECTED_DOCUMENTS", "8"))
TEXT_CHUNK_SIZE = int(os.getenv("ARIA_RAG_TEXT_CHUNK_SIZE", "1500"))
TEXT_CHUNK_OVERLAP = int(os.getenv("ARIA_RAG_TEXT_CHUNK_OVERLAP", "220"))
TABLE_ROWS_PER_CHUNK = int(os.getenv("ARIA_RAG_TABLE_ROWS_PER_CHUNK", "25"))
RETRIEVAL_FETCH_K = int(os.getenv("ARIA_RAG_RETRIEVAL_FETCH_K", "18"))
RETRIEVAL_TOP_K = int(os.getenv("ARIA_RAG_RETRIEVAL_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("ARIA_RAG_MAX_CONTEXT_CHARS", "24000"))
RECENT_HISTORY_TURNS = int(os.getenv("ARIA_RAG_RECENT_HISTORY_TURNS", "8"))


def validate_rag_config() -> None:
    """Fail fast on unsafe or internally inconsistent RAG settings."""
    errors: list[str] = []
    if not (0.0 <= ROUTER_MIN_CONFIDENCE <= 1.0):
        errors.append("ARIA_RAG_ROUTER_MIN_CONFIDENCE must be between 0 and 1")
    if ROUTER_TIMEOUT_SECONDS < 1:
        errors.append("ARIA_RAG_ROUTER_TIMEOUT_SECONDS must be >= 1")
    if TEXT_CHUNK_SIZE < 256:
        errors.append("ARIA_RAG_TEXT_CHUNK_SIZE must be >= 256")
    if TEXT_CHUNK_OVERLAP < 0 or TEXT_CHUNK_OVERLAP >= TEXT_CHUNK_SIZE:
        errors.append("ARIA_RAG_TEXT_CHUNK_OVERLAP must be >= 0 and smaller than chunk size")
    if TABLE_ROWS_PER_CHUNK < 1:
        errors.append("ARIA_RAG_TABLE_ROWS_PER_CHUNK must be >= 1")
    if RETRIEVAL_TOP_K < 1 or RETRIEVAL_FETCH_K < RETRIEVAL_TOP_K:
        errors.append("retrieval fetch_k must be >= top_k >= 1")
    if MAX_CONTEXT_CHARS < 4000:
        errors.append("ARIA_RAG_MAX_CONTEXT_CHARS must be >= 4000")
    if MAX_PDF_MB < 1:
        errors.append("ARIA_RAG_MAX_PDF_MB must be >= 1")
    if not (1 <= MAX_SELECTED_DOCUMENTS <= 20):
        errors.append("ARIA_RAG_MAX_SELECTED_DOCUMENTS must be between 1 and 20")
    if RECENT_HISTORY_TURNS < 1:
        errors.append("ARIA_RAG_RECENT_HISTORY_TURNS must be >= 1")
    if errors:
        raise RuntimeError("Invalid ARIA RAG configuration: " + "; ".join(errors))


def user_key(user_id: str | int) -> str:
    """Return a filesystem/collection-safe stable key for one authenticated user."""
    import hashlib

    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:20]


def user_dir(user_id: str | int) -> Path:
    path = RAG_USERS_DIR / user_key(user_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "documents").mkdir(parents=True, exist_ok=True)
    (path / "conversations").mkdir(parents=True, exist_ok=True)
    (path / "chunks").mkdir(parents=True, exist_ok=True)
    return path
