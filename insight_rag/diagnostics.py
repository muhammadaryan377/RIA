"""Request-local execution diagnostics; no questions, secrets or PDF text."""

from contextvars import ContextVar

TRACE = ContextVar("aria_rag_trace", default=None)


def record(name: str, value) -> None:
    trace = TRACE.get()
    if trace is not None:
        trace[name] = value
