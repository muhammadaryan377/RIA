"""Request-local execution diagnostics with privacy-safe observability.

Diagnostics deliberately contain stage names, counts, statuses and timings only.
User questions, PDF text, API keys and prompts must never be written here.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

TRACE = ContextVar("aria_rag_trace", default=None)


def record(name: str, value) -> None:
    trace = TRACE.get()
    if trace is not None:
        trace[name] = value


def increment(name: str, amount: int = 1) -> None:
    trace = TRACE.get()
    if trace is None:
        return
    counters = trace.setdefault("counters", {})
    counters[name] = int(counters.get(name, 0) or 0) + int(amount)


def event(name: str, value=True) -> None:
    trace = TRACE.get()
    if trace is None:
        return
    events = trace.setdefault("events", {})
    events[name] = value


@contextmanager
def stage(name: str):
    """Measure one pipeline stage without storing request content."""
    started = time.perf_counter()
    try:
        yield
    finally:
        trace = TRACE.get()
        if trace is None:
            return
        timings = trace.setdefault("timings_ms", {})
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        timings[name] = round(float(timings.get(name, 0.0) or 0.0) + elapsed, 2)
