"""PDF RAG capability used by ARIA's existing Insight Agent.

This module extends Insight Agent with conversational document intelligence; it
is intentionally not a fifth autonomous agent.
"""

from .groq_stability import apply_groq_stability_patches

# Apply narrow Groq structured-output/rate-limit compatibility before the public
# RAG class imports the semantic router and provider runtime.
apply_groq_stability_patches()

from .scope_layer import InsightPDFRAG

__all__ = ["InsightPDFRAG"]
