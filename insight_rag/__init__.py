"""PDF RAG capability used by ARIA's existing Insight Agent.

This module extends Insight Agent with conversational document intelligence; it
is intentionally not a fifth autonomous agent.
"""

from .groq_stability import apply_groq_stability_patches

# Apply provider compatibility before the public RAG class imports the semantic
# router/provider runtime.
apply_groq_stability_patches()

from .scope_layer import InsightPDFRAG
from .answer_stability import apply_answer_stability_patches

# Broad summaries use a citation-complete generation prompt so the strict
# fail-closed verifier does not reject otherwise grounded teaching summaries.
apply_answer_stability_patches()

__all__ = ["InsightPDFRAG"]
