"""PDF RAG capability used by ARIA's existing Insight Agent.

This module extends Insight Agent with conversational document intelligence; it
is intentionally not a fifth autonomous agent.
"""

from .provider_stability import apply_provider_stability_patches

# Apply cloud-provider output compatibility before the public RAG class imports
# the semantic router/provider runtime.
apply_provider_stability_patches()

from .scope_layer import InsightPDFRAG
from .answer_stability import apply_answer_stability_patches

# Broad summaries use a citation-complete generation prompt so the strict
# fail-closed verifier does not reject otherwise grounded teaching summaries.
apply_answer_stability_patches()

__all__ = ["InsightPDFRAG"]
