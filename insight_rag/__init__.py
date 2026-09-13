"""RAG capabilities used by ARIA's existing Insight Agent.

The mature PDF pipeline remains available as ``InsightPDFRAG``. The
``InsightMultiSourceRAG`` orchestration layer adds structured SQL/CSV retrieval
and hybrid document + structured answers without creating a fifth autonomous
agent.
"""

from .provider_stability import apply_provider_stability_patches

# Apply cloud-provider output compatibility before the public RAG class imports
# the semantic router/provider runtime.
apply_provider_stability_patches()

from .scope_layer import InsightPDFRAG
from .answer_stability import apply_answer_stability_patches
from .verification_stability import apply_verification_stability_patches
from .summary_retry import apply_summary_retry_patch

# Broad summaries use source-bound generation, while both evidence sufficiency
# and final answer auditing use provider-stable binary verdicts. A single bounded
# retry handles rare hosted-model false negatives without bypassing verification.
apply_answer_stability_patches()
apply_verification_stability_patches()
apply_summary_retry_patch()

from .multisource import InsightMultiSourceRAG

__all__ = ["InsightPDFRAG", "InsightMultiSourceRAG"]
