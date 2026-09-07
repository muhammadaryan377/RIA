"""PDF RAG capability used by ARIA's existing Insight Agent.

This module extends Insight Agent with conversational document intelligence; it
is intentionally not a fifth autonomous agent.
"""

from .conversation_layer import InsightPDFRAG

__all__ = ["InsightPDFRAG"]
