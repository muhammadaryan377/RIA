"""Additive advanced capabilities for ARIA's Insight Agent.

Nothing in this package replaces the existing Insight Agent. The modules here
consume the Goal Agent's processed output and enrich advanced insight results.
"""

from .decision_intelligence import DecisionIntelligence

__all__ = ["DecisionIntelligence"]
