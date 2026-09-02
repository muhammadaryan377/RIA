"""ARIA Goal Agent package.

Public API (unchanged):
    from goal_agent import GoalAgent, DEFAULT_DB_URI

Component layout (spec section 14):
    constants    - schema-agnostic synonym / goal / SQL constants
    semantics    - generic business-concept matching (section 5)
    intent       - intent parsing, KPI mapping, clarification gate (3, 7, 9)
    schema       - schema normalization + relationship graph (5, 10)
    grounding    - subject / dimension / metric resolution (6)
    sql          - SQL generation, repair, semantic validation
    sql_contract - structured output contract (9)
    suggestions  - template suggestions
    data         - execution + null-preserving cleaning
    agent        - orchestration (GoalAgent)
"""
from .agent import GoalAgent, DEFAULT_DB_URI

__all__ = ["GoalAgent", "DEFAULT_DB_URI"]
