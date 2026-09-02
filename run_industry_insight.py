"""Run ARIA's additive industry Insight Agent without changing the existing app.

Usage:
    python run_industry_insight.py processed_data.json insights_industry.json

If the paths are omitted, the defaults above are used.
"""

from __future__ import annotations

import sys

from insight_agent_industry import IndustryInsightAgent


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "processed_data.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "insights_industry.json"
    agent = IndustryInsightAgent()
    agent.analyze(input_path, output_path)


if __name__ == "__main__":
    main()
