"""Offline benchmark CLI for captured ARIA PDF-RAG results.

Input JSONL format, one case per line:
{"id":"case-1","result":{...chat result...},"expected":{...gold contract...}}

Example:
    python rag_eval.py data/rag_benchmark_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insight_rag.evaluation import aggregate_scores, score_case


def load_cases(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(payload, dict) or "result" not in payload or "expected" not in payload:
            raise SystemExit(f"Line {line_number} must contain result and expected objects")
        rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Score captured ARIA PDF-RAG results without an LLM")
    parser.add_argument("jsonl", type=Path, help="Benchmark JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    cases = load_cases(args.jsonl)
    scored = []
    details = []
    for index, case in enumerate(cases, start=1):
        score = score_case(case["result"], case["expected"])
        scored.append(score)
        details.append({"id": case.get("id") or f"case-{index}", **score})

    report = {"summary": aggregate_scores(scored), "cases": details}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
