# ARIA Advanced Insight Agent

This upgrade is **additive**. The existing `insight_agent.py` is intentionally left unchanged and remains the current application runtime implementation.

The architecture remains:

`Schema Agent -> Goal Agent -> processed_data.json -> Insight Agent -> insights.json`

The new `insight_agent_advanced.py` extends the existing `InsightAgent` as a subclass. This lets us improve the Insight layer without deleting, rewriting, or weakening the working KPI, chart, hypothesis, time-series prediction, prescription, and storytelling code.

Prediction/encoding stays in the Insight layer. It is intentionally **not** moved into the Schema Agent.

## Added from the earlier ARIA Insight upgrade

- automatic `chart_recommendation`
- manager-friendly `anomalies[*].natural_language`
- `anomaly_summary`
- mixed-table `encoded_prediction`
- numeric median imputation + standard scaling
- categorical most-frequent imputation + one-hot encoding
- date expansion to year/month/day/day-of-week
- identifier/key and high-cardinality leakage protection
- held-out prediction metrics, reliability, feature importance, prediction samples
- legacy `predictions` time-series output preserved

## Additional advanced capabilities

- `drivers`: ranked numeric associations with explicit non-causal wording
- `segments`: leading/weakest category groups and concentration evidence
- `data_quality`: nulls, duplicates, high-null and constant columns
- `insight_confidence`: evidence/data-quality based confidence score
- `executive_summary`: headline, supporting evidence and recommended checks
- `analysis_version`: current advanced contract version

## Existing files preserved

These existing project files are not required to be rewritten for the advanced implementation:

- `insight_agent.py`
- `api/query_routes.py`
- `prediction/__init__.py`
- `requirements.txt`

The advanced work is isolated in new files so it can be tested first and integrated only when desired.

## New files

- `insight_agent_advanced.py`
- `prediction/tabular.py`
- `requirements_insight_advanced.txt`
- `tests/test_encoded_prediction.py`
- `tests/test_advanced_insight_agent.py`

## Install and test the advanced layer

```bash
pip install -r requirements.txt
pip install -r requirements_insight_advanced.txt
pytest -q tests/test_encoded_prediction.py tests/test_advanced_insight_agent.py
```

The current application continues using the original agent until the advanced layer is explicitly selected for integration.
