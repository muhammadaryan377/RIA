# ARIA Advanced Insight Agent

The runtime Insight API now uses `insight_agent_advanced.py`, which extends the existing stable `insight_agent.py` rather than deleting its working chart, time-series prediction, hypothesis, prescription, and storytelling behavior.

The architecture remains:

`Schema Agent -> Goal Agent -> processed_data.json -> Insight Agent -> insights.json`

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

## Runtime wiring

`api/query_routes.py` imports the advanced agent for `POST /api/insight`.

## Install and test

```bash
pip install -r requirements.txt
pytest -q tests/test_encoded_prediction.py tests/test_advanced_insight_agent.py
```

Then run the normal application and use the existing `/api/ask` -> `/api/insight` flow.
