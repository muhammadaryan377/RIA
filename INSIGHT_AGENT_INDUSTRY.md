# ARIA Industry Insight Agent

`insight_agent_industry.py` is an additive v3 layer. It keeps the original `insight_agent.py` unchanged and extends the already additive `insight_agent_advanced.py`.

## Architecture

```text
Schema Agent
    -> Goal Agent
    -> processed_data.json
    -> existing InsightAgent
    -> AdvancedInsightAgent
    -> IndustryInsightAgent
    -> insights.json
```

The current application wiring is intentionally unchanged. The industry agent can be tested independently before it is selected for production integration.

## What v3 adds

### Goal-aware analysis
The engine reads the Goal Agent contract (`goal`, `analysis_plan`, `data_selection`) and resolves the requested business measure, dimensions, time column, filters and analysis focus. The result is no longer treated as a generic dataframe.

### Period-over-period comparison
When a valid time dimension exists, ARIA compares the latest two meaningful periods and reports current value, previous value, absolute change and percentage change.

### Contribution analysis
ARIA identifies which region/category/customer/product-like segment contributes most to the observed metric or period change.

### Root-cause drill-down
ARIA drills through up to three goal-relevant dimensions and identifies where a change is concentrated. The output explicitly says that concentration is not proof of causation.

### Retail-specific signals
The deterministic extension can detect supported signals such as:
- discount effectiveness associations,
- profit margin when profit and revenue-like measures exist,
- customer/product/category/region concentration,
- revenue per unit when quantity and revenue-like measures exist.

### Evidence-based ranking
Findings are ranked using magnitude, confidence, goal alignment, business relevance and available evidence. Similar findings are deduplicated.

### Fact / association / hypothesis separation
The `evidence_model` separates:
- verified facts,
- associations with causation warnings,
- hypotheses from the existing hypothesis engine,
- recommended next checks.

### Business impact
When a period comparison is available, ARIA exposes the absolute metric change, percentage change and the leading visible contributor without inventing a currency or causal explanation.

## Main new output

```json
{
  "analysis_version": "3.0",
  "goal_profile": {},
  "period_comparison": {},
  "contribution_analysis": [],
  "root_cause_drilldown": {},
  "retail_signals": [],
  "business_impact": {},
  "ranked_insights": [],
  "evidence_model": {},
  "decision_brief": {}
}
```

All previous KPI, trend, anomaly, hypothesis, dashboard, prediction, prescription and business-story fields remain available.

## Run without changing the existing app

```bash
pip install -r requirements.txt
pip install -r requirements_insight_advanced.txt
python run_industry_insight.py processed_data.json insights_industry.json
```

## Tests

```bash
pytest -q \
  tests/test_encoded_prediction.py \
  tests/test_advanced_insight_agent.py \
  tests/test_decision_intelligence.py \
  tests/test_industry_insight_agent.py
```

GitHub Actions also compiles the new modules and runs these additive Insight tests.
