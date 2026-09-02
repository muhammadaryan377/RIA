import json

from insight_agent_industry import IndustryInsightAgent


class DummyProvider:
    def chat(self, *args, **kwargs):
        raise RuntimeError("LLM disabled in deterministic integration test")


def test_industry_agent_appends_decision_intelligence_without_removing_legacy_output(tmp_path):
    rows = []
    for month, factor in [(1, 1.0), (2, 1.25)]:
        for region, region_factor in [("North", 1.0), ("South", 1.5)]:
            for category, category_factor in [("Beverages", 1.0), ("Electronics", 2.0)]:
                for day in range(1, 7):
                    sales = 100 * factor * region_factor * category_factor + day * 4
                    rows.append(
                        {
                            "order_id": len(rows) + 1,
                            "order_date": f"2026-{month:02d}-{day:02d}",
                            "region": region,
                            "category": category,
                            "discount": 0.05 if category == "Beverages" else 0.15,
                            "quantity": day + 1,
                            "sales": sales,
                            "profit": sales * 0.22,
                        }
                    )

    processed = {
        "user_goal": "Why did sales change by region and category?",
        "goal": {
            "original_question": "Why did sales change by region and category?",
            "intent": "analysis",
            "analysis_type": "diagnostic",
        },
        "analysis_plan": {
            "measures": ["SUM(order_details.sales) AS sales"],
            "dimensions": ["region", "category"],
            "aggregation": ["SUM"],
            "group_by": ["region", "category"],
        },
        "data_selection": {
            "tables": ["orders", "order_details", "products", "categories"],
            "filters": [],
            "joins": [],
        },
        "row_count": len(rows),
        "prediction_target": "sales",
        "data": rows,
    }

    input_path = tmp_path / "processed_data.json"
    output_path = tmp_path / "insights.json"
    input_path.write_text(json.dumps(processed), encoding="utf-8")

    agent = IndustryInsightAgent(provider=DummyProvider())
    agent.prediction.forecast = lambda *args, **kwargs: None
    agent.prescription.prescribe = lambda *args, **kwargs: {"actions": [], "source": "test"}
    agent.analyze(str(input_path), str(output_path))

    result = json.loads(output_path.read_text(encoding="utf-8"))

    # Existing advanced/legacy outputs remain available.
    assert "kpis" in result
    assert "hypotheses" in result
    assert "dashboard" in result
    assert "business_story" in result
    assert "encoded_prediction" in result

    # Industry-level additive decision outputs.
    assert result["analysis_version"] == "3.0"
    assert result["goal_profile"]["primary_measure"] == "sales"
    assert result["period_comparison"] is not None
    assert result["contribution_analysis"]
    assert result["root_cause_drilldown"] is not None
    assert result["ranked_insights"]
    assert result["evidence_model"]["facts"]
    assert result["decision_brief"]["headline"]
    assert result["decision_intelligence"]["version"] == "1.0"
