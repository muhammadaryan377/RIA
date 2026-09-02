import pandas as pd

from insight_extensions import DecisionIntelligence


def _retail_frame():
    rows = []
    for month, multiplier in [("2026-01", 1.0), ("2026-02", 1.35)]:
        for region, region_factor in [("North", 1.0), ("South", 1.8)]:
            for category, category_factor in [("Beverages", 1.0), ("Electronics", 2.2)]:
                for i in range(6):
                    qty = 2 + i
                    discount = 0.05 if category == "Beverages" else 0.18
                    sales = 100 * multiplier * region_factor * category_factor + qty * 6
                    profit = sales * (0.28 - discount * 0.55)
                    rows.append(
                        {
                            "order_date": f"{month}-{i + 1:02d}",
                            "region": region,
                            "category": category,
                            "discount": discount,
                            "quantity": qty,
                            "sales": sales,
                            "profit": profit,
                        }
                    )
    return pd.DataFrame(rows)


def _processed(df):
    return {
        "user_goal": "Why did sales change by region and category and how effective were discounts?",
        "goal": {
            "original_question": "Why did sales change by region and category and how effective were discounts?",
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
            "filters": [],
            "tables": ["orders", "order_details", "products", "categories"],
        },
        "data": df.to_dict(orient="records"),
    }


def test_goal_profile_uses_goal_agent_plan():
    df = _retail_frame()
    engine = DecisionIntelligence()
    profile = engine.goal_profile(
        _processed(df),
        df,
        ["discount", "quantity", "sales", "profit"],
        ["region", "category"],
        ["order_date"],
    )

    assert profile["primary_measure"] == "sales"
    assert profile["dimensions"][:2] == ["region", "category"]
    assert profile["time_dimension"] == "order_date"
    assert "discount_effectiveness" in profile["focus"]
    assert "drivers" in profile["focus"]


def test_period_contribution_and_drilldown_are_evidence_backed():
    df = _retail_frame()
    engine = DecisionIntelligence()

    period = engine.period_comparison(df, "sales", "order_date")
    assert period is not None
    assert period["current_period"].startswith("2026-02")
    assert period["previous_period"].startswith("2026-01")
    assert period["absolute_change"] > 0

    contributions = engine.contribution_analysis(
        df,
        "sales",
        ["region", "category"],
        "order_date",
        period,
    )
    assert contributions
    assert contributions[0]["contributors"]
    assert "largest" in contributions[0]["natural_language"].lower()

    drilldown = engine.root_cause_drilldown(
        df,
        "sales",
        ["region", "category"],
        "order_date",
        period,
    )
    assert drilldown is not None
    assert len(drilldown["path"]) >= 1
    assert "does not by itself prove causation" in drilldown["natural_language"].lower()


def test_full_decision_analysis_ranks_goal_aligned_findings():
    df = _retail_frame()
    engine = DecisionIntelligence()
    result = engine.analyze(
        df,
        _processed(df),
        ["discount", "quantity", "sales", "profit"],
        ["region", "category"],
        ["order_date"],
        hypotheses=[
            {
                "hypothesis": "South region appears to be a major contributor to sales.",
                "basis": "segment comparison",
                "confidence": "medium",
            }
        ],
        anomalies=[],
        trends=[],
        drivers={"target": "sales", "associations": []},
        segments=[],
        encoded_prediction=None,
    )

    assert result["goal_profile"]["primary_measure"] == "sales"
    assert result["period_comparison"] is not None
    assert result["contribution_analysis"]
    assert result["root_cause_drilldown"] is not None
    assert result["ranked_insights"]
    assert result["ranked_insights"][0]["score"] >= result["ranked_insights"][-1]["score"]
    assert result["evidence_model"]["facts"]
    assert result["evidence_model"]["next_checks"]


def test_discount_signal_never_claims_causation():
    df = _retail_frame()
    engine = DecisionIntelligence()
    signals = engine.retail_signals(
        df,
        "sales",
        ["discount", "quantity", "sales", "profit"],
        ["region", "category"],
    )

    discount = [s for s in signals if s["type"] == "discount_effectiveness"]
    assert discount
    assert "does not prove" in discount[0]["natural_language"].lower()
