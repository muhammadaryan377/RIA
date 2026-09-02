import json

import pandas as pd

from insight_agent_advanced import AdvancedInsightAgent


class DummyProvider:
    def chat(self, *args, **kwargs):
        raise RuntimeError("offline in unit test")


def test_anomaly_contains_manager_friendly_language():
    agent = AdvancedInsightAgent(provider=DummyProvider())
    df = pd.DataFrame(
        {
            "region": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
            "sales": [100, 101, 99, 102, 98, 100, 101, 99, 100, 1000],
        }
    )

    anomalies = agent.detect_anomalies(df, ["sales"], ["region"], [])

    assert anomalies
    assert anomalies[0]["natural_language"]
    assert "worth checking" in anomalies[0]["natural_language"].lower()
    assert anomalies[0]["severity"] in {"medium", "high"}


def test_chart_recommendation_explains_primary_chart():
    agent = AdvancedInsightAgent(provider=DummyProvider())
    dashboard = [
        {
            "chart_type": "bar",
            "title": "sales by category",
            "x": "category",
            "y": "sales",
        }
    ]

    result = agent.chart_recommendation(dashboard)

    assert result["primary_chart_type"] == "bar"
    assert result["title"] == "sales by category"
    assert "comparing" in result["reason"].lower()


def test_driver_analysis_never_claims_causation():
    agent = AdvancedInsightAgent(provider=DummyProvider())
    df = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40, 50, 60],
            "quantity": [1, 2, 3, 4, 5, 6],
            "discount": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        }
    )

    result = agent.driver_analysis(df, ["sales", "quantity", "discount"])

    assert result["target"] == "sales"
    assert result["associations"]
    assert "not proof of cause" in result["associations"][0]["natural_language"].lower()


def test_full_analysis_preserves_legacy_and_adds_advanced_keys(tmp_path):
    agent = AdvancedInsightAgent(provider=DummyProvider())

    rows = []
    for i in range(40):
        rows.append(
            {
                "order_id": i + 1,
                "order_date": f"2026-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                "region": ["North", "South", "East"][i % 3],
                "sales": float(100 + i * 5),
                "quantity": int(1 + i % 5),
            }
        )

    processed = {
        "user_goal": "show sales performance by region and predict sales",
        "row_count": len(rows),
        "prediction_target": "sales",
        "data": rows,
    }
    input_path = tmp_path / "processed_data.json"
    output_path = tmp_path / "insights.json"
    input_path.write_text(json.dumps(processed), encoding="utf-8")

    # Keep the integration test deterministic and independent from forecast/LLM quality.
    agent.prediction.forecast = lambda *args, **kwargs: None
    agent.prescription.prescribe = lambda *args, **kwargs: {"actions": [], "source": "template"}

    agent.analyze(str(input_path), str(output_path))
    result = json.loads(output_path.read_text(encoding="utf-8"))

    # Existing contract remains available.
    assert "kpis" in result
    assert "hypotheses" in result
    assert "predictions" in result
    assert "dashboard" in result
    assert "business_story" in result

    # Advanced contract.
    assert result["analysis_version"] == "2.1"
    assert "anomaly_summary" in result
    assert "chart_recommendation" in result
    assert "encoded_prediction" in result
    assert "drivers" in result
    assert "segments" in result
    assert "data_quality" in result
    assert "insight_confidence" in result
    assert "executive_summary" in result
