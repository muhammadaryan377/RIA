import numpy as np
import pandas as pd

from prediction.tabular import TabularPredictor


def test_encoded_prediction_handles_categories_dates_and_missing_values():
    rng = np.random.default_rng(42)
    n = 120
    region = rng.choice(["North", "South", "East"], size=n)
    category = rng.choice(["Beverages", "Electronics", "Clothing"], size=n)
    discount = rng.uniform(0, 0.25, size=n)
    qty = rng.integers(1, 12, size=n)
    region_effect = pd.Series(region).map({"North": 100, "South": 50, "East": 20}).to_numpy()
    category_effect = pd.Series(category).map(
        {"Beverages": 20, "Electronics": 150, "Clothing": 70}
    ).to_numpy()
    sales = 25 * qty + category_effect + region_effect - 80 * discount + rng.normal(0, 15, size=n)

    rows = []
    for i in range(n):
        rows.append(
            {
                "order_id": i + 1,
                "order_date": f"2026-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                "region": region[i],
                "category": category[i],
                "discount": None if i % 17 == 0 else float(discount[i]),
                "quantity": int(qty[i]),
                "sales": float(sales[i]),
            }
        )

    payload = {
        "user_goal": "predict sales from region, category, quantity and discount",
        "prediction_target": "sales",
        "data": rows,
    }
    result = TabularPredictor(min_rows=20).predict(payload)

    assert result["mode"] == "encoded_tabular"
    assert result["target"] == "sales"
    assert result["task"] == "regression"
    assert "region" in result["encoding"]["categorical"]["columns"]
    assert "category" in result["encoding"]["categorical"]["columns"]
    assert result["encoding"]["date_features_created"]
    assert any(
        item["reason"].startswith("identifier")
        for item in result["encoding"]["dropped_columns"]
    )
    assert result["feature_importance"]
    assert result["prediction_samples"]


def test_encoded_prediction_returns_safe_result_for_small_dataset():
    payload = {
        "user_goal": "predict sales",
        "prediction_target": "sales",
        "data": [{"category": "A", "sales": 10}, {"category": "B", "sales": 20}],
    }
    result = TabularPredictor(min_rows=20).predict(payload)

    assert result["valid"] is False
    assert result["reason_code"] == "insufficient_data"
    assert result["mode"] == "encoded_tabular"
