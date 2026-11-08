import pandas as pd
import pytest

from engine import clean_demand_data, generate_demo_data, run_forecast, simulate_inventory


def test_demo_model_predicts():
    result = run_forecast(generate_demo_data())
    assert len(result.eval_frame) == 20
    assert result.mae >= 0
    assert result.next_prediction >= 0
    assert result.training_rows == 63
    assert result.baseline_quantity == 27


def test_stock_expiration_fifo():
    eval_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "demand_kg": [0.0, 0.0, 0.0],
        "predicted_kg": [0.0, 0.0, 0.0],
    })
    rows, result = simulate_inventory(eval_df, "fixed", 1, 5, 0, 1, 3, 2)
    assert result["ordered_kg"] == 15
    assert result["waste_kg"] == 10
    assert result["ending_inventory_kg"] == 5
    assert rows["cost_krw"].sum() == result["total_cost"]


def test_invalid_duplicate_dates():
    df = generate_demo_data()
    df.loc[2, "date"] = df.loc[1, "date"]
    with pytest.raises(ValueError, match="같은 날짜"):
        clean_demand_data(df)
