"""Reproducible demand forecast and FIFO inventory simulation for a class demo.

The synthetic dataset and output are illustrative, NOT the exact original paper data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEMO_START_DATE = "2026-08-03"


@dataclass
class ForecastResult:
    history: pd.DataFrame
    eval_frame: pd.DataFrame
    next_date: pd.Timestamp
    next_prediction: float
    mae: float
    baseline_quantity: int
    training_rows: int


def generate_demo_data(n_days: int = 90, seed: int = 24) -> pd.DataFrame:
    """Create a deterministic food-ingredient consumption dataset (kg/day)."""
    if n_days < 40:
        raise ValueError("At least 40 days are required.")
    dates = pd.date_range(DEMO_START_DATE, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    weekend = (dates.dayofweek >= 5).astype(float)
    # Trend + weekly pattern + occasional demand spikes + observation noise.
    demand = (
        25.1
        + 0.019 * t
        + 3.8 * weekend
        + 2.0 * np.sin(2 * np.pi * t / 7 + 0.7)
        + rng.normal(0, 2.15, n_days)
        + (rng.random(n_days) < 0.07) * rng.uniform(2.0, 6.0, n_days)
    )
    return pd.DataFrame({"date": dates, "demand_kg": np.maximum(0, np.round(demand, 1))})


def clean_demand_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate exactly one daily nonnegative demand series."""
    required = {"date", "demand_kg"}
    if not required.issubset(raw.columns):
        raise ValueError("CSV에는 date, demand_kg 열이 필요합니다.")
    df = raw.loc[:, ["date", "demand_kg"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand_kg"] = pd.to_numeric(df["demand_kg"], errors="coerce")
    if df.isna().any().any():
        raise ValueError("날짜나 소비량에 빈 값/잘못된 형식이 있습니다.")
    if (df["demand_kg"] < 0).any():
        raise ValueError("소비량은 0 이상이어야 합니다.")
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("같은 날짜가 두 번 등장합니다.")
    if len(df) < 40:
        raise ValueError("최소 40일의 연속된 데이터가 필요합니다.")
    if (df["date"].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("날짜는 하루도 빠짐없이 연속되어야 합니다.")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features only use prior-day observations + calendar date information."""
    out = df.copy()
    out["day_idx"] = np.arange(len(out))
    dow = out["date"].dt.dayofweek
    out["weekday_sin"] = np.sin(2 * np.pi * dow / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * dow / 7)
    out["prev_demand"] = out["demand_kg"].shift(1)
    out["avg_prev_7"] = out["demand_kg"].shift(1).rolling(7).mean()
    return out


FEATURE_COLUMNS = ["weekday_sin", "weekday_cos", "prev_demand", "avg_prev_7", "day_idx"]


def run_forecast(data: pd.DataFrame, eval_days: int = 20) -> ForecastResult:
    df = clean_demand_data(data)
    if len(df) <= eval_days + 14:
        raise ValueError("훈련 데이터가 부족합니다. 40일 이상을 입력해 주세요.")
    prepared = make_features(df).dropna().reset_index(drop=True)
    train_limit_date = df["date"].iloc[-eval_days]
    train = prepared[prepared["date"] < train_limit_date].copy()
    test = prepared[prepared["date"] >= train_limit_date].copy()
    if len(train) < 15 or len(test) != eval_days:
        raise ValueError("훈련/평가 기간을 나눌 수 없습니다.")

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(12, 8),
            activation="relu",
            solver="lbfgs",
            alpha=1.0,
            max_iter=1600,
            random_state=42,
        ),
    )
    model.fit(train[FEATURE_COLUMNS], train["demand_kg"])
    test["predicted_kg"] = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
    mae = float(mean_absolute_error(test["demand_kg"], test["predicted_kg"]))

    tomorrow = df["date"].iloc[-1] + pd.Timedelta(days=1)
    day_of_week = tomorrow.dayofweek
    input_features = pd.DataFrame([{
        "weekday_sin": np.sin(2 * np.pi * day_of_week / 7),
        "weekday_cos": np.cos(2 * np.pi * day_of_week / 7),
        "prev_demand": float(df["demand_kg"].iloc[-1]),
        "avg_prev_7": float(df["demand_kg"].tail(7).mean()),
        "day_idx": len(df),
    }])
    # A production system should retrain on all known observations before
    # next-day forecasting. This is a time-ordered evaluation prototype.
    production_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(12, 8), activation="relu", solver="lbfgs",
            alpha=1.0, max_iter=1600, random_state=42,
        ),
    )
    production_model.fit(prepared[FEATURE_COLUMNS], prepared["demand_kg"])
    next_pred = max(0.0, float(production_model.predict(input_features)[0]))
    baseline = int(round(df["demand_kg"].iloc[:-eval_days].mean()))
    return ForecastResult(
        history=df,
        eval_frame=test[["date", "demand_kg", "predicted_kg"]].reset_index(drop=True),
        next_date=tomorrow,
        next_prediction=next_pred,
        mae=mae,
        baseline_quantity=baseline,
        training_rows=len(train),
    )


def simulate_inventory(
    eval_df: pd.DataFrame,
    policy: str,
    shelf_life: int,
    fixed_order: int,
    safety_stock: float,
    purchase_cost: float,
    stockout_cost: float,
    waste_cost: float,
) -> tuple[pd.DataFrame, dict]:
    """Apply the same actual demand sequence to fixed-order and AI-order policies.

    Assumptions: zero initial inventory, morning arrival, FIFO consumption,
    goods usable on receipt day and the next (shelf_life-1) days.
    """
    if policy not in {"fixed", "ai"}:
        raise ValueError("Unknown policy")
    if shelf_life < 1:
        raise ValueError("Shelf life must be >= 1")
    batches: deque[list[float]] = deque()  # [exclusive_expiry_day, kilograms]
    daily = []
    totals = {"ordered_kg": 0., "waste_kg": 0., "shortage_kg": 0., "total_cost": 0.}

    for day, row in enumerate(eval_df.itertuples(index=False)):
        expired = 0.0
        while batches and batches[0][0] <= day:
            _, amount = batches.popleft()
            expired += amount
        inventory_before_order = float(sum(batch[1] for batch in batches))
        predicted = float(row.predicted_kg)
        actual = float(row.demand_kg)
        if policy == "fixed":
            order = float(fixed_order)
        else:
            order = float(max(0, int(np.ceil(predicted + safety_stock - inventory_before_order))))
        if order > 0:
            batches.append([float(day + shelf_life), order])

        remaining = actual
        while remaining > 1e-9 and batches:
            expiry, qty = batches[0]
            taken = min(qty, remaining)
            remaining -= taken
            qty -= taken
            if qty < 1e-9:
                batches.popleft()
            else:
                batches[0] = [expiry, qty]
        short = max(0.0, remaining)
        stock_after = float(sum(batch[1] for batch in batches))
        cost = order * purchase_cost + short * stockout_cost + expired * waste_cost
        totals["ordered_kg"] += order
        totals["waste_kg"] += expired
        totals["shortage_kg"] += short
        totals["total_cost"] += cost
        daily.append({
            "date": row.date, "actual_kg": actual, "predicted_kg": predicted,
            "ordered_kg": order, "expired_kg": expired,
            "shortage_kg": short, "remaining_kg": stock_after,
            "cost_krw": cost,
        })
    totals["ending_inventory_kg"] = float(sum(b[1] for b in batches))
    return pd.DataFrame(daily), totals
