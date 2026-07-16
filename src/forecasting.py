"""Forecasting layer: loads the trained Random Forest bundle and produces
4-week demand forecasts with uncertainty.

Uncertainty comes from the spread across the forest's individual trees:
every estimator predicts separately and we report the mean and std of those
per-tree predictions. The std drives the safety stock calculation in
src/reorder_logic.py.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "demand_model.joblib")

FEATURE_COLS = [
    "week_of_year",
    "month",
    "category",
    "avg_weekly_sales_past_4_weeks",
    "avg_weekly_sales_same_week_last_year",
    "coefficient_of_variation_weekly_demand_past_12_weeks",
    "unit_price",
    "is_major_holiday_week",
]

FORECAST_HORIZON_WEEKS = 4


def load_model_bundle(path: str = MODEL_PATH) -> dict:
    """Load the trained model bundle (model + category map + metadata).

    Raises FileNotFoundError with setup instructions if training hasn't run.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python models/train_model.py` "
            "first (after `python -m src.data_pipeline`) to train and save it."
        )
    return joblib.load(path)


def predict_with_uncertainty(model, X: pd.DataFrame, log_target: bool = True) -> tuple:
    """Predict with per-tree uncertainty.

    Returns (mean, std) arrays where mean/std are taken across the individual
    trees' predictions for each row, not just the forest's point prediction.

    The production model is trained on log1p(units) to handle the heavy skew
    in weekly demand; with `log_target=True` each tree's prediction is mapped
    back through expm1 first, so both mean and std are in units.
    """
    per_tree = np.stack([est.predict(X.values) for est in model.estimators_])
    if log_target:
        per_tree = np.expm1(per_tree)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def _holiday_week_starts() -> set:
    from src.data_pipeline import UK_HOLIDAYS, _week_start
    return set(_week_start(pd.to_datetime(pd.Series(UK_HOLIDAYS))))


def forecast_inventory(bundle: dict, features: pd.DataFrame, stock_codes: list,
                       horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
    """Recursive multi-week forecast for a set of products.

    For each future week, rolling features are recomputed from the observed
    history extended with the previous steps' predicted demand, then all
    products are predicted in one batch (one call per tree per step, so a
    40-product/4-week forecast is ~4 batched passes, not 160 single rows).

    Returns one row per (StockCode, week) with columns: StockCode, week_start,
    horizon, forecast, forecast_std.
    """
    model = bundle["model"]
    holiday_weeks = _holiday_week_starts()

    feats = features[features["StockCode"].isin(stock_codes)].copy()
    feats["week_start"] = pd.to_datetime(feats["week_start"])
    feats = feats.sort_values(["StockCode", "week_start"])
    last_week = feats["week_start"].max()

    state = {}
    for code, grp in feats.groupby("StockCode"):
        state[code] = {
            "history": grp["units_sold"].tolist(),
            "category": int(grp["category"].iloc[-1]),
            "unit_price": float(grp["unit_price"].iloc[-1]),
        }
    codes = [c for c in stock_codes if c in state]

    rows = []
    for h in range(1, horizon + 1):
        week = last_week + pd.Timedelta(weeks=h)
        iso = week.isocalendar()
        batch = []
        for code in codes:
            s = state[code]
            hist = pd.Series(s["history"])
            tail4 = hist.tail(4)
            tail12 = hist.tail(12)
            cv = float(tail12.std() / tail12.mean()) if tail12.mean() > 0 else 0.0
            last_year = hist.iloc[-52] if len(hist) >= 52 else float(tail4.mean())
            batch.append({
                "week_of_year": int(iso.week),
                "month": int(week.month),
                "category": s["category"],
                "avg_weekly_sales_past_4_weeks": float(tail4.mean()),
                "avg_weekly_sales_same_week_last_year": float(last_year),
                "coefficient_of_variation_weekly_demand_past_12_weeks": cv,
                "unit_price": s["unit_price"],
                "is_major_holiday_week": int(week.normalize() - pd.Timedelta(days=week.dayofweek) in holiday_weeks),
            })
        X = pd.DataFrame(batch)[FEATURE_COLS]
        mean, std = predict_with_uncertainty(model, X, log_target=bundle.get("log_target", True))
        mean = np.clip(mean, 0, None)
        for code, m, sd in zip(codes, mean, std):
            rows.append({"StockCode": code, "week_start": week, "horizon": h,
                         "forecast": float(m), "forecast_std": float(sd)})
            state[code]["history"].append(float(m))

    return pd.DataFrame(rows)
