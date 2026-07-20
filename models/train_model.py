"""Train the demand forecasting Random Forest for the Retail IDSS.

Usage (from the project root, after running `python -m src.data_pipeline`):

    python models/train_model.py

Evaluation: the last 8 weeks of data are held out as a test set. Prints the
test MAE in units and a simulated stockout rate: for each product-week in the
test window we simulate following the recommended reorder quantity under the
"safe" buffer setting (z=2, 7-day lead time) and count how often actual
demand exceeded available stock.

The final production model is then retrained on the full dataset and saved to
models/demand_model.joblib together with the category mapping, so the app can
load it without retraining.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline import CATEGORY_MAP_PATH, FEATURES_PATH  # noqa: E402
from src.forecasting import FEATURE_COLS, MODEL_PATH, predict_with_uncertainty  # noqa: E402
from src.reorder_logic import FORECAST_HORIZON_WEEKS, recommend  # noqa: E402

TEST_WEEKS = 8
LEAD_TIME_DAYS = 7
# The target is heavily right-skewed (median 3 units/week, max ~81k), so the
# forest is trained on log1p(units); predictions are mapped back per tree.
# This cut held-out MAE from ~55 to ~33 units vs training on raw counts.
RF_PARAMS = dict(
    n_estimators=100,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)


def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature table not found at {FEATURES_PATH}. "
            "Run `python -m src.data_pipeline` first."
        )
    df = pd.read_csv(FEATURES_PATH, dtype={"StockCode": str}, parse_dates=["week_start"])
    return df


def split_train_test(df: pd.DataFrame):
    """Hold out the last TEST_WEEKS calendar weeks as the test set."""
    weeks = np.sort(df["week_start"].unique())
    cutoff = weeks[-TEST_WEEKS]
    return df[df["week_start"] < cutoff], df[df["week_start"] >= cutoff]


def simulate_stockout_rate(test: pd.DataFrame, forecasts: np.ndarray, stds: np.ndarray) -> float:
    """Simulate following the safe-buffer recommendations through the test weeks.

    Each product starts at its order-up-to level (4 weeks of forecasted
    demand). Every week we apply the reorder rule; orders arrive after the
    7-day lead time (i.e. at the start of the next weekly review). A
    product-week is a stockout when actual demand exceeds stock on hand.
    """
    sim = test[["StockCode", "week_start", "units_sold"]].copy()
    sim["forecast"] = forecasts
    sim["forecast_std"] = stds
    sim = sim.sort_values(["StockCode", "week_start"])

    stockout_weeks = 0
    total_weeks = 0
    for _, grp in sim.groupby("StockCode", sort=False):
        f = grp["forecast"].to_numpy()
        s = grp["forecast_std"].to_numpy()
        actual = grp["units_sold"].to_numpy()
        stock = FORECAST_HORIZON_WEEKS * max(f[0], 0.0)
        incoming = 0.0
        for w in range(len(grp)):
            stock += incoming  # last week's order arrives (7-day lead time)
            incoming = 0.0
            rec = recommend(stock, f[w], s[w], LEAD_TIME_DAYS, buffer="safe")
            if rec["decision"] == "order_now":
                incoming = rec["recommended_order_qty"]
            if actual[w] > stock:
                stockout_weeks += 1
            stock = max(0.0, stock - actual[w])
            total_weeks += 1
    return stockout_weeks / total_weeks


def main():
    df = load_features()
    train, test = split_train_test(df)
    print(f"Feature table: {len(df):,} rows, {df['StockCode'].nunique():,} products")
    print(f"Train: {len(train):,} rows (through {train['week_start'].max().date()}) | "
          f"Test: {len(test):,} rows (last {TEST_WEEKS} weeks)")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(train[FEATURE_COLS], np.log1p(train["units_sold"]))

    pred_mean, pred_std = predict_with_uncertainty(model, test[FEATURE_COLS], log_target=True)
    mae = mean_absolute_error(test["units_sold"], pred_mean)
    stockout_rate = simulate_stockout_rate(test, pred_mean, pred_std)

    print(f"\nTest MAE: {mae:.2f} units per product-week")
    print(f"Simulated stockout rate (safe buffer, z=2, {LEAD_TIME_DAYS}-day lead): "
          f"{stockout_rate:.1%} of product-weeks")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).round(3).to_string())

    # Retrain on the full dataset for the production model the app loads.
    print("\nRetraining on full dataset for production model...")
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(df[FEATURE_COLS], np.log1p(df["units_sold"]))

    category_map = pd.read_csv(CATEGORY_MAP_PATH, dtype={"StockCode": str})
    bundle = {
        "model": final_model,
        "log_target": True,
        "feature_cols": FEATURE_COLS,
        "category_map": category_map,
        "trained_through": df["week_start"].max(),
        "test_mae": mae,
        "test_stockout_rate": stockout_rate,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"Saved model bundle to {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
