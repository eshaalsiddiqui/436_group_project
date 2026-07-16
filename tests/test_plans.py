"""Tests for the three reorder plans and the demo promotion flip.

The final test exercises the real trained model on the demo inventory to
guarantee the promotion toggle actually flips at least one product from
"don't order" to "order now" in the app; it is skipped if the pipeline/model
artifacts haven't been built yet.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import plans
from src.data_pipeline import FEATURES_PATH
from src.forecasting import MODEL_PATH


@pytest.fixture
def recs():
    return pd.DataFrame([
        # code, decision, days_until_stockout, revenue, cv
        {"StockCode": "A", "decision": "order_now", "days_until_stockout": 9.0,
         "weekly_revenue_at_risk": 100.0, "cv": 0.5},
        {"StockCode": "B", "decision": "order_now", "days_until_stockout": 3.0,
         "weekly_revenue_at_risk": 500.0, "cv": 1.2},
        {"StockCode": "C", "decision": "order_now", "days_until_stockout": 6.0,
         "weekly_revenue_at_risk": 250.0, "cv": 0.2},
        {"StockCode": "D", "decision": "dont_order", "days_until_stockout": 40.0,
         "weekly_revenue_at_risk": 900.0, "cv": 0.1},
    ])


def test_all_plans_flag_the_same_products(recs):
    sets = {p: set(plans.plan_view(recs, p)["StockCode"]) for p in plans.PLAN_NAMES}
    assert all(s == {"A", "B", "C"} for s in sets.values())


def test_stockout_urgency_sorts_most_urgent_first(recs):
    assert plans.plan_view(recs, "stockout_urgency")["StockCode"].tolist() == ["B", "C", "A"]


def test_revenue_protection_sorts_highest_revenue_first(recs):
    # revenue at risk: B 500 > C 250 > A 100
    assert plans.plan_view(recs, "revenue_protection")["StockCode"].tolist() == ["B", "C", "A"]


def test_forecast_certainty_sorts_lowest_cv_first(recs):
    assert plans.plan_view(recs, "forecast_certainty")["StockCode"].tolist() == ["C", "A", "B"]


def test_unknown_plan_raises(recs):
    with pytest.raises(ValueError):
        plans.plan_view(recs, "vibes")


def test_flagged_by_plans_labels_top_products(recs):
    labels = plans.flagged_by_plans(recs, top_n=1)
    assert labels[recs["StockCode"] == "B"].iloc[0] == "Stockout urgency, Revenue protection"
    assert labels[recs["StockCode"] == "C"].iloc[0] == "Forecast certainty"
    assert labels[recs["StockCode"] == "D"].iloc[0] == ""


artifacts_missing = not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH))


@pytest.mark.skipif(artifacts_missing, reason="run data pipeline + train_model first")
def test_promotion_flips_at_least_one_real_product():
    """With the demo inventory, the 1.3x promotion uplift must flip at least
    one product from 'don't order' to 'order now' (spec requirement)."""
    from src.forecasting import forecast_inventory, load_model_bundle

    bundle = load_model_bundle()
    feats = pd.read_csv(FEATURES_PATH, dtype={"StockCode": str}, parse_dates=["week_start"])
    recent = feats[feats["week_start"] >= feats["week_start"].max() - pd.Timedelta(weeks=12)]
    inventory = recent.groupby("StockCode")["units_sold"].sum().nlargest(40).index.tolist()

    forecasts = forecast_inventory(bundle, feats, inventory)
    stock = plans.default_stock_levels(inventory, forecasts)

    flipped = []
    for buffer in ("lean", "safe"):
        base = plans.build_recommendations(forecasts, feats, stock, {}, buffer=buffer)
        promo = plans.build_recommendations(forecasts, feats, stock, {}, buffer=buffer,
                                            promotion=True)
        merged = base.merge(promo, on="StockCode", suffixes=("_base", "_promo"))
        flips = merged[(merged["decision_base"] == "dont_order")
                       & (merged["decision_promo"] == "order_now")]
        flipped.extend(flips["StockCode"].tolist())
    assert flipped, "promotion toggle flipped no product in the demo inventory"
