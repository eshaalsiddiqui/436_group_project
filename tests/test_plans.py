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


class TestUrgencyTier:
    """Urgency is relative to lead time: will the order arrive in time?"""

    def test_critical_when_stockout_before_lead_time_elapses(self):
        assert plans.urgency_tier(days_until_stockout=5, lead_time_days=7) == "\U0001F534 Critical"

    def test_critical_at_exact_lead_time_boundary(self):
        assert plans.urgency_tier(days_until_stockout=7, lead_time_days=7) == "\U0001F534 Critical"

    def test_soon_when_within_two_lead_times(self):
        assert plans.urgency_tier(days_until_stockout=10, lead_time_days=7) == "\U0001F7E1 Soon"

    def test_monitor_when_comfortable_margin(self):
        assert plans.urgency_tier(days_until_stockout=20, lead_time_days=7) == "\U0001F7E2 Monitor"

    def test_shorter_lead_time_shrinks_the_critical_window(self):
        # 5-day runway is Critical against a 7-day lead time, but comfortably
        # past both the lead time and its 2x margin against a 2-day lead time
        assert plans.urgency_tier(days_until_stockout=5, lead_time_days=2) == "\U0001F7E2 Monitor"


class TestConfidenceTier:
    def test_high_confidence_below_threshold(self):
        assert plans.confidence_tier(0.5) == "\U0001F7E2 High confidence"

    def test_medium_confidence_mid_range(self):
        assert plans.confidence_tier(1.4) == "\U0001F7E1 Medium confidence"

    def test_low_confidence_above_threshold(self):
        assert plans.confidence_tier(2.5) == "\U0001F534 Low confidence"

    def test_boundaries_are_exclusive_upward(self):
        assert plans.confidence_tier(plans.CV_HIGH_CONFIDENCE_MAX) == "\U0001F7E1 Medium confidence"
        assert plans.confidence_tier(plans.CV_MEDIUM_CONFIDENCE_MAX) == "\U0001F534 Low confidence"


def test_build_recommendations_includes_urgency_and_confidence_tiers():
    forecasts = pd.DataFrame([
        {"StockCode": "A", "forecast": 20.0, "forecast_std": 5.0},
    ])
    features = pd.DataFrame([
        {"StockCode": "A", "week_start": "2011-01-03", "description": "WIDGET",
         "category_name": "WIDGETS", "unit_price": 2.5,
         "coefficient_of_variation_weekly_demand_past_12_weeks": 0.3},
    ])
    out = plans.build_recommendations(forecasts, features, {"A": 5}, {}, buffer="safe")
    assert out.loc[0, "confidence"] == "\U0001F7E2 High confidence"
    assert out.loc[0, "urgency"] in {"\U0001F534 Critical", "\U0001F7E1 Soon", "\U0001F7E2 Monitor"}


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
