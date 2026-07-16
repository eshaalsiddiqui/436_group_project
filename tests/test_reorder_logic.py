"""Tests for the reorder decision logic, anchored to the proposal's worked
example: current stock 45, forecasted demand 20 units/week, lead time 7 days.

The proposal quotes reorder points of ~28 (lean) and ~36 (safe), and decisions
of "don't order yet" (lean) vs "order now" (safe). With the proposal's own
formula those two claims need different forecast_std values, so each is
back-calculated and tested separately:

- forecast_std = 8   reproduces the quoted reorder points exactly
  (20 + 1*8 = 28, 20 + 2*8 = 36).
- forecast_std = 12.5 reproduces the decision flip at stock 45
  (lean ROP 32.5 -> don't order; safe ROP 45 -> order now). With std = 8 both
  buffers would say "don't order" at stock 45 (the flip would happen at 30).
"""

import math

import pytest

from src.reorder_logic import (
    PROMOTION_UPLIFT,
    recommend,
    reorder_point,
    z_for_buffer,
)

STOCK = 45
DEMAND = 20.0
LEAD_TIME_DAYS = 7


class TestWorkedExampleReorderPoints:
    """forecast_std = 8 reproduces the proposal's quoted reorder points."""

    STD = 8.0

    def test_lean_reorder_point_is_28(self):
        rop = reorder_point(DEMAND, self.STD, LEAD_TIME_DAYS, z=z_for_buffer("lean"))
        assert rop == pytest.approx(28.0)

    def test_safe_reorder_point_is_36(self):
        rop = reorder_point(DEMAND, self.STD, LEAD_TIME_DAYS, z=z_for_buffer("safe"))
        assert rop == pytest.approx(36.0)

    def test_lean_decision_is_dont_order(self):
        rec = recommend(STOCK, DEMAND, self.STD, LEAD_TIME_DAYS, buffer="lean")
        assert rec["decision"] == "dont_order"
        # Runway before hitting the reorder point: (45 - 28) / (20/7) days.
        assert rec["days_until_reorder_point"] == pytest.approx(5.95)


class TestWorkedExampleDecisionFlip:
    """forecast_std = 12.5 reproduces the proposal's lean/safe decision flip."""

    STD = 12.5

    def test_lean_says_dont_order_yet(self):
        rec = recommend(STOCK, DEMAND, self.STD, LEAD_TIME_DAYS, buffer="lean")
        assert rec["decision"] == "dont_order"
        assert rec["recommended_order_qty"] == 0
        assert rec["days_until_reorder_point"] > 0

    def test_safe_says_order_now(self):
        rec = recommend(STOCK, DEMAND, self.STD, LEAD_TIME_DAYS, buffer="safe")
        assert rec["decision"] == "order_now"
        # Order up to 4 weeks of demand: 4*20 - 45 = 35 units.
        assert rec["recommended_order_qty"] == 35

    def test_days_until_stockout(self):
        rec = recommend(STOCK, DEMAND, self.STD, LEAD_TIME_DAYS, buffer="safe")
        assert rec["days_until_stockout"] == pytest.approx(45 / (20 / 7))


class TestPromotionToggle:
    def test_promotion_multiplies_demand_by_1_3(self):
        base = recommend(100, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="lean")
        promo = recommend(100, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="lean", promotion=True)
        assert promo["effective_weekly_demand"] == pytest.approx(DEMAND * PROMOTION_UPLIFT)
        assert promo["reorder_point"] > base["reorder_point"]

    def test_promotion_flips_dont_order_to_order_now(self):
        # stock 30, demand 20, std 5, lean: ROP = 25 -> don't order;
        # with promotion demand becomes 26: ROP = 31 > 30 -> order now.
        base = recommend(30, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="lean")
        promo = recommend(30, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="lean", promotion=True)
        assert base["decision"] == "dont_order"
        assert promo["decision"] == "order_now"
        # Order up to 4 weeks of uplifted demand: 4*26 - 30 = 74 units.
        assert promo["recommended_order_qty"] == 74


class TestEdgeCases:
    def test_on_order_reduces_recommended_quantity(self):
        with_on_order = recommend(10, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="safe", on_order=20)
        without = recommend(10, DEMAND, 5.0, LEAD_TIME_DAYS, buffer="safe")
        assert with_on_order["recommended_order_qty"] == without["recommended_order_qty"] - 20

    def test_zero_demand_never_stocks_out(self):
        rec = recommend(10, 0.0, 0.0, LEAD_TIME_DAYS, buffer="safe")
        assert rec["days_until_stockout"] == math.inf
        assert rec["decision"] == "dont_order"

    def test_continuous_z_overrides_buffer(self):
        rec = recommend(STOCK, DEMAND, 8.0, LEAD_TIME_DAYS, z=1.5)
        assert rec["reorder_point"] == pytest.approx(32.0)

    def test_unknown_buffer_raises(self):
        with pytest.raises(ValueError):
            z_for_buffer("reckless")

    def test_longer_lead_time_scales_reorder_point(self):
        # 14 days: 20*2 + 2*8*sqrt(2) = 62.63
        rop = reorder_point(DEMAND, 8.0, 14, z=2.0)
        assert rop == pytest.approx(40 + 16 * math.sqrt(2))
