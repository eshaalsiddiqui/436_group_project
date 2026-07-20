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
    FORECAST_HORIZON_WEEKS,
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


class TestOrderQuantityClearsReorderPoint:
    """When forecast uncertainty is high, the safety-stock term in the
    reorder point can exceed 4 weeks of raw demand coverage. The recommended
    order quantity must still bring stock to at least the reorder point -
    otherwise the app is telling the owner to order and remain under their
    own threshold, which is exactly what undermined trust in the demo.

    Numbers below reproduce product 23843 (PAPER CRAFT, LITTLE BIRDIE) from
    the live app: stock 489, reorder point 847, and a pre-fix recommendation
    of 212 units (489 + 212 = 701 < 847). demand/std are back-calculated so
    reorder_point() lands on exactly 847 and the pre-fix formula on exactly
    212, reproducing the reported bug precisely before checking the fix.
    """

    STOCK = 489
    DEMAND = 175.25
    STD = 335.875
    LEAD_TIME_DAYS = 7

    def test_reorder_point_matches_reported_value(self):
        rop = reorder_point(self.DEMAND, self.STD, self.LEAD_TIME_DAYS, z=z_for_buffer("safe"))
        assert rop == pytest.approx(847.0, abs=0.01)

    def test_old_four_week_only_formula_would_undershoot(self):
        # Documents the bug: 4 weeks of demand alone doesn't reach the
        # reorder point, which is exactly why (a) alone was insufficient.
        four_week_only = FORECAST_HORIZON_WEEKS * self.DEMAND - self.STOCK
        assert four_week_only == pytest.approx(212.0, abs=0.01)
        rop = reorder_point(self.DEMAND, self.STD, self.LEAD_TIME_DAYS, z=z_for_buffer("safe"))
        assert self.STOCK + four_week_only < rop

    def test_recommended_order_brings_stock_to_or_above_reorder_point(self):
        rec = recommend(self.STOCK, self.DEMAND, self.STD, self.LEAD_TIME_DAYS, buffer="safe")
        assert rec["decision"] == "order_now"
        assert rec["recommended_order_qty"] == 358
        assert self.STOCK + rec["recommended_order_qty"] >= rec["reorder_point"]

    @pytest.mark.parametrize("stock,demand,std,lead_time", [
        (489, 175.25, 335.875, 7),   # high-uncertainty case from the bug report
        (10, 20.0, 5.0, 7),          # low-uncertainty case, 4-week term dominates
        (5, 50.0, 2.0, 3),           # short lead time, still order_now
        (0, 1.0, 50.0, 14),          # near-zero demand, huge uncertainty
    ])
    def test_order_now_always_clears_reorder_point(self, stock, demand, std, lead_time):
        rec = recommend(stock, demand, std, lead_time, buffer="safe")
        if rec["decision"] == "order_now":
            assert stock + rec["recommended_order_qty"] >= rec["reorder_point"]


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
