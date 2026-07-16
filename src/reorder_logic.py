"""Reorder decision logic for the Retail Inventory Restocking IDSS.

Implements the reorder point formula from the project proposal:

    reorder_point = forecast * (lead_time_days / 7)
                    + z * forecast_std * sqrt(lead_time_days / 7)

where z is 1 for the "lean" buffer strategy and 2 for "safe". A promotion
toggle applies a 1.3x uplift to forecasted demand before any calculation.
"""

from __future__ import annotations

import math
from typing import Optional

Z_BY_BUFFER = {"lean": 1.0, "safe": 2.0}
PROMOTION_UPLIFT = 1.3
FORECAST_HORIZON_WEEKS = 4


def z_for_buffer(buffer: str) -> float:
    """Map a buffer strategy name to its z value (std devs of safety stock)."""
    try:
        return Z_BY_BUFFER[buffer]
    except KeyError:
        raise ValueError(f"Unknown buffer strategy {buffer!r}; expected one of {list(Z_BY_BUFFER)}")


def reorder_point(forecasted_weekly_demand: float, forecast_std: float,
                  lead_time_days: float, z: float) -> float:
    """Reorder point: expected demand over the lead time plus z-sigma safety stock."""
    lead_time_weeks = lead_time_days / 7.0
    return (forecasted_weekly_demand * lead_time_weeks
            + z * forecast_std * math.sqrt(lead_time_weeks))


def recommend(current_stock: float,
              forecasted_weekly_demand: float,
              forecast_std: float,
              lead_time_days: float,
              buffer: str = "safe",
              z: Optional[float] = None,
              promotion: bool = False,
              on_order: float = 0.0) -> dict:
    """Produce the reorder recommendation for one product.

    If `z` is given it overrides the buffer-strategy mapping (used by the
    app's continuous slider). With `promotion=True`, forecasted demand is
    multiplied by 1.3 before every calculation.

    Returns a dict with the decision ("order_now" / "dont_order"), the
    reorder point, recommended order quantity (0 when not ordering),
    days_until_stockout, and days_until_reorder_point (None when already
    at/below the reorder point).
    """
    if z is None:
        z = z_for_buffer(buffer)
    demand = forecasted_weekly_demand * (PROMOTION_UPLIFT if promotion else 1.0)

    rop = reorder_point(demand, forecast_std, lead_time_days, z)
    daily_demand = demand / 7.0

    if daily_demand > 0:
        days_until_stockout = current_stock / daily_demand
    else:
        days_until_stockout = math.inf

    if current_stock <= rop:
        decision = "order_now"
        # Order up to 4 weeks of forecasted demand (the forecast horizon),
        # net of what we already have and what is already on order.
        order_qty = max(0.0, FORECAST_HORIZON_WEEKS * demand - current_stock - on_order)
        order_qty = math.ceil(order_qty)
        days_until_reorder_point = None
    else:
        decision = "dont_order"
        order_qty = 0
        days_until_reorder_point = (
            (current_stock - rop) / daily_demand if daily_demand > 0 else math.inf
        )

    return {
        "decision": decision,
        "reorder_point": rop,
        "recommended_order_qty": order_qty,
        "days_until_stockout": days_until_stockout,
        "days_until_reorder_point": days_until_reorder_point,
        "effective_weekly_demand": demand,
        "z": z,
    }
