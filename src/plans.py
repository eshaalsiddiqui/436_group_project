"""The three reorder plans of the IDSS.

All plans start from the exact same recommendation table (the model runs
once); they differ only in filtering (products flagged "order now") and sort
order. Switching plans is therefore a pure re-sort, never new inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reorder_logic import recommend

PLAN_NAMES = {
    "stockout_urgency": "Stockout urgency",
    "revenue_protection": "Revenue protection",
    "forecast_certainty": "Forecast certainty",
}


def build_recommendations(forecasts: pd.DataFrame,
                          features: pd.DataFrame,
                          stock_levels: dict,
                          lead_times: dict,
                          default_lead_time_days: float = 7,
                          buffer: str = "safe",
                          z: float = None,
                          promotion: bool = False) -> pd.DataFrame:
    """Combine per-week forecasts into one decision row per product.

    `forecasts` is the output of forecasting.forecast_inventory (one row per
    product-week over the 4-week horizon); forecasted_weekly_demand and its
    std are the horizon averages. `features` supplies unit price, description
    and the demand CV used by the forecast-certainty plan.
    """
    agg = forecasts.groupby("StockCode").agg(
        forecast_weekly=("forecast", "mean"),
        forecast_std=("forecast_std", "mean"),
    ).reset_index()

    latest = (features.sort_values("week_start")
              .groupby("StockCode")
              .agg(description=("description", "last"),
                   category_name=("category_name", "last"),
                   unit_price=("unit_price", "last"),
                   cv=("coefficient_of_variation_weekly_demand_past_12_weeks", "last")))
    agg = agg.merge(latest, on="StockCode", how="left")

    rows = []
    for r in agg.itertuples(index=False):
        rec = recommend(
            current_stock=stock_levels[r.StockCode],
            forecasted_weekly_demand=r.forecast_weekly,
            forecast_std=r.forecast_std,
            lead_time_days=lead_times.get(r.StockCode, default_lead_time_days),
            buffer=buffer,
            z=z,
            promotion=promotion,
        )
        rows.append({
            "StockCode": r.StockCode,
            "description": r.description,
            "category": r.category_name,
            "current_stock": stock_levels[r.StockCode],
            "lead_time_days": lead_times.get(r.StockCode, default_lead_time_days),
            "forecast_weekly": r.forecast_weekly,
            "forecast_std": r.forecast_std,
            "unit_price": r.unit_price,
            "cv": r.cv,
            "reorder_point": rec["reorder_point"],
            "decision": rec["decision"],
            "recommended_order_qty": rec["recommended_order_qty"],
            "days_until_stockout": rec["days_until_stockout"],
            "days_until_reorder_point": rec["days_until_reorder_point"],
            "weekly_revenue_at_risk": rec["effective_weekly_demand"] * r.unit_price,
        })
    return pd.DataFrame(rows)


def plan_view(recommendations: pd.DataFrame, plan: str) -> pd.DataFrame:
    """Sort/filter the shared recommendation table for one plan.

    - stockout_urgency:   flagged products, most urgent (fewest days) first
    - revenue_protection: flagged products, highest weekly revenue at risk first
    - forecast_certainty: flagged products, most confident forecasts (lowest
                          demand CV) first, for cash-constrained ordering
    """
    flagged = recommendations[recommendations["decision"] == "order_now"]
    if plan == "stockout_urgency":
        return flagged.sort_values("days_until_stockout")
    if plan == "revenue_protection":
        return flagged.sort_values("weekly_revenue_at_risk", ascending=False)
    if plan == "forecast_certainty":
        return flagged.sort_values("cv")
    raise ValueError(f"Unknown plan {plan!r}; expected one of {list(PLAN_NAMES)}")


def flagged_by_plans(recommendations: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """For each flagged product, which plans rank it in their top `top_n`.

    Every plan flags the same product set (they only re-sort it), so this
    reports where each product lands near the top of each priority list.
    Returns a Series indexed like `recommendations`.
    """
    labels = pd.Series("", index=recommendations.index)
    for plan, nice in PLAN_NAMES.items():
        top = plan_view(recommendations, plan).head(top_n).index
        labels.loc[top] = labels.loc[top].where(labels.loc[top] == "", labels.loc[top] + ", ") + nice
    return labels


def default_stock_levels(stock_codes: list, forecasts: pd.DataFrame) -> dict:
    """Deterministic demo starting stock: 1.5-4.5 weeks of forecasted demand.

    The owner edits real stock counts in the app sidebar; this seeds a varied
    demo inventory (some products below their reorder point, some with
    runway) so every plan and toggle has visible effect.
    """
    weekly = forecasts.groupby("StockCode")["forecast"].mean()
    rng = np.random.RandomState(42)
    weeks_of_cover = rng.uniform(1.5, 4.5, size=len(stock_codes))
    return {
        code: max(1, int(round(weekly.get(code, 0) * w)))
        for code, w in zip(sorted(stock_codes), weeks_of_cover)
    }
