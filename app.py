"""Retail Inventory Restocking IDSS - Streamlit front end.

Run from the project root:

    streamlit run app.py

Loads the pre-trained model from models/demand_model.joblib (train it first
with `python models/train_model.py`). The forecast is computed once per
inventory and cached; buffer/promotion toggles and plan switching only re-run
the cheap reorder math and re-sorts - never model inference.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import FEATURES_PATH
from src.forecasting import forecast_inventory, load_model_bundle
from src.plans import (
    PLAN_NAMES,
    build_recommendations,
    default_stock_levels,
    flagged_by_plans,
    plan_view,
)
from src.reorder_logic import FORECAST_HORIZON_WEEKS

# Chart colors: validated reference palette (dataviz method) - blue series,
# status-critical threshold, muted ink and hairline grid.
C_SERIES = "#2a78d6"
C_CRITICAL = "#d03b3b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"

st.set_page_config(page_title="Retail Restocking IDSS", page_icon="📦", layout="wide")


# --------------------------------------------------------------------------
# Cached loading and the single forecast pass
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained model...")
def get_bundle():
    return load_model_bundle()


@st.cache_data(show_spinner="Loading weekly demand features...")
def get_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(FEATURES_PATH)
    return pd.read_csv(FEATURES_PATH, dtype={"StockCode": str}, parse_dates=["week_start"])


@st.cache_data(show_spinner="Forecasting demand for your inventory...")
def get_forecasts(inventory_size: int):
    """One model pass for the whole inventory: the top `inventory_size`
    products by recent 12-week volume. Every plan and toggle reuses this."""
    feats = get_features()
    recent = feats[feats["week_start"] >= feats["week_start"].max() - pd.Timedelta(weeks=12)]
    inventory = (recent.groupby("StockCode")["units_sold"].sum()
                 .nlargest(inventory_size).index.tolist())
    forecasts = forecast_inventory(get_bundle(), feats, inventory)
    return forecasts, inventory


def fail(msg: str):
    st.error(msg, icon="🚨")
    st.stop()


try:
    bundle = get_bundle()
except FileNotFoundError:
    fail("**No trained model found.** Run the two setup steps first:\n\n"
         "1. `python -m src.data_pipeline` (builds the weekly feature table)\n"
         "2. `python models/train_model.py` (trains and saves the model)\n\n"
         "Then restart the app.")

try:
    features = get_features()
except FileNotFoundError:
    fail("**Processed data not found.** Place the UCI *Online Retail* Excel file in "
         "`data/raw/` and run `python -m src.data_pipeline`, then restart the app.")


# --------------------------------------------------------------------------
# Sidebar: decision levers
# --------------------------------------------------------------------------

st.sidebar.title("📦 Restocking IDSS")

inventory_size = st.sidebar.slider(
    "Inventory size (top sellers)", min_value=20, max_value=50, value=40,
    help="How many of your best-selling products to manage here.")

buffer = st.sidebar.select_slider(
    "Buffer strategy", options=["lean", "safe"], value="safe",
    help="lean = 1 std dev of safety stock (less cash tied up), "
         "safe = 2 std devs (fewer stockouts).")

promotion = st.sidebar.toggle(
    "Promotion week (+30% demand)",
    help="Applies a 1.3x uplift to forecasted demand before all reorder math.")

default_lead_time = st.sidebar.number_input(
    "Default supplier lead time (days)", min_value=1, max_value=60, value=7)

forecasts, inventory = get_forecasts(inventory_size)
names = (features.sort_values("week_start").groupby("StockCode")["description"].last())

# Per-product stock and lead-time overrides, seeded with a demo heuristic.
editor_key = f"stock_editor_{inventory_size}"
if editor_key not in st.session_state:
    seed_stock = default_stock_levels(inventory, forecasts)
    st.session_state[editor_key] = pd.DataFrame({
        "StockCode": inventory,
        "product": [str(names.get(c, c))[:30] for c in inventory],
        "current_stock": [seed_stock[c] for c in inventory],
        "lead_time_days": [default_lead_time] * len(inventory),
    })

with st.sidebar.expander("Current stock & lead time per product"):
    st.caption("Demo stock counts are pre-filled - edit them to match your shelves.")
    edited = st.data_editor(
        st.session_state[editor_key],
        column_config={
            "StockCode": st.column_config.TextColumn(disabled=True),
            "product": st.column_config.TextColumn(disabled=True),
            "current_stock": st.column_config.NumberColumn(min_value=0, step=1),
            "lead_time_days": st.column_config.NumberColumn(min_value=1, max_value=60, step=1),
        },
        hide_index=True, key=f"editor_widget_{inventory_size}")
    st.session_state[editor_key] = edited

stock_levels = dict(zip(edited["StockCode"], edited["current_stock"]))
lead_times = dict(zip(edited["StockCode"], edited["lead_time_days"]))

# Cheap pure-math pass over the cached forecasts (no model inference).
recs = build_recommendations(
    forecasts, features[features["StockCode"].isin(inventory)],
    stock_levels, lead_times, default_lead_time_days=default_lead_time,
    buffer=buffer, promotion=promotion)
recs["flagged_by"] = flagged_by_plans(recs)

flagged = recs[recs["decision"] == "order_now"]
st.sidebar.divider()
st.sidebar.metric("Products to order now", f"{len(flagged)} of {len(recs)}")
st.sidebar.metric("Est. reorder spend",
                  f"£{(flagged['recommended_order_qty'] * flagged['unit_price']).sum():,.0f}")


# --------------------------------------------------------------------------
# Main panel, top: 4-week forecast with uncertainty for a selected product
# --------------------------------------------------------------------------

st.title("Weekly reorder decisions")
st.caption(f"Demand model trained through {pd.Timestamp(bundle['trained_through']).date()} · "
           f"held-out MAE {bundle['test_mae']:.1f} units/week · buffer: **{buffer}**"
           + (" · **promotion uplift on**" if promotion else ""))

selected = st.selectbox(
    "Product", options=inventory,
    format_func=lambda c: f"{c} - {str(names.get(c, ''))[:40]}")

sel_fc = forecasts[forecasts["StockCode"] == selected]
sel_rec = recs[recs["StockCode"] == selected].iloc[0]
uplift = 1.3 if promotion else 1.0

col1, col2 = st.columns(2)

with col1:
    st.subheader("4-week demand forecast")
    fig = go.Figure(go.Bar(
        x=[f"Week +{h}" for h in sel_fc["horizon"]],
        y=sel_fc["forecast"] * uplift,
        error_y=dict(type="data", array=sel_fc["forecast_std"], color=C_MUTED, thickness=1.5),
        marker_color=C_SERIES, width=0.55,
        hovertemplate="%{x}: %{y:.0f} units (±%{error_y.array:.0f})<extra></extra>"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="units", gridcolor=C_GRID, zerolinecolor=C_GRID,
                   tickfont=dict(color=C_MUTED), rangemode="nonnegative"),
        xaxis=dict(tickfont=dict(color=C_MUTED)))
    st.plotly_chart(fig)
    st.caption("Error bars: ±1 std dev across the forest's individual trees, clipped at 0 "
               "(demand can't go negative)" + (" · uplift applied" if promotion else ""))

# --------------------------------------------------------------------------
# Main panel, middle: stock depletion vs reorder point
# --------------------------------------------------------------------------

with col2:
    st.subheader("Stock runway vs reorder point")
    daily = sel_rec["forecast_weekly"] * uplift / 7.0
    days = np.arange(0, FORECAST_HORIZON_WEEKS * 7 + 1)
    projected = np.clip(sel_rec["current_stock"] - daily * days, 0, None)
    rop = sel_rec["reorder_point"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=days, y=projected, mode="lines", line=dict(color=C_SERIES, width=2),
        name="projected stock",
        hovertemplate="day %{x}: %{y:.0f} units<extra></extra>"))
    fig2.add_hline(y=rop, line=dict(color=C_CRITICAL, width=2, dash="dash"),
                   annotation_text=f"⚠ reorder point ({rop:.0f})",
                   annotation_font_color=C_CRITICAL)
    if daily > 0 and sel_rec["current_stock"] > rop:
        cross = (sel_rec["current_stock"] - rop) / daily
        fig2.add_trace(go.Scatter(
            x=[cross], y=[rop], mode="markers+text",
            marker=dict(color=C_CRITICAL, size=10),
            text=[f"  day {cross:.0f}"], textposition="middle right",
            textfont=dict(color=C_CRITICAL), name="crosses reorder point",
            hovertemplate=f"crosses reorder point on day {cross:.0f}<extra></extra>"))
    fig2.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="units on hand", gridcolor=C_GRID, zerolinecolor=C_GRID,
                   tickfont=dict(color=C_MUTED), rangemode="nonnegative"),
        xaxis=dict(title="days from today", gridcolor=C_GRID,
                   tickfont=dict(color=C_MUTED)))
    st.plotly_chart(fig2)

if sel_rec["decision"] == "order_now":
    st.warning(f"**Order now:** stock ({sel_rec['current_stock']:.0f}) is at/below the "
               f"reorder point ({sel_rec['reorder_point']:.0f}). Recommended order: "
               f"**{sel_rec['recommended_order_qty']:.0f} units** "
               f"(covers {FORECAST_HORIZON_WEEKS} weeks of forecasted demand).", icon="🛒")
else:
    st.success(f"**No order needed yet:** {sel_rec['days_until_reorder_point']:.0f} days of "
               f"runway before stock ({sel_rec['current_stock']:.0f}) reaches the reorder "
               f"point ({sel_rec['reorder_point']:.0f}).", icon="✅")


# --------------------------------------------------------------------------
# Main panel, bottom: urgency table + the three plans (pure re-sorts)
# --------------------------------------------------------------------------

st.subheader("Reorder plans")
st.caption("All plans share one forecast run - switching tabs only re-sorts the same table.")

BASE_COLS = {
    "StockCode": st.column_config.TextColumn("SKU"),
    "description": st.column_config.TextColumn("Product", width="medium"),
    "current_stock": st.column_config.NumberColumn("Stock", format="%d"),
    "reorder_point": st.column_config.NumberColumn("Reorder pt", format="%.0f"),
    "days_until_stockout": st.column_config.NumberColumn("Days to stockout", format="%.1f"),
    "recommended_order_qty": st.column_config.NumberColumn("Order qty", format="%d"),
    "weekly_revenue_at_risk": st.column_config.NumberColumn("Revenue at risk (weekly)", format="£%.0f"),
    "cv": st.column_config.NumberColumn("Demand CV", format="%.2f"),
    "urgency": st.column_config.TextColumn("Urgency"),
    "confidence": st.column_config.TextColumn("Confidence"),
    "flagged_by": st.column_config.TextColumn("Also top-10 in", width="medium"),
}

# Each plan leads with the column it actually sorts by, emphasized: urgency
# tier (color-coded relative to lead time) for stockout urgency, a revenue
# bar for revenue protection (the sort key is derived, not a raw column, so
# it's surfaced explicitly rather than left for the owner to infer), and a
# confidence tier for forecast certainty.
PLAN_COLUMNS = {
    "stockout_urgency": ["StockCode", "description", "urgency", "days_until_stockout",
                         "current_stock", "reorder_point", "recommended_order_qty", "flagged_by"],
    "revenue_protection": ["StockCode", "description", "weekly_revenue_at_risk",
                           "recommended_order_qty", "current_stock", "reorder_point",
                           "days_until_stockout", "flagged_by"],
    "forecast_certainty": ["StockCode", "description", "confidence", "cv",
                           "days_until_stockout", "current_stock", "reorder_point",
                           "recommended_order_qty", "flagged_by"],
}

tabs = st.tabs(list(PLAN_NAMES.values()))
for tab, plan_key in zip(tabs, PLAN_NAMES):
    with tab:
        view = plan_view(recs, plan_key)
        if view.empty:
            st.info("No products are flagged for reorder under the current settings.")
            continue
        cols = PLAN_COLUMNS[plan_key]
        col_config = dict(BASE_COLS)
        if plan_key == "revenue_protection":
            col_config["weekly_revenue_at_risk"] = st.column_config.ProgressColumn(
                "Revenue at risk (weekly)", format="£%.0f", min_value=0.0,
                max_value=float(view["weekly_revenue_at_risk"].max()))
        st.dataframe(view[cols], column_config=col_config, hide_index=True, width="stretch")

with st.expander("Products not flagged (runway remaining)"):
    watch = recs[recs["decision"] == "dont_order"].sort_values("days_until_reorder_point")
    st.dataframe(
        watch[["StockCode", "description", "current_stock", "reorder_point",
               "days_until_reorder_point", "days_until_stockout"]],
        column_config={
            "days_until_reorder_point": st.column_config.NumberColumn(
                "Days until reorder point", format="%.1f"),
            **{k: v for k, v in BASE_COLS.items() if k in
               ("StockCode", "description", "current_stock", "reorder_point",
                "days_until_stockout")},
        },
        hide_index=True, width="stretch")
