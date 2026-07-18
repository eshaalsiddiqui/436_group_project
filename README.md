# Retail Inventory Restocking IDSS

An intelligent decision support system for a small independent retail shop
owner making weekly reorder decisions. A Random Forest forecasts each
product's demand for the next 4 weeks (with uncertainty taken from the
variance across the forest's trees), classic reorder-point logic turns those
forecasts into "order now / don't order yet" recommendations with safety
stock sized by a lean/safe buffer strategy, and a Streamlit app presents the
results as three switchable reorder plans — by stockout urgency, by revenue
at risk, and by forecast certainty — so the owner can prioritize according to
what they care about that week. Built on the UCI Online Retail dataset.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the data and train the model

```bash
# 1. Clean, aggregate to weekly demand per SKU, build the feature table
#    (writes data/processed/weekly_features.csv and the category mapping)
python -m src.data_pipeline

# 2. Train the Random Forest, print evaluation, save models/demand_model.joblib
python models/train_model.py
```

Training holds out the last 8 weeks as a test set and prints the MAE in units
plus a simulated stockout rate (how often following the safe-buffer
recommendations would still have led to a stockout). The production model is
then retrained on all data and saved with its category encoder, so the app
never retrains. A pre-trained `models/demand_model.joblib` is included in the
repo; step 2 only needs re-running if you rebuild the data.

## Launch the app

```bash
streamlit run app.py
```

- **Sidebar** — inventory size (top 20–50 sellers), lean/safe buffer
  strategy (1 vs 2 standard deviations of safety stock), a promotion toggle
  (1.3x demand uplift), default supplier lead time, and editable per-product
  current stock / lead-time overrides.
- **Main panel** — 4-week demand forecast with uncertainty bars for a
  selected product; a stock depletion curve against the reorder point line;
  and the flagged-product table under three plan tabs. All plans share one
  cached forecast run — switching tabs or toggles only re-sorts and re-runs
  the cheap reorder math, never the model.

## Tests

```bash
python -m pytest tests/
```

Includes the proposal's worked example (stock 45, demand 20/week, 7-day lead:
lean says wait, safe says order) and a check that the promotion toggle flips
at least one real product from "don't order" to "order now".

## Notes on evaluation

Weekly SKU-level demand in this dataset is highly intermittent (median 3
units/week, max ~81k), so the model is trained on log1p(units) and its MAE
(~33 units/week) sits close to a strong rolling-average baseline — most of
the predictable signal lives in the recent-history features the spec
prescribes. The per-tree standard deviation measures *model* uncertainty
rather than full demand variability, which is why the simulated stockout
rate is highest for volatile high-volume products; this is a known limitation
of using ensemble variance for safety stock and a good place for future work
(e.g. quantile regression forests).
