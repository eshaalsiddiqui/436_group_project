"""Tests for the forecasting layer's non-negativity guarantee.

With the current RandomForest + log1p(units) setup, forecasts already come
out >= 0 as an incidental consequence of tree-based averaging over
non-negative training targets (a leaf's prediction is the mean of the
training targets that land in it, so it can never fall outside
[min(train_y), max(train_y)], and log1p(units) >= 0). These tests use stub
estimators to prove the explicit floor in predict_with_uncertainty actually
does the clamping, independent of that incidental property - so the
guarantee holds even if the model or target encoding changes later.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import FEATURES_PATH
from src.forecasting import (
    MODEL_PATH,
    forecast_inventory,
    load_model_bundle,
    predict_with_uncertainty,
)


class _StubTree:
    def __init__(self, value):
        self._value = value

    def predict(self, X):
        return np.full(len(X), self._value)


class _StubForest:
    def __init__(self, values):
        self.estimators_ = [_StubTree(v) for v in values]


def test_negative_raw_predictions_are_floored_at_zero():
    model = _StubForest([-5.0, -1.0, 2.0, 4.0])
    X = pd.DataFrame({"x": [0, 0]})
    mean, std = predict_with_uncertainty(model, X, log_target=False)
    # trees clamp to [0, 0, 2, 4] before aggregating -> mean 1.5, not -0.5
    assert mean[0] == pytest.approx(1.5)
    assert (mean >= 0).all()


def test_negative_log_space_predictions_floor_after_expm1():
    # expm1(-5) ~= -0.9933, expm1(-0.1) ~= -0.0952: both negative, must floor to 0
    model = _StubForest([-5.0, -0.1, 0.0, 1.0])
    X = pd.DataFrame({"x": [0]})
    mean, std = predict_with_uncertainty(model, X, log_target=True)
    assert mean[0] >= 0
    assert mean[0] == pytest.approx((0 + 0 + 0 + (np.e - 1)) / 4)


def test_all_positive_predictions_are_unaffected():
    model = _StubForest([2.0, 4.0, 6.0])
    X = pd.DataFrame({"x": [0]})
    mean, std = predict_with_uncertainty(model, X, log_target=False)
    assert mean[0] == pytest.approx(4.0)
    assert std[0] == pytest.approx(np.std([2.0, 4.0, 6.0]))


artifacts_missing = not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH))


@pytest.mark.skipif(artifacts_missing, reason="run data pipeline + train_model first")
def test_real_forecast_inventory_never_predicts_negative_demand():
    bundle = load_model_bundle()
    feats = pd.read_csv(FEATURES_PATH, dtype={"StockCode": str}, parse_dates=["week_start"])
    recent = feats[feats["week_start"] >= feats["week_start"].max() - pd.Timedelta(weeks=12)]
    inventory = recent.groupby("StockCode")["units_sold"].sum().nlargest(15).index.tolist()

    forecasts = forecast_inventory(bundle, feats, inventory)
    assert (forecasts["forecast"] >= 0).all()
