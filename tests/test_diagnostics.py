import numpy as np
import pandas as pd
import pytest
from quantmind.core import demo_prices, features_and_labels
from quantmind.diagnostics import compare_models, prior_probability, sensitivity, calibration_bins

@pytest.fixture(scope="module")
def experiment():
    return features_and_labels(demo_prices(n=540, seed=19))


def test_prior_is_training_only(experiment):
    from sklearn.model_selection import TimeSeriesSplit
    from quantmind.core import FEATURES
    clean = experiment.dropna(subset=FEATURES + ["future_return", "target"])
    series = prior_probability(experiment, n_splits=3)
    for train_idx, test_idx in TimeSeriesSplit(n_splits=3, gap=2).split(clean):
        expected = clean.iloc[train_idx]["target"].mean()
        assert np.allclose(series.loc[clean.iloc[test_idx].index], expected)


def test_models_share_same_oos(experiment):
    table, frames, baselines = compare_models(experiment, n_splits=3)
    a, b = frames.values()
    assert a.index.equals(b.index)
    assert a["target"].equals(b["target"])
    assert len(table) == 2
    assert baselines.attrs["sample_count"] == len(a)
    assert baselines.loc[baselines.name == "Constant P(up)=50%", "brier"].iloc[0] == pytest.approx(.25)


def test_sensitivity_costs_and_zero_cost(experiment):
    _, frames, _ = compare_models(experiment, n_splits=3)
    model = frames["Random Forest"]
    sensitivity_table = sensitivity(model)
    assert sensitivity_table.loc[0, "total_return"] == pytest.approx((1+model["model_gross"]).prod()-1)
    assert sensitivity_table.loc[30, "total_return"] <= sensitivity_table.loc[0, "total_return"]


def test_calibration_bin_counts(experiment):
    _, frames, _ = compare_models(experiment, n_splits=3)
    test = frames["Logistic Regression"]
    calibration = calibration_bins(test, bins=5)
    assert int(calibration["count"].sum()) == len(test)
    assert calibration["observed_rate"].between(0,1).all()
