import numpy as np
import pandas as pd
import pytest
from quantmind.core import demo_prices, features_and_labels, walk_forward, evaluation_table, forward_probability


def data_fixture():
    return features_and_labels(demo_prices(n=550, seed=17))


def test_demo_is_deterministic():
    assert demo_prices(n=100).equals(demo_prices(n=100))


def test_future_target_uses_next_close_to_following_close():
    prices = demo_prices(n=100)
    df = features_and_labels(prices)
    for i in (25, 55, 89):
        expected = prices["Close"].iloc[i+2] / prices["Close"].iloc[i+1] - 1
        assert df["future_return"].iloc[i] == pytest.approx(expected)
    assert pd.isna(df["target"].iloc[-1])
    assert pd.isna(df["target"].iloc[-2])


def test_rolling_features_do_not_use_future():
    p1 = demo_prices(n=250)
    p2 = p1.copy()
    p2.loc[p2.index[180]:, "Close"] *= 1.6
    p2.loc[p2.index[180]:, "High"] *= 1.6
    p2.loc[p2.index[180]:, "Low"] *= 1.6
    p2.loc[p2.index[180]:, "Open"] *= 1.6
    x1,x2=features_and_labels(p1),features_and_labels(p2)
    pd.testing.assert_frame_equal(x1.iloc[:180].drop(columns=["future_return", "target"]), x2.iloc[:180].drop(columns=["future_return", "target"]))

@pytest.mark.parametrize("model", ["Logistic Regression", "Random Forest"])
def test_walkforward_complete(model):
    result, folds = walk_forward(data_fixture(), model, n_splits=3)
    assert len(result) > 100
    assert len(folds) == 3
    for _, row in folds.iterrows():
        assert pd.Timestamp(row.train_end) < pd.Timestamp(row.test_start)
    assert result.index.is_unique
    assert result[["model_net", "momentum_net", "buyhold_net"]].notna().all().all()
    assert ((result.prob_up >= 0) & (result.prob_up <= 1)).all()
    metrics = evaluation_table(result)
    assert np.isfinite(metrics.loc["ML strategy", "total_return"])


def test_larger_cost_cannot_increase_strategy_wealth():
    data = data_fixture()
    a,_=walk_forward(data, "Logistic Regression", n_splits=3, cost_bps=0)
    b,_=walk_forward(data, "Logistic Regression", n_splits=3, cost_bps=40)
    assert (b["model_net"] <= a["model_net"] + 1e-12).all()
    assert b["model_net"].sum() <= a["model_net"].sum()


def test_latest_prediction_is_valid():
    date, prob = forward_probability(data_fixture(), "Logistic Regression")
    assert date == data_fixture().index[-1]
    assert 0 <= prob <= 1


def test_reject_short_history():
    with pytest.raises(ValueError, match="180"):
        walk_forward(features_and_labels(demo_prices(n=80)), "Logistic Regression")
