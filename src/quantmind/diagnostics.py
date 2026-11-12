"""Exploratory, fold-aligned diagnostics for QuantMind.

All baseline forecasts and model fits are made from earlier labeled rows.
Comparing models on the same OOS segment is exploratory, not confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from .core import FEATURES, performance, walk_forward

MODELS = ("Logistic Regression", "Random Forest")


def prior_probability(data: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """Predict the class frequency known from the corresponding training fold.

    This is a pre-test, expanding-window baseline, not the test-set prevalence.
    """
    clean = data.dropna(subset=FEATURES + ["future_return", "target"])
    baseline = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits, gap=2).split(clean):
        train, test = clean.iloc[train_idx], clean.iloc[test_idx]
        past_fraction = float(train["target"].mean())
        baseline.append(pd.Series(past_fraction, index=test.index, dtype=float))
    return pd.concat(baseline).sort_index().rename("prior_up")


def compare_models(data: pd.DataFrame, n_splits: int = 5, threshold: float = 0.55,
                   cost_bps: float = 10.0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Return same-fold comparison, raw prediction frames and summary baselines."""
    prior = prior_probability(data, n_splits)
    results: dict[str, pd.DataFrame] = {}
    rows = []
    for model in MODELS:
        pred, folds = walk_forward(data, model, n_splits, threshold, cost_bps)
        if not pred.index.equals(prior.index):
            raise ValueError("Model and prior baseline do not share identical test dates")
        if not pred["target"].equals(data.loc[pred.index, "target"]):
            raise ValueError("Test labels changed across comparison")
        results[model] = pred
        actual = pred["target"].astype(int)
        net = performance(pred["model_net"])
        gross = performance(pred["model_gross"])
        model_brier = float(brier_score_loss(actual, pred["prob_up"]))
        prior_brier = float(brier_score_loss(actual, prior))
        rows.append({
            "model": model,
            "direction_accuracy": float(accuracy_score(actual, pred["pred_up"])),
            "balanced_accuracy": float(balanced_accuracy_score(actual, pred["pred_up"])),
            "brier": model_brier,
            "brier_vs_prior": 1 - model_brier / prior_brier if prior_brier else np.nan,
            "total_return_net": net["total_return"],
            "total_return_gross": gross["total_return"],
            "annualized_return_net": net["annualized_return"],
            "sharpe_net": net["sharpe"],
            "max_drawdown_net": net["max_drawdown"],
            "turnover": int(pred["model_turnover"].sum()),
            "exposure": float(pred["position"].mean()),
        })
    reference = results[MODELS[0]]
    target = reference["target"].astype(int)
    baselines = pd.DataFrame([{
        "name": "Always predict up",
        "accuracy": float(target.mean()),
        "brier": 1 - float(target.mean()),  # all predictions exactly 1.0
    }, {
        "name": "Constant P(up)=50%",
        "accuracy": float(target.mean()),  # P >= 0.5 classified as up
        "brier": 0.25,
    }, {
        "name": "Past-fold positive rate",
        "accuracy": float(accuracy_score(target, (prior >= 0.5).astype(int))),
        "brier": float(brier_score_loss(target, prior)),
    }])
    baselines.attrs["sample_count"] = len(reference)
    baselines.attrs["positive_rate"] = float(target.mean())
    return pd.DataFrame(rows).set_index("model"), results, baselines


def sensitivity(result: pd.DataFrame, costs_bps: tuple[int, ...] = (0, 5, 10, 20, 30)) -> pd.DataFrame:
    """Reprice frozen decisions at alternative fee assumptions; never refit models."""
    rows = []
    for bps in costs_bps:
        net = result["model_gross"] - result["model_turnover"] * (bps / 10_000)
        stats = performance(net)
        rows.append({
            "cost_bps": bps,
            "total_return": stats["total_return"],
            "annualized_return": stats["annualized_return"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def calibration_bins(result: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """OOS reliability diagram data (diagnostic only; no calibration fitting)."""
    pred = result[["prob_up", "target"]].copy()
    pred["bin"] = pd.cut(pred["prob_up"], np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = pred.groupby("bin", observed=False).agg(
        count=("target", "size"), mean_predicted=("prob_up", "mean"), observed_rate=("target", "mean")
    )
    return grouped[grouped["count"] > 0].reset_index().assign(bin=lambda x: x["bin"].astype(str))
