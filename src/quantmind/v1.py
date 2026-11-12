"""QuantMind 1.0: prequential, time-aware research and reproducible audit.

At end of session t: compute features(t). An order based on that signal
is notionally filled at close(t+1), earns close(t+2)/close(t+1)-1.
A model refitted at t may only train on rows <= t-3 (2-row embargo),
so labels and price features for training are known by then.
Validation-only model selection is fixed *before* final-period evaluation.
This remains a simplified retrospective exercise, not an execution simulator.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from typing import Optional
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss

from .core import FEATURES, features_and_labels, model_for, performance
from .audit import _validate_prices, price_snapshot_csv

MODELS = ("Logistic Regression", "Random Forest", "Hist Gradient Boosting")


@dataclass(frozen=True)
class StudyConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    retrain_every: int = 60
    train_window: Optional[int] = 756  # approx. 3 trading years; None=expanding
    threshold: float = 0.55
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    min_train_rows: int = 252
    bootstrap_repeats: int = 300
    bootstrap_block: int = 20
    random_seed: int = 42

    def validate(self):
        if not (0.4 <= self.train_fraction <= 0.8):
            raise ValueError("train_fraction must be between 0.4 and 0.8")
        if not (0.1 <= self.validation_fraction <= 0.35):
            raise ValueError("validation_fraction must be between 0.1 and 0.35")
        if self.train_fraction + self.validation_fraction >= 0.9:
            raise ValueError("at least 10% of observations must be reserved for final evaluation")
        if not (5 <= self.retrain_every <= 252):
            raise ValueError("retrain_every must be 5..252")
        if self.train_window is not None and self.train_window < 80:
            raise ValueError("train_window must be >=80 rows or expanding")
        if not (0.50 <= self.threshold <= 0.90):
            raise ValueError("threshold must be between 0.50 and 0.90")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("costs must be non-negative")
        if self.min_train_rows < 80:
            raise ValueError("min_train_rows must be >=80")
        if not (0 <= self.bootstrap_repeats <= 3000):
            raise ValueError("bootstrap_repeats must be 0..3000")
        if not (2 <= self.bootstrap_block <= 120):
            raise ValueError("bootstrap_block must be 2..120")


def clean_research_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """A unified, fully-labeled frame. Latest unlabeled rows are excluded."""
    _validate_prices(prices)
    data = features_and_labels(prices)
    data["trend_20"] = data["close"].pct_change(20, fill_method=None)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES + ["trend_20", "target", "future_return"])
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("input dates must be strictly increasing")
    if len(data) < 350:
        raise ValueError("At least 350 labeled feature rows required")
    return data


def chronological_boundaries(data: pd.DataFrame, config: StudyConfig) -> tuple[int,int]:
    config.validate()
    n = len(data)
    val_start = int(n * config.train_fraction)
    test_start = int(n * (config.train_fraction + config.validation_fraction))
    if val_start - 2 < config.min_train_rows:
        raise ValueError("Not enough initial history for the requested training size")
    if test_start - val_start < 30 or n - test_start < 30:
        raise ValueError("Validation/test windows need at least 30 rows each")
    return val_start, test_start


def _new_model(name: str):
    if name == "Hist Gradient Boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=70, learning_rate=0.05,
                                              max_leaf_nodes=9, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=42)
    return model_for(name)


def walk_period(data: pd.DataFrame, name: str, begin: int, end: int,
                config: StudyConfig, phase: str) -> pd.DataFrame:
    """Produce a complete contiguous sequence of predictions and an audit trail.

    Each fit uses only training rows < i-2, never test label data from the future.
    A model may learn from *past* validation labels as time progresses.
    """
    if name not in MODELS:
        raise ValueError("unsupported model")
    if not (0 <= begin < end <= len(data)):
        raise ValueError("invalid segment")
    rows = []
    model = None
    last_train = None
    for i in range(begin, end):
        retrain = model is None or (i - begin) % config.retrain_every == 0
        if retrain:
            # Row j's target ends at close(j+2); conservatively embargo two rows.
            historical = data.iloc[:i-2]
            if config.train_window is not None:
                historical = historical.iloc[-config.train_window:]
            if len(historical) < config.min_train_rows:
                raise ValueError("Insufficient train history at prediction date")
            y = historical["target"].astype(int)
            if y.nunique() < 2:
                raise ValueError("Training sample has only one label class")
            model = clone(_new_model(name))
            model.fit(historical[FEATURES], y)
            last_train = historical.index[-1]
            prior_up = float(y.mean())
            n_train = len(historical)
        sample = data.iloc[[i]]
        prob = float(model.predict_proba(sample[FEATURES])[0, list(model.classes_).index(1)])
        assert last_train < data.index[i - 2], "training label not embargoed"
        rows.append({
            "date": data.index[i], "phase": phase, "model": name,
            "target": int(sample["target"].iloc[0]),
            "future_return": float(sample["future_return"].iloc[0]),
            "prob_up": prob, "prior_up": prior_up,
            "momentum_signal": int(float(sample["ret_5"].iloc[0]) > 0),
            "vol_20": float(sample["vol_20"].iloc[0]),
            "trend_20": float(sample["trend_20"].iloc[0]),
            "retrained": retrain, "train_end": last_train,
            "train_rows": n_train,
        })
    return pd.DataFrame(rows).set_index("date")


def apply_policy(predictions: pd.DataFrame, threshold: float, config: StudyConfig) -> pd.DataFrame:
    """Mark-to-market one-session returns; costs paid when target position changes.

    Fee/slippage are illustrative per-position-change deductions; no market impact,
    closing auction uncertainty, taxation, dividends or actual fills are modeled.
    """
    x = predictions.copy().sort_index()
    rate = (config.fee_bps + config.slippage_bps) / 10000
    x["position"] = (x["prob_up"] >= threshold).astype(int)
    x["buyhold_position"] = 1
    x["momentum_position"] = x["momentum_signal"].astype(int)
    x["cash_position"] = 0
    for base, signal in [("ml", "position"), ("momentum", "momentum_position"),
                         ("buyhold", "buyhold_position"), ("cash", "cash_position")]:
        x[f"{base}_turnover"] = x[signal].diff().abs().fillna(x[signal].abs())
        x[f"{base}_gross"] = x[signal] * x["future_return"]
        x[f"{base}_net"] = x[f"{base}_gross"] - rate * x[f"{base}_turnover"]
        x[f"{base}_wealth"] = (1 + x[f"{base}_net"]).cumprod()
    x["pred_up"] = (x["prob_up"] >= 0.5).astype(int)
    return x


def probability_metrics(x: pd.DataFrame) -> dict:
    y = x["target"].astype(int)
    return {
        "observations": int(len(x)),
        "accuracy": float(accuracy_score(y, x["pred_up"])),
        "balanced_accuracy": float(balanced_accuracy_score(y, x["pred_up"])),
        "brier": float(brier_score_loss(y, x["prob_up"])),
        "prior_brier": float(brier_score_loss(y, x["prior_up"])),
        "brier_skill": float(1 - brier_score_loss(y, x["prob_up"]) / brier_score_loss(y, x["prior_up"]))
                         if brier_score_loss(y, x["prior_up"]) > 0 else float("nan"),
        "up_frequency": float(y.mean()),
        "always_up_accuracy": float(y.mean()),
    }


def policy_scorecard(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in [("ml", "ML strategy"),("momentum", "5-day momentum"),
                       ("buyhold", "Buy & Hold"),("cash", "Cash")]:
        perf = performance(x[f"{key}_net"])
        perf.update({"strategy": label, "position_changes": int(x[f"{key}_turnover"].sum()),
                     "exposure": float(x["position" if key == "ml" else f"{key}_position"].mean())})
        rows.append(perf)
    return pd.DataFrame(rows).set_index("strategy")


def regimes(x: pd.DataFrame, volatility_cutoff: float) -> pd.DataFrame:
    r = x.copy()
    r["regime"] = np.where(r["vol_20"] > volatility_cutoff, "Higher trailing volatility", "Lower trailing volatility")
    r["trend"] = np.where(r["trend_20"] > 0, "Positive 20d trend", "Non-positive 20d trend")
    groups = []
    for by in ("regime", "trend"):
        for group, part in r.groupby(by):
            groups.append({"dimension": by,"segment": group, "observations": int(len(part)),
                           "accuracy": float((part["pred_up"] == part["target"]).mean()),
                           "brier": float(brier_score_loss(part["target"], part["prob_up"])),
                           "mean_net_daily_return": float(part["ml_net"].mean()),
                           "annualized_net_daily_mean": float(part["ml_net"].mean() * 252)})
    return pd.DataFrame(groups)


def block_bootstrap_mean_difference(x: pd.DataFrame, config: StudyConfig, against: str = "buyhold") -> dict:
    """Descriptive moving-block CI of annualized mean daily return differential.

    NOT a confirmatory significance test; especially if historical outcomes were inspected.
    """
    diff = (x["ml_net"] - x[f"{against}_net"]).to_numpy(float)
    n = len(diff)
    center = float(diff.mean() * 252)
    if config.bootstrap_repeats == 0:
        return {"observed": center, "ci_low": None, "ci_high": None, "samples": 0}
    rng = np.random.default_rng(config.random_seed)
    block = min(config.bootstrap_block, n)
    samples = []
    for _ in range(config.bootstrap_repeats):
        starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
        path = np.concatenate([diff[s:s+block] for s in starts])[:n]
        samples.append(float(path.mean() * 252))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"observed": center, "ci_low": float(low), "ci_high": float(high),
            "samples": int(config.bootstrap_repeats), "block": block, "unit": "annualized arithmetic return difference"}


def run_study(prices: pd.DataFrame, config: StudyConfig = StudyConfig(),
              candidates: tuple[str, ...] = MODELS, ticker: str = "SYNTHETIC",
              source: str = "synthetic") -> dict:
    """Validation Brier selects model once; test is only scored afterwards.

    Previous AAPL v0.3 dates have already been viewed, so even test dates from
    that archive are *exploratory* despite clean procedural separation here.
    """
    config.validate()
    if not candidates or len(set(candidates)) != len(candidates) or any(m not in MODELS for m in candidates):
        raise ValueError("choose 1..3 unique supported models")
    prices = _validate_prices(prices)
    data = clean_research_frame(prices)
    val_start, test_start = chronological_boundaries(data, config)
    val_predictions = {}
    ranking = []
    for m in candidates:
        pred = walk_period(data, m, val_start, test_start - 2, config, "validation")
        pred = apply_policy(pred, config.threshold, config)
        val_predictions[m] = pred
        met = probability_metrics(pred)
        ranking.append({"model": m, **met, "net_return": float(performance(pred["ml_net"])["total_return"])})
    validation = pd.DataFrame(ranking).sort_values(["brier", "model"]).reset_index(drop=True)
    winner = str(validation.iloc[0]["model"])
    # Only after selection, fit/refit through time and evaluate the held-out era.
    test_predictions = walk_period(data, winner, test_start, len(data), config, "test")
    test_predictions = apply_policy(test_predictions, config.threshold, config)
    test_metrics = probability_metrics(test_predictions)
    scores = policy_scorecard(test_predictions)
    # Train-only distribution for grouping the final period; not a look-ahead percentile.
    vol_cutoff = float(data.iloc[:test_start]["vol_20"].median())
    segments = regimes(test_predictions, vol_cutoff)
    uncertainty = block_bootstrap_mean_difference(test_predictions, config)
    return {
        "ticker": ticker.upper(), "source": source, "config": asdict(config),
        "validation": validation, "validation_predictions": val_predictions,
        "champion": winner, "test": test_predictions, "test_metrics": test_metrics,
        "scorecard": scores, "regimes": segments, "bootstrap": uncertainty,
        "price_sha256": sha256(price_snapshot_csv(prices)).hexdigest(),
        "prices": prices, "feature_rows": len(data),
        "split": {"train_end": str(data.index[val_start-1].date()),
                  "validation_start": str(data.index[val_start].date()),
                  "validation_end": str(data.index[test_start-3].date()),
                  "validation_to_test_embargo_rows": 2,
                  "test_start": str(data.index[test_start].date()),
                  "test_end": str(data.index[-1].date()),
                  "n_train": val_start, "n_validation": test_start-val_start-2,
                  "n_test": len(data)-test_start},
    }


def manifest_from_study(study: dict) -> dict:
    import importlib.metadata
    return {
        "app": "QuantMind AI", "version": "1.0.0", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": study["ticker"], "source": study["source"],
        "price_snapshot_sha256": study["price_sha256"],
        "selection_rule": "Lowest validation-period Brier score; no test-period tuning",
        "selected_model": study["champion"], "config": study["config"],
        "split": study["split"],
        "environment": {"python":platform.python_version(), "numpy":np.__version__,
                        "pandas":pd.__version__, "scikit_learn":sklearn.__version__},
        "verification_scope": "Research pipeline; not a verified live/executable trading system",
        "study_status": "exploratory, historical dates may already have been inspected",
        "limitations": ["Adjusted OHLCV may be unavailable at executable auction prices",
                        "Fixed fee/slippage assumption, not market microstructure or taxes",
                        "Past AAPL historical dates were inspected in earlier versions",
                        "Single-asset signals; returns across currencies/venues not directly combinable",
                        "Validation-based candidate choice can still overfit to historical regimes",
                        "Moving-block CI descriptive, not alpha/significance evidence",
                        "Yahoo Finance personal use and redistribution terms may apply"],
    }


def study_bundle(study: dict, include_prices: bool = True) -> bytes:
    """Export locally for audit. Never commit raw licensed prices to the repository."""
    payload = BytesIO()
    m = manifest_from_study(study)
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(m, ensure_ascii=False, indent=2, default=str))
        z.writestr("validation_model_ranking.csv", study["validation"].to_csv(index=False))
        z.writestr("final_test_predictions.csv", study["test"].to_csv(index=True))
        z.writestr("final_test_policy_scorecard.csv", study["scorecard"].to_csv(index=True))
        z.writestr("test_regime_audit.csv", study["regimes"].to_csv(index=False))
        z.writestr("bootstrap_summary.json", json.dumps(study["bootstrap"], indent=2))
        for mname, p in study["validation_predictions"].items():
            slug = mname.lower().replace(" ", "_")
            z.writestr(f"validation_{slug}.csv", p.to_csv(index=True))
        if include_prices:
            z.writestr("prices_snapshot.csv", price_snapshot_csv(study["prices"]))
        z.writestr("REPRODUCE.md", (
            "# QuantMind AI v1.0 research bundle\n\n"
            "Keep this private if provider terms restrict data redistribution.\n"
            "Use the exported prices_snapshot.csv on the v1.0 dashboard (CSV snapshot mode).\n"
            "Set controls per manifest.json and rerun. For exact model predictions, pin environment and git commit.\n"
            "Validation chooses the champion; only then is the final period evaluated.\n"
            "These are hypothetical historical returns, not executable trades or advice.\n"
        ))
    return payload.getvalue()


def json_safe(value):
    """Convert nonfinite numeric metrics to JSON null for strict clients."""
    import math
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
