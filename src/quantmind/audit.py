"""Reproducibility and exploratory diagnostics for historical ML experiments.

This module does NOT optimize strategies, calibrate on test labels, or claim
that previously inspected historical dates constitute a pristine holdout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import zipfile
import platform
import sklearn

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from .core import FEATURES, performance
from .diagnostics import MODELS, prior_probability

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
VERSION = "0.3.0"


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price index must be dates")
    if prices.index.hasnans:
        raise ValueError("Price dates cannot be missing")
    missing = set(COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    p = prices[COLUMNS].copy()
    if p.index.tz is not None:
        p.index = p.index.tz_localize(None)
    p.index = p.index.normalize()
    if p.index.has_duplicates:
        raise ValueError("Duplicate dates are not allowed")
    p = p.sort_index()
    for c in COLUMNS:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    if not np.isfinite(p.to_numpy(dtype=float)).all():
        raise ValueError("Prices and volumes must be finite")
    if (p[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (p["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if ((p["High"] + 1e-9 < p[["Open", "Close", "Low"]].max(axis=1)) |
        (p["Low"] - 1e-9 > p[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("OHLC consistency check failed")
    if len(p) < 180:
        raise ValueError("At least 180 daily price rows required")
    p.index.name = "Date"
    return p


def price_snapshot_csv(prices: pd.DataFrame) -> bytes:
    """Canonical input-only price snapshot for repeatable research."""
    p = _validate_prices(prices)
    return p.to_csv(index_label="Date", float_format="%.17g", date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")


def load_price_snapshot(payload: bytes) -> pd.DataFrame:
    """Load a UTF-8 OHLCV CSV; no arbitrary Python/pickle execution."""
    if len(payload) > 30_000_000:
        raise ValueError("CSV larger than 30 MB")
    try:
        frame = pd.read_csv(BytesIO(payload), parse_dates=["Date"], float_precision="round_trip")
    except Exception as exc:
        raise ValueError("Unable to read CSV. Required: Date,Open,High,Low,Close,Volume") from exc
    if "Date" not in frame or frame["Date"].isna().any():
        raise ValueError("Invalid Date column")
    if not pd.api.types.is_datetime64_any_dtype(frame["Date"]):
        raise ValueError("Date column must contain valid dates")
    return _validate_prices(frame.set_index("Date"))


def price_fingerprint(prices: pd.DataFrame) -> str:
    return sha256(price_snapshot_csv(prices)).hexdigest()


def fold_audit(data: pd.DataFrame, frames: dict[str, pd.DataFrame], n_splits: int = 5) -> pd.DataFrame:
    """Compute honest descriptive fold metrics against a training-only base rate."""
    prior = prior_probability(data, n_splits=n_splits)
    clean = data.dropna(subset=FEATURES + ["future_return", "target"])
    rows = []
    expected = list(TimeSeriesSplit(n_splits=n_splits, gap=2).split(clean))
    for name in MODELS:
        pred = frames[name]
        if not pred.index.equals(prior.index):
            raise ValueError("Fold audit dates differ from baseline")
        for fold_no, (train_idx, test_idx) in enumerate(expected, 1):
            test_dates = clean.index[test_idx]
            block = pred.loc[test_dates]
            y = block["target"].astype(int)
            block_prior = prior.loc[test_dates]
            br = float(brier_score_loss(y, block["prob_up"]))
            prior_br = float(brier_score_loss(y, block_prior))
            rows.append({
                "model": name,
                "fold": fold_no,
                "train_end": clean.index[train_idx[-1]].date().isoformat(),
                "test_start": test_dates[0].date().isoformat(),
                "test_end": test_dates[-1].date().isoformat(),
                "observations": len(block),
                "actual_up_rate": float(y.mean()),
                "model_accuracy": float(accuracy_score(y, block["pred_up"])),
                "always_up_accuracy": float(y.mean()),
                "model_brier": br,
                "past_rate_brier": prior_br,
                "brier_skill": float(1 - br / prior_br) if prior_br > 0 else np.nan,
                "gross_return": performance(block["model_gross"])["total_return"],
                "net_return": performance(block["model_net"])["total_return"],
                "turnover": float(block["model_turnover"].sum()),
                "exposure": float(block["position"].mean()),
            })
    return pd.DataFrame(rows)


def fee_decomposition(frames: dict[str, pd.DataFrame], cost_bps: float) -> pd.DataFrame:
    """Cost effect in compounded wealth, not additive percentage-point fees."""
    rows = []
    for name, frame in frames.items():
        gross = performance(frame["model_gross"])
        net = performance(frame["model_net"])
        count = float(frame["model_turnover"].sum())
        rows.append({
            "model": name,
            "cost_bps": float(cost_bps),
            "position_changes": int(count),
            "gross_total_return": gross["total_return"],
            "net_total_return": net["total_return"],
            "cumulative_wealth_impact": (1 + gross["total_return"]) - (1 + net["total_return"]),
            "sum_of_daily_cost_rates": count * float(cost_bps) / 10000,
            "gross_sharpe": gross["sharpe"],
            "net_sharpe": net["sharpe"],
        })
    return pd.DataFrame(rows).set_index("model")


def experiment_archive(prices: pd.DataFrame, frames: dict[str, pd.DataFrame],
                       scorecard: pd.DataFrame, baselines: pd.DataFrame,
                       fold_table: pd.DataFrame, fees: pd.DataFrame, *,
                       source: str, ticker: str, start_date: str, n_splits: int,
                       threshold: float, cost_bps: float) -> bytes:
    snapshot = price_snapshot_csv(prices)
    p = _validate_prices(prices)
    manifest = {
        "app": "QuantMind AI", "version": VERSION, "study_status": "exploratory_only",
        "warning": "These past dates have already been inspected; they are NOT an untouched confirmatory holdout.",
        "source": source, "ticker": ticker, "requested_start_date": start_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_first_date": p.index[0].date().isoformat(),
        "data_last_date": p.index[-1].date().isoformat(),
        "price_rows": len(p), "snapshot_sha256": sha256(snapshot).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "target_definition": "close(t+2) / close(t+1) - 1, using auto-adjusted daily OHLCV",
        "information_available_at": "close(t)", "entry_time": "close(t+1)",
        "exit_time": "close(t+2)", "time_split": "TimeSeriesSplit expanding window, gap=2",
        "n_splits": int(n_splits), "threshold": float(threshold), "fee_per_position_change_bps": float(cost_bps),
        "models": {"Logistic Regression": "StandardScaler + LogisticRegression(max_iter=1500,random_state=42)",
                   "Random Forest": "RandomForestClassifier(n_estimators=120,min_samples_leaf=12,max_depth=6,random_state=42)"},
        "limitations": ["single-asset retrospective exploration", "no slippage modeling beyond fee assumption",
                        "adjusted OHLCV fills may not be attainable", "past test dates are no longer untouched",
                        "no live paper-trading validation", "no corporate-event point-in-time database"],
    }
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("prices_snapshot.csv", snapshot)
        z.writestr("model_scorecard.csv", scorecard.to_csv())
        z.writestr("prediction_baselines.csv", baselines.to_csv(index=False))
        z.writestr("per_fold_audit.csv", fold_table.to_csv(index=False))
        z.writestr("fee_decomposition.csv", fees.to_csv())
        for name, frame in frames.items():
            safe = name.lower().replace(" ", "_")
            z.writestr(f"predictions_{safe}.csv", frame.to_csv(index_label="Date"))
        z.writestr("HOW_TO_REPRODUCE.txt", "Use prices_snapshot.csv as the input source in Research Audit v0.3.\n"
                   "Match folds, threshold, and fee from manifest.json.\n"
                   "Hash checks only verify raw price snapshot identity; all predictions are exploratory.\n"
                   "Offline replay: PYTHONPATH=src python scripts/reproduce.py path/to/this_bundle.zip\n")
    return out.getvalue()
