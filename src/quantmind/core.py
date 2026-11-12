"""Leakage-aware, close-to-close research pipeline.

Features at closing time t; target: close(t+2)/close(t+1)-1.
Position decided after close(t), entered at close(t+1), exited at close(t+2).
An embargo of two rows prevents training labels overlapping test-time prices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap_5", "ma_gap_20", "range_1", "volume_change"]


def demo_prices(n: int = 1400, seed: int = 42) -> pd.DataFrame:
    """Synthetic data, explicitly NOT real financial history or evidence of alpha."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    regimes = np.where(np.arange(n) % 310 < 170, 0.0004, -0.0001)
    innovations = rng.normal(0, 0.014, n)
    rets = regimes + innovations
    close = 100 * np.exp(np.cumsum(rets))
    opn = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(close, opn) * (1 + rng.uniform(0, 0.009, n))
    low = np.minimum(close, opn) * (1 - rng.uniform(0, 0.009, n))
    volume = rng.integers(1_000_000, 6_000_000, n)
    return pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def market_prices(ticker: str, start: str) -> pd.DataFrame:
    """Fetch adjusted daily prices; caller handles network and provider restrictions."""
    import yfinance as yf
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError("No price data returned. Check symbol, date and network.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()


def features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy().sort_index()
    if not p.index.is_unique or not p.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if (p[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    close = p["Close"].astype(float)
    r = close.pct_change(fill_method=None)
    x = pd.DataFrame(index=p.index)
    x["ret_1"] = r
    x["ret_5"] = close.pct_change(5, fill_method=None)
    x["ret_10"] = close.pct_change(10, fill_method=None)
    x["vol_10"] = r.rolling(10).std()
    x["vol_20"] = r.rolling(20).std()
    x["ma_gap_5"] = close / close.rolling(5).mean() - 1
    x["ma_gap_20"] = close / close.rolling(20).mean() - 1
    x["range_1"] = (p["High"] - p["Low"]) / close
    x["volume_change"] = p["Volume"].astype(float).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    # Execute at t+1 close. Hold until t+2 close. Features available at t close.
    x["future_return"] = close.shift(-2) / close.shift(-1) - 1
    x["target"] = np.where(x["future_return"].notna(), (x["future_return"] > 0).astype(float), np.nan)
    x["close"] = close
    return x.replace([np.inf, -np.inf], np.nan)


def model_for(name: str):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=42))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=120, min_samples_leaf=12, max_depth=6, random_state=42, n_jobs=1)
    raise ValueError(f"Unknown model: {name}")


def walk_forward(data: pd.DataFrame, model_name: str, n_splits: int = 5, threshold: float = .55, cost_bps: float = 10.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window OOS predictions; each test fold uses one frozen past-trained model.

    Adjacent folds have no retrain on their own evaluation period. Training row i
    requires close(i+2) to be known, so last training row <= first test row - 3.
    """
    if not (0.5 <= threshold <= 0.95):
        raise ValueError("threshold must be in [0.5, 0.95]")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")
    clean = data.dropna(subset=FEATURES + ["future_return", "target"]).copy()
    if len(clean) < max(180, (n_splits + 1) * 25):
        raise ValueError("Need at least 180 cleaned observations; choose an earlier start date")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=2)
    out = []
    fold_info = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(clean), 1):
        train, test = clean.iloc[train_idx], clean.iloc[test_idx]
        assert train_idx[-1] + 2 < test_idx[0], "label horizon leaked into test fold"
        assert train.index.max() < test.index.min()
        if len(np.unique(train["target"])) < 2:
            raise ValueError("Training fold has only one target class")
        model = clone(model_for(model_name))
        model.fit(train[FEATURES], train["target"].astype(int))
        p_up = model.predict_proba(test[FEATURES])[:, list(model.classes_).index(1)]
        frame = test[["future_return", "target", "close", "ret_5"]].copy()
        frame["prob_up"] = p_up
        frame["pred_up"] = (p_up >= .5).astype(int)
        frame["position"] = (p_up >= threshold).astype(int)
        frame["momentum_position"] = (frame["ret_5"] > 0).astype(int)
        frame["fold"] = fold
        out.append(frame)
        fold_info.append({"fold": fold, "train_start": train.index.min().date().isoformat(), "train_end": train.index.max().date().isoformat(), "test_start": test.index.min().date().isoformat(), "test_end": test.index.max().date().isoformat(), "train_n": len(train), "test_n": len(test)})
    result = pd.concat(out).sort_index()
    # All folds are consecutive in the concatenated OOS region. Count transaction
    # cost on every entry / exit, including crossing a fold boundary.
    cost = cost_bps / 10_000
    for name, col in [("model", "position"), ("momentum", "momentum_position")]:
        trades = result[col].diff().abs().fillna(result[col].abs())
        result[f"{name}_gross"] = result[col] * result["future_return"]
        result[f"{name}_net"] = result[f"{name}_gross"] - trades * cost
        result[f"{name}_turnover"] = trades
    result["buyhold_net"] = result["future_return"] - (pd.Series(0.0, index=result.index).mask(np.arange(len(result)) == 0, cost))
    result["buyhold_gross"] = result["future_return"]
    return result, pd.DataFrame(fold_info)


def performance(returns: pd.Series) -> dict:
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("No strategy returns")
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    volatility = returns.std(ddof=1)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annualized_return": float(equity.iloc[-1] ** (252 / len(returns)) - 1) if equity.iloc[-1] > 0 else float("nan"),
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "volatility": float(volatility * np.sqrt(252)),
    }


def evaluation_table(result: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
    rows = []
    for label, col in [("ML strategy", "model_net"), ("5-day momentum", "momentum_net"), ("Buy & Hold", "buyhold_net")]:
        metric = performance(result[col])
        metric["strategy"] = label
        metric["turnover"] = float(result["model_turnover" if label == "ML strategy" else "momentum_turnover"].sum()) if label != "Buy & Hold" else 1.0
        rows.append(metric)
    metrics = pd.DataFrame(rows).set_index("strategy")
    metrics.attrs["direction_accuracy"] = float(accuracy_score(result["target"].astype(int), result["pred_up"]))
    metrics.attrs["balanced_accuracy"] = float(balanced_accuracy_score(result["target"].astype(int), result["pred_up"]))
    metrics.attrs["brier"] = float(brier_score_loss(result["target"].astype(int), result["prob_up"]))
    return metrics


def forward_probability(data: pd.DataFrame, name: str) -> tuple[pd.Timestamp, float]:
    """Train only on fully observable targets and forecast latest feature row."""
    observed = data.dropna(subset=FEATURES + ["target"])
    latest = data.dropna(subset=FEATURES).iloc[[-1]]
    # The newest label known at latest t belongs to t-2.
    observed = observed.loc[observed.index <= data.index[-3]]
    if observed.empty or len(observed["target"].unique()) < 2:
        raise ValueError("Insufficient labeled history")
    model = model_for(name)
    model.fit(observed[FEATURES], observed["target"].astype(int))
    return latest.index[-1], float(model.predict_proba(latest[FEATURES])[0, list(model.classes_).index(1)])
