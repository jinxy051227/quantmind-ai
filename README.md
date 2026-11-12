# QuantMind AI 1.0 — Reproducible Stock Prediction Research Studio

[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-2563eb)](https://www.python.org/) [![Tests](https://img.shields.io/badge/CI-pytest-0d9488)](.github/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-64748b)](LICENSE)

**Research, not trading.** A full offline-capable research workflow for testing whether price-based ML classifiers improve on simple probability and investing baselines **out of sample, after hypothetical transaction costs**. It preserves the earlier v0.3 comparison/audit pages and adds a v1.0 study engine, multi-symbol dashboard, read-only demo API, and forward-only paper log helper.

**中文：** 一套适合 GitHub 作品集的机器学习金融研究平台：时间序列训练／验证／最终评价分离，滚动重新训练、多模型比较、简单基准、风险分析、研究 ZIP 与数据哈希。**本项目不能证明可以稳定预测股价或赚取利润。**

**日本語：** 時系列順序を守って株価予測モデルを評価する研究用 Web アプリです。検証期間だけでモデルを選び、その後の期間で確率予測・手数料込みの仮想運用成績を測定します。**実売買・投資勧誘用ではありません。**

## Current research finding (not a trading performance claim)

On the **previously inspected** AAPL historical series, the v1.0 validation procedure selected Logistic Regression. The later evaluation interval (2024-11-20 to 2026-11-06) reported **49.59% direction accuracy**, **−28.10% illustrative net ML strategy return**, and **+53.39% illustrative buy-and-hold return**. These are exploratory historical results, not a clean confirmatory holdout or evidence of profitable forecasting. Reproduction data are kept private, while the code and a short text report are in [`research_notes/`](research_notes/).

**What this project demonstrates:** data engineering, chronological evaluation, baselines, transaction-cost analysis, experiment auditing, and software engineering—not stock-picking alpha.

## Publish safely

The public repository contains only source, documentation, tests and **non-proprietary** research summaries. Keep downloaded OHLCV snapshots, local forecasts, `.env` files, and reproducibility ZIP exports private. See [GitHub publishing guide (中文)](GITHUB_PUBLISH_ZH.md).

## Start locally (macOS / Linux)

Extract the ZIP, open Terminal *inside the extracted `quantmind-ai-v1.0` directory*, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m streamlit run app.py
```

Open <http://localhost:8501>. To close: `Control+C` in Terminal. The **Synthetic demo** option works without an API key or market-data network access (package installation still needs internet). A real stock-data run uses Yahoo Finance via `yfinance`; access or provider restrictions may block requests. You can upload your own local `Date,Open,High,Low,Close,Volume` CSV instead. If using the old v0.3 audit ZIP, extract its `prices_snapshot.csv` and upload it in the new app's **CSV snapshot** mode.

To run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

## What v1.0 includes

| Research / engineering feature | How it works |
|---|---|
| **Time-aware modeling** | Features at close *t*; hypothetical entry at close *t+1*; target *close(t+2) / close(t+1) − 1* |
| **Train / validate / test** | Initial training history, forward validation for **model selection**, later final-period evaluation; 2-row gap at validation→test boundary |
| **Rolling refits** | Retrain every 20/40/60/120/240 sessions on trailing 252/504/756 sessions or expanding history |
| **Models** | Logistic Regression, Random Forest, Histogram Gradient Boosting, plus a training-only positive-rate probability baseline |
| **Strategies** | Probability-threshold long/cash policy; 5-day momentum; buy/hold; cash; cost/slippage sensitivity |
| **Quality diagnostics** | Accuracy, balanced accuracy, Brier, baseline Brier, returns, Sharpe, volatility, drawdown, turnover, exposure |
| **Uncertainty & market context** | Descriptive moving-block bootstrap CI; volatility/trend groups defined using known historical features |
| **Multi-symbol dashboard** | Compare separate studies across US/JP symbols (no invalid cross-currency pooling) |
| **Reproducible artifacts** | Local ZIP with manifest, file hash, config, environment, predictions, scores, and optional private price CSV |
| **Engineering** | Modular Python, deterministic random seeds, pytest, GitHub Actions CI, Docker, optional FastAPI demo endpoint, CLI tools |
| **Previous work** | Original v0.2 Model Comparison and v0.3 Research Audit Streamlit pages retained |

**Default research protocol:** 60% initial history, 20% validation, 20% later final evaluation; 2-row embargo after validation before test; 60-session refits on trailing 756-session history, 55% signal threshold, 10 bps fee + 5 bps illustrative slippage. The final test window is **procedurally separated** from validation, but historical AAPL data from the earlier versions has already been repeatedly viewed; it is not honestly a pristine confirmatory test. More models and parameter combinations increase overfitting risk. See [Methods](docs/METHODS.md).

### Data timing

```text
Close t: known OHLCV → calculate features → estimate P(up)
Close t+1: hypothetical execution, not a guaranteed fill
Close t+2: evaluate target interval; charge fees on position changes

For prediction at row i:
  model.fit only on rows j <= i-3;
  last training label's price endpoints are in the known past.
```

There is **no real brokerage integration**, leverage, shorts, automatic trades, secrets handling, or claims of executable close fills. Adjusted OHLCV are an approximation; slippage is illustrative. Cross-market outputs are not an investable portfolio. Adjusted price feeds, survivorship, delistings, splits, taxes, dividends and publication timing are research limitations.

## Reproduce an offline experiment

```bash
PYTHONPATH=src python scripts/run_v1_research.py \
  --source demo --seed 42 --models 'Logistic Regression' 'Random Forest' \
  --output data/demo_v1_bundle.zip
```

A private downloaded AAPL/other-symbol CSV snapshot:

```bash
PYTHONPATH=src python scripts/run_v1_research.py \
  --source csv --csv /path/to/prices_snapshot.csv --ticker AAPL \
  --models 'Logistic Regression' 'Random Forest' 'Hist Gradient Boosting' \
  --output data/aapl_v1_bundle.zip --exclude-snapshot
```

`--exclude-snapshot` omits copyrighted/provider market prices from the bundle. **Do not commit your own licensed historical price snapshots to public GitHub.** The `data/` folder is git-ignored.

For a *paper-only* prediction log after collecting data (not proof of contemporaneous collection by itself):

```bash
PYTHONPATH=src python scripts/log_paper_forecast.py \
  --csv /path/to/todays_prices_snapshot.csv --ticker AAPL
```

Append-only JSON lines with a UTC timestamp and the available price date are written under `data/`. The helper detects duplicate ticker/date logs. The backdated snapshot risk remains: use the tool when new data arrives, keep independent logs, and evaluate only once future outcomes become available.

## Optional read-only API

```bash
python -m pip install -r requirements-api.txt
PYTHONPATH=src uvicorn service_api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>. The API only exposes `/health` and a **bounded, offline synthetic** study endpoint `POST /v1/demo/study`. It offers no real-trading endpoints or personal market-data upload. See [Architecture](docs/ARCHITECTURE.md).

## Docker

```bash
docker build -t quantmind-ai:1.0 .
docker run --rm -p 8501:8501 quantmind-ai:1.0
```

For public deployment, prefer the offline synthetic demo. Yahoo Finance/yfinance's published documentation says it is designed for research/education and the underlying Yahoo API is intended for personal use; review terms before a public, shared, or commercial service. See [yfinance documentation](https://ranaroussi.github.io/yfinance/) and the [download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

## Research report & interview presentation

- [Research methods, leakage controls, threats to validity](docs/METHODS.md)
- [Application architecture and security boundary](docs/ARCHITECTURE.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Japanese 5-minute interview demonstration](docs/DEMO_SCRIPT_JP.md)
- [Chinese portfolio briefing](docs/PORTFOLIO_BRIEF_ZH.md)
- [AAPL exploratory study from the user's archived v0.3 price snapshot](research_notes/AAPL_v1_exploratory_2026-11-11.md)
- [Original v0.3 research note](research_notes/AAPL_exploratory_2026-11-11.md)

## Versioning and scientific integrity

Version `1.0` means **the planned portfolio research application is functionally packaged**, not scientifically validated forecasting performance or production deployment. We do not advertise predictive alpha. Even if an ML model loses to buy-and-hold, the result remains in the study archive. Statistical confidence intervals here are descriptive, not formal multiple-testing-corrected significance guarantees.

**Supported sample deployment environment:** Python 3.11–3.13; Python 3.14 may work with sufficiently recent wheels but is not included in CI. Exact environment and input hashes are saved per research bundle; release requirements are constrained ranges, not a bit-for-bit cross-platform lockfile. Numerical results from tree models may differ across library versions.

### Official references

- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn probability calibration and disjoint calibration requirements](https://scikit-learn.org/stable/modules/calibration.html)
- [yfinance project/data use disclaimer](https://ranaroussi.github.io/yfinance/)
- [Streamlit documentation](https://docs.streamlit.io/)

MIT licensed **source code only**. Market-data rights are not granted by this repository. See [LICENSE](LICENSE).
