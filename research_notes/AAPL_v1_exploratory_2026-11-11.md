# AAPL — QuantMind AI 1.0 exploratory re-analysis

**Status: retrospective / exploratory only. Not a new pristine holdout.**

The AAPL OHLCV in this study came from the user's previously saved local v0.3 Research Audit archive (`quantmind_v0.3_aapl_audit.zip`), not a fresh download or a market-data set committed to this repository. This report contains only aggregated numeric results; source prices are intentionally excluded from the public project files.

## Dataset and protocol

- Historical adjusted OHLCV range: **2017-01-03 to 2026-11-10**, 2,478 source rows (the supervised frame excludes warm-up and the last unlabeled rows).
- Initial training observations: **1,473**; initial train ending 2022-12-06.
- Model selection validation: 2022-12-07 to **2024-11-15**, **489** predictions; two validation→final gap rows are not scored.
- Later final evaluation: 2024-11-20 to **2026-11-06**, **492** predictions.
- Candidate models: Logistic Regression, Random Forest, Histogram Gradient Boosting.
- Validation-only selection by minimum Brier Score: **Logistic Regression**.
- Rolling model refit every **60** observed sessions, trailing **756** labeled sessions; training embargo **2 rows**.
- Long/cash threshold P(up) ≥ **0.55**, fee **10 bps** plus illustrative slippage **5 bps** per position change.

## Final-period evidence (hypothetical only)

| Metric | Result |
|---|---:|
| Selected ML model direction accuracy | **49.59%** |
| Always-up direction accuracy | **53.46%** |
| Model Brier Score (lower better) | **0.25325** |
| Training-only prior-probability Brier | **0.24913** |
| Model Brier Skill versus prior | **−1.66%** |
| ML strategy net cumulative return | **−28.10%** |
| Buy & Hold net cumulative return | **+53.39%** |
| 5-day momentum net cumulative return | **−3.96%** |
| ML strategy sample Sharpe | **−0.75** |
| ML strategy maximum drawdown | **−39.62%** |

## Interpretation

The selected machine-learning strategy **failed to beat** both the simple probability baseline and buy-and-hold in this dataset and period. The results do not support a claim of reliable AI prediction or trading edge. Much of this AAPL history had already been viewed during v0.1–v0.3 research; the chronological protocol stops obvious software-level label leakage, but it does not remove human research-selection bias.

The evaluation differs deliberately from v0.3's five-fold expanding-window study, so v0.3 and v1.0 return figures should not be compared as though they were identical protocols. The new v1.0 results depend on model, split, retraining, fee and slippage settings and may differ in another environment. See the provided private study bundle in your own downloaded experiments for full inputs and per-date outputs.

## Future research

Pre-register next settings, begin dated forward-only paper forecasts, preserve input hashes and independent timestamps, and do not select strategies by newly observed final-period outcomes. Study a broader pre-specified symbol list, different trading venues, survivorship and corporate-action biases, before making claims of generality.
