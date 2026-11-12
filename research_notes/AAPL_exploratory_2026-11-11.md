# QuantMind AI · AAPL exploratory research log

Date: 2026-11-11 (Japan Standard Time)  
Status: **EXPLORATORY / NOT CONFIRMATORY / NO LIVE TRADING**

## Study configuration

- Source: Yahoo Finance (historical), AAPL, requested start `2017-01-01`
- Models: Logistic Regression, Random Forest, 5 expanding-window folds, embargo 2 rows
- Decision and execution: features at close `t`; enter at close `t+1`; hold through close `t+2`
- Long/cash probability threshold: 0.55
- Fee assumption: 10 bps per entry/exit position change
- Samples: 2,045 out-of-sample predictions; 2,478 input price rows shown by the v0.2 UI
- Numbers below were **manually transcribed from user-provided screenshots**, rather than regenerated from an archived vendor price file. Archive a v0.3 snapshot before making reproducibility claims.

## Classification and probability results

| Metric | Logistic Regression | Random Forest | Benchmark |
|---|---:|---:|---:|
| Direction accuracy | 52.1% | 51.1% | Always-up 53.4% |
| Balanced accuracy | 49.1% | 49.0% | — |
| Brier score (lower better) | 0.2570 | 0.2543 | Fold-specific historical base rate 0.2489 |
| Brier skill vs fold base rate | -3.2% | -2.2% | 0.0% |

Interpretation: neither model beats the always-up direction-accuracy comparator or the fold-specific base-rate Brier benchmark. These metrics alone do not show statistical significance.

## Trading fee sensitivity: compounded cumulative return

| Cost (bps) | Logistic Regression | Random Forest |
|---|---:|---:|
| 0 | +59.04% | +14.83% |
| 5 | +29.18% | -19.23% |
| 10 | +4.91% | -43.20% |
| 20 | -30.82% | -71.92% |
| 30 | -54.40% | -86.13% |

At 10 bps, the earlier v0.1 screenshot showed logistic regression annualized return +0.59%, Sharpe 0.13, max drawdown -40.97%, and 416 position changes. The Random Forest screenshot showed ~-6.59% annualized, Sharpe -0.19, max drawdown -61.44%, and 701 changes. The updated v0.2 audit shows -61.91% maximum drawdown for Random Forest; there was no fixed data snapshot across runs, so these exact numbers should not be assumed identical across historical data refreshes.

The same 10-bps historical strategy comparison showed 5-day momentum annualized ~12.11%, Buy & Hold annualized ~26.41%, with Buy & Hold compounded cumulative return ~569.8%. All were exploratory historical calculations, not prospective live outcomes.

## Reliability calibration bins for Logistic Regression

| Predicted range | Count | Mean predicted | Observed positive |
|---|---:|---:|---:|
| (0.4,0.6] | 1,792 | 53.6% | 53.7% |
| (0.6,0.8] | 206 | 66.7% | 51.0% |
| (0.8,1.0] | 47 | 88.6% | 53.2% |

High-confidence outputs appear overconfident in this sample. The final bin is small (47 observations). No calibrated probability model has been trained or independently validated.

## Controls and next experiments

1. Freeze raw OHLCV into a SHA-256 identified CSV snapshot and export a manifest + code commit hash.
2. Replay the same input offline to validate reproducible metrics. Inspect any changes from the earlier data refresh.
3. Use v0.3 per-fold audit to identify whether a few periods dominate gains/losses and model Brier.
4. Quantify gross vs net return, average position changes and fee sensitivity, without tuning thresholds on already-seen test dates.
5. Begin future-only paper evaluation for genuinely unseen observations. Model selection on the 2018–2026 AAPL backtest cannot be treated as untouched confirmation.

The project is educational research, not a trading signal or investment advice.
