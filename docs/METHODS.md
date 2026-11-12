# QuantMind AI 1.0 — Methodology & validity

## Research question

Can statistical classifiers using only adjusted daily OHLCV features predict a later close-to-close return sign and deliver cost-adjusted simulated performance beyond simple baselines? The null-friendly approach means a negative result is valid and should be disclosed.

## Data, features and target

Input is OHLCV daily data, normalized as a unique sorted daily index with positive/consistent prices and non-negative volume. Features available at the end of session t: 1/5/10-day returns, 10/20-day historical realized volatility, moving-average gaps, intraday range and volume change. `trend_20` is used for descriptive market-state grouping.

**Target** `r(t) = adjusted_close(t+2) / adjusted_close(t+1) - 1`, label `1[r(t)>0]`. Signal is computed after close t. Hypothetical order at close t+1, outcome through close t+2. This is not necessarily an attainable closing-auction fill and needs exchange-specific execution detail before use in reality. Daily position signals correspond to non-overlapping return intervals; 1→0 transitions incur an assumed cost.

## Model selection protocol

1. Compute valid, fully labeled feature rows; determine chronological 60% training, 20% validation, 20% later test boundaries (user-configurable within safe ranges).
2. At each validation date i, fit on rows `[:i-2]` (last training row i−3), with a trailing window or expanding history; retrain every k trading days. Two rows are embargoed relative to the signal date.
3. Exclude the **last two validation rows** from scoring so the remaining validation labels end **before** the first final test date. This prevents validation-time model selection from using test-era closing prices.
4. Rank candidate classifiers by **validation Brier Score only**. Trading entry threshold is fixed before validation; test outcomes do not influence the choice.
5. Freeze the selected model identity; in the later test period, allow walk-forward refits on previously *observed* labels only. Score final-period predictions once per intended study.
6. Compare train-only historical positive-rate, direction majority baseline, 5-day momentum, buy-and-hold and cash. Publish all metric values including underperformance.

> A chronological split does not magically make already inspected AAPL history an untouched confirmatory test. From v0.1 to v0.3, project members saw much of 2018–2026 historical AAPL. The v1.0 re-analysis is therefore **exploratory**, and future genuinely unseen observations are required for stronger evidence.

## Hypothetical strategy and limitations

`position(t)=1[P(up|features(t))≥threshold]`. No shorting. `gross_return(t)=position(t)*r(t)`. `position_change=abs(position(t)−position(t−1))`, charging the initial entry. `net_return=gross_return−position_change*(fee_bps+slippage_bps)/10000`. Simple subtraction can understate real complex execution effects. Wealth = cumulative product of `1+net_return`.

Buy-and-hold pays an initial entry charge; 5-day momentum uses past 5-day return sign; cash has zero return before interest (not included). Returns are not matched for beta, volatility target or market exposure: raw strategy returns alone do not prove factor-adjusted alpha.

## Evaluation

Classification: accuracy, balanced accuracy, Brier score, train-only positive-rate Brier and Brier Skill Score. Investment simulation: total return, annualized compounded return, sample Sharpe assuming zero risk-free rate and 252 sessions, maximum drawdown, annualized realized volatility, turnover and exposure. No annual dividend cash payments are separately modeled.

Descriptive moving-block bootstrap re-samples contiguous blocks of strategy-vs-buyhold **daily return differences** and reports 2.5/97.5 percentiles of annualized arithmetic mean difference. It does not adjust for model/parameter search, selection bias or other multiple comparisons; it is NOT a p-value or guarantee.

Regime groups are by trailing 20-day return sign and 20-day trailing realized volatility above/below the *pre-test* median. Avoid testing many post-hoc subgroups and reporting only favorable ones.

## Threats to validity

- Vendor adjusted historical prices can be revised and not directly executable at the same quoted levels; use point-in-time raw prices and transaction-specific fill assumptions for practical execution simulation.
- Single-asset historical selection and survivor bias; multi-symbol results are separate, not portfolio aggregations.
- Market changing rules, missing events, look-ahead due to knowledge of the ticker universe, and unrealistically simple flat fees and slippage.
- Retrospective experimentation and re-running experiments multiple times may inflate reported superiority even with clean procedural splits.
- Model refits at prescribed intervals; historical features do not include raw fundamentals, macro release lags, order-book liquidity, or news timestamps.
- No real broker, no live execution, no SSO, no financial institution security claims, no investment advice.

## How to make a stronger claim

Pre-register the model candidate set, thresholds, decision time, costs and evaluation horizon. Start the paper journal on the calendar date the prediction is made. Store signed or independently timestamped immutable outputs. Reconcile realized future outcomes after the scheduled horizon without retroactive editing. Collect multiple market regimes before making any generalization. If publishing hypotheses tested across assets, report multiple-testing-adjusted inference and account for common market factors.
