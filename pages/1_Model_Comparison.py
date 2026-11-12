"""QuantMind v0.2 exploratory side-by-side comparison dashboard."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from quantmind.core import demo_prices, market_prices, features_and_labels
from quantmind.diagnostics import MODELS, compare_models, sensitivity, calibration_bins

st.set_page_config(page_title="QuantMind | Model Comparison", page_icon="🧪", layout="wide")
st.title("QuantMind AI · Model Comparison")
st.caption("v0.2 research diagnostics | Exploratory backtests; not financial forecasts or trading advice")
st.warning("These historical test periods have already been inspected. Model selection here is exploratory; reserve genuinely unseen future data for confirmatory evaluation.")

with st.sidebar:
    st.header("Comparison setup")
    source = st.radio("Data source", ["Yahoo Finance (historical)", "Synthetic demo (offline)"])
    ticker = st.selectbox("Ticker", ["AAPL", "MSFT", "NVDA", "SPY", "7203.T", "6758.T"], disabled=source.startswith("Synthetic"))
    start = st.date_input("History from", value=pd.Timestamp("2017-01-01"), disabled=source.startswith("Synthetic"))
    threshold = st.slider("Entry probability threshold", 0.50, 0.80, 0.55, 0.01)
    cost_bps = st.slider("Fee per entry/exit (bps)", 0, 50, 10, 1)
    folds = st.slider("Walk-forward folds", 3, 6, 5)
    seed = st.number_input("Demo seed", 0, 100000, 42, disabled=not source.startswith("Synthetic"))
    run = st.button("Compare models", type="primary", use_container_width=True)

@st.cache_data(ttl=3600)
def cached_history(symbol: str, from_date: str):
    return market_prices(symbol, from_date)

if run:
    try:
        with st.spinner("Running matched expanding-window experiments..."):
            prices = demo_prices(seed=int(seed)) if source.startswith("Synthetic") else cached_history(ticker, str(start))
            data = features_and_labels(prices)
            table, frames, baselines = compare_models(data, n_splits=folds, threshold=threshold, cost_bps=cost_bps)
            st.session_state.compare_result = (table, frames, baselines, source, ticker, threshold, cost_bps, folds, len(prices))
    except Exception as exc:
        st.error(f"Could not compare: {exc}")

if "compare_result" not in st.session_state:
    st.info("Use the sidebar to run Logistic Regression and Random Forest on the same dates.")
else:
    table, frames, baselines, source, ticker, threshold, cost_bps, folds, price_count = st.session_state.compare_result
    st.info(f"Saved run: {source} | {ticker if 'Yahoo' in source else 'Synthetic'} | threshold {threshold:.2f} | cost {cost_bps} bps | folds {folds} | price rows {price_count}")
    if source.startswith("Synthetic"):
        st.error("Synthetic data: never interpret these numbers as evidence about markets.")
    st.subheader("Out-of-sample · matched-model scorecard")
    st.dataframe(table.style.format({
        "direction_accuracy": "{:.1%}", "balanced_accuracy": "{:.1%}",
        "brier": "{:.4f}", "brier_vs_prior": "{:+.1%}",
        "total_return_net": "{:+.2%}", "total_return_gross": "{:+.2%}",
        "annualized_return_net": "{:+.2%}", "sharpe_net": "{:.2f}",
        "max_drawdown_net": "{:.2%}", "turnover": "{:.0f}", "exposure": "{:.1%}"}), use_container_width=True)
    st.caption("Brier vs prior = 1 - model Brier / training-fold class-rate Brier. Positive values are better. Gross/net cumulative returns differ due to entry/exit costs; this is not a causal test of model skill.")

    st.subheader("Non-ML prediction baselines")
    st.write(f"Positive labels in the shared OOS segment: {baselines.attrs['positive_rate']:.1%} | Observations: {baselines.attrs['sample_count']:,}")
    st.dataframe(baselines.style.format({"accuracy": "{:.1%}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.caption("Past-fold positive rate uses only already labeled training data. 'Always up' probability is 1.0, so its Brier score can be poor even if direction accuracy is high.")

    st.subheader("Compounded strategy wealth, same OOS dates")
    fig = go.Figure()
    for name in MODELS:
        series = frames[name]["model_net"]
        fig.add_trace(go.Scatter(x=series.index, y=(1 + series).cumprod(), mode="lines", name=name))
    first = frames[MODELS[0]]
    for name, col in (("5-day momentum", "momentum_net"), ("Buy & Hold", "buyhold_net")):
        series = first[col]
        fig.add_trace(go.Scatter(x=series.index, y=(1 + series).cumprod(), mode="lines", name=name))
    fig.update_layout(height=440, yaxis_title="Equity multiple", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trading-cost sensitivity (frozen signals)")
    l, r = st.columns(2)
    for col, name in ((l, MODELS[0]), (r, MODELS[1])):
        with col:
            st.write(f"**{name}**")
            sens = sensitivity(frames[name])
            st.dataframe(sens.style.format({"total_return": "{:+.2%}", "annualized_return": "{:+.2%}", "sharpe": "{:.2f}", "max_drawdown": "{:.2%}"}), use_container_width=True)
            st.download_button("Download OOS predictions", frames[name].to_csv().encode("utf-8"), file_name=f"quantmind_{name.lower().replace(' ','_')}.csv", key=name)

    st.subheader("Probability reliability (exploratory)")
    model = st.selectbox("Diagnostic model", MODELS)
    bins = calibration_bins(frames[model])
    st.dataframe(bins.style.format({"mean_predicted": "{:.1%}", "observed_rate": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.caption("These are descriptive OOS bins. Do not fit calibration on these test labels and then report the same period as untouched test data.")
