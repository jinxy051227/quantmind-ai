"""QuantMind AI v0.3: reproduce and audit exploratory historical experiments."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantmind.core import demo_prices, market_prices, features_and_labels
from quantmind.diagnostics import compare_models, MODELS, calibration_bins
from quantmind.audit import (
    price_snapshot_csv, load_price_snapshot, price_fingerprint, fold_audit,
    fee_decomposition, experiment_archive,
)

st.set_page_config(page_title="QuantMind | Research Audit", page_icon="🔬", layout="wide")
st.title("QuantMind AI · Research Audit")
st.caption("v0.3 | Price snapshot • per-fold diagnostics • reproducible experiment bundle")
st.warning("Historical AAPL periods were previously inspected. This is exploratory analysis, not untouched confirmatory validation or investment advice.")

with st.sidebar:
    st.header("Reproducible experiment")
    source = st.radio("Input data", ["Yahoo Finance (historical)", "Load CSV snapshot", "Synthetic demo (offline)"])
    ticker = st.selectbox("Ticker", ["AAPL", "MSFT", "NVDA", "SPY", "7203.T", "6758.T"], disabled=source != "Yahoo Finance (historical)")
    start = st.date_input("History from", value=pd.Timestamp("2017-01-01"), disabled=source != "Yahoo Finance (historical)")
    upload = st.file_uploader("Price CSV snapshot", type=["csv"], disabled=source != "Load CSV snapshot")
    threshold = st.slider("Long entry probability threshold", 0.50, 0.80, 0.55, 0.01)
    fee = st.slider("Entry/exit fee (bps)", 0, 50, 10, 1)
    splits = st.slider("Walk-forward folds", 3, 6, 5)
    seed = st.number_input("Demo seed", min_value=0, max_value=100000, value=42,
                           disabled=source != "Synthetic demo (offline)")
    run = st.button("Run research audit", type="primary", use_container_width=True)
    st.caption("Historical simulations only. No live execution or broker connections.")

@st.cache_data(ttl=3600, show_spinner=False)
def history(symbol, date):
    return market_prices(symbol, date)

if run:
    try:
        with st.spinner("Building a fold-matched, reproducible research audit..."):
            if source == "Load CSV snapshot":
                if upload is None:
                    raise ValueError("Choose a CSV snapshot first")
                prices = load_price_snapshot(upload.getvalue())
                tag = "CSV snapshot"
                sym = "uploaded"
            elif source.startswith("Synthetic"):
                prices = demo_prices(seed=int(seed))
                tag = "synthetic (not market evidence)"
                sym = "SYNTHETIC"
            else:
                prices = history(ticker, str(start))
                tag = "Yahoo Finance adjusted OHLCV"
                sym = ticker
            snapshot = price_snapshot_csv(prices)
            prices = load_price_snapshot(snapshot)  # run against the exact exported bytes
            features = features_and_labels(prices)
            table, frames, baselines = compare_models(features, n_splits=int(splits), threshold=threshold, cost_bps=fee)
            folds = fold_audit(features, frames, n_splits=int(splits))
            impact = fee_decomposition(frames, fee)
            bundle = experiment_archive(prices, frames, table, baselines, folds, impact,
                                        source=tag, ticker=sym, start_date=(str(start) if source == "Yahoo Finance (historical)" else str(prices.index.min().date())),
                                        n_splits=int(splits), threshold=threshold, cost_bps=fee)
            st.session_state.audit_run = (table, frames, baselines, folds, impact, bundle, snapshot,
                                         tag, sym, threshold, fee, splits, price_fingerprint(prices),
                                         prices.index.min(), prices.index.max())
    except Exception as exc:
        st.error(f"Research audit failed: {exc}")

if "audit_run" not in st.session_state:
    st.info("Run the audit, then download the study bundle. Import the included prices_snapshot.csv to replicate the price input without Yahoo Finance.")
else:
    table, frames, baselines, folds, impact, bundle, snapshot, tag, sym, threshold, fee, splits, digest, first_date, last_date = st.session_state.audit_run
    st.info(f"Saved audit | {tag} | {sym} | {first_date.date()} – {last_date.date()} | folds={splits} | threshold={threshold:.2f} | fee={fee}bps")
    st.code(f"Price snapshot SHA-256: {digest}")
    st.download_button("Download complete reproducibility bundle (.zip)", data=bundle,
                       file_name=f"quantmind_v0.3_{sym.lower()}_audit.zip", mime="application/zip",
                       type="primary")
    st.download_button("Download price snapshot (.csv)", data=snapshot,
                       file_name=f"quantmind_{sym.lower()}_prices_snapshot.csv", mime="text/csv")
    st.caption("Save the ZIP to the GitHub research artifacts folder. The hash identifies input data, not the fitted model or code environment.")
    st.subheader("Prediction scorecard on matched OOS dates")
    st.dataframe(table.style.format({"direction_accuracy":"{:.1%}","balanced_accuracy":"{:.1%}",
             "brier":"{:.4f}","brier_vs_prior":"{:+.1%}","total_return_net":"{:+.2%}",
             "total_return_gross":"{:+.2%}","annualized_return_net":"{:+.2%}",
             "sharpe_net":"{:.2f}","max_drawdown_net":"{:.2%}","turnover":"{:.0f}","exposure":"{:.1%}"}),
             use_container_width=True)
    st.subheader("Per-fold audit — model stability")
    st.dataframe(folds.style.format({"actual_up_rate":"{:.1%}","model_accuracy":"{:.1%}",
        "always_up_accuracy":"{:.1%}","model_brier":"{:.4f}","past_rate_brier":"{:.4f}",
        "brier_skill":"{:+.1%}","gross_return":"{:+.2%}","net_return":"{:+.2%}",
        "turnover":"{:.0f}","exposure":"{:.1%}"}), hide_index=True, use_container_width=True)
    st.caption("These are exploratory historical folds, not proof of a live edge. Comparing or selecting strategies using them reuses inspected labels.")
    st.subheader("Transaction-cost impact")
    st.dataframe(impact.style.format({"cost_bps":"{:.1f}","position_changes":"{:.0f}",
        "gross_total_return":"{:+.2%}","net_total_return":"{:+.2%}",
        "cumulative_wealth_impact":"{:+.2%}","sum_of_daily_cost_rates":"{:.2%}",
        "gross_sharpe":"{:.2f}","net_sharpe":"{:.2f}"}), use_container_width=True)
    st.caption("Sum of daily cost rates is NOT the total compounded equity penalty. Look at gross vs net wealth to compare the latter.")
    st.subheader("Reliability diagnostics")
    selected = st.selectbox("Model", MODELS)
    bins = calibration_bins(frames[selected], bins=5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=bins["mean_predicted"],y=bins["observed_rate"],
                            mode="markers+lines",name=selected,
                            text=[f"count={v}" for v in bins["count"]]))
    fig.update_layout(xaxis_title="Mean predicted probability",yaxis_title="Observed positive fraction",
                      xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]),height=400)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(bins.style.format({"mean_predicted":"{:.1%}","observed_rate":"{:.1%}"}), hide_index=True)
    st.caption("Reliability bins are descriptive. Calibration would require separately trained calibrators and independent evaluation dates.")
    st.subheader("Next: forward-only paper research")
    st.write("Future observations not yet seen by the team are needed for a genuinely confirmatory evaluation. Do not choose a winner from these historical folds and relabel the same dates as untouched test data.")
