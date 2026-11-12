"""QuantMind AI 1.0 portfolio dashboard.  PYTHONPATH=src streamlit run app.py"""
from __future__ import annotations
from dataclasses import replace
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from quantmind.core import demo_prices, market_prices
from quantmind.audit import load_price_snapshot
from quantmind.v1 import (StudyConfig, MODELS, run_study, study_bundle,
                          apply_policy, policy_scorecard)

st.set_page_config(page_title="QuantMind AI | Portfolio Research 1.0", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1380px;padding-top:1.15rem;padding-bottom:3rem}
.hero{background:linear-gradient(105deg,#132b45,#144d5f);color:#fff;padding:28px 33px;border-radius:18px;margin-bottom:18px}
.hero h1{font-size:2.2rem;color:#fff;margin:0;font-weight:770}
.hero p{color:#d8e8f4;margin:7px 0 0;font-size:1.0rem}
.subtle{font-size:.84rem;color:#8595a6}
[data-testid="stMetricValue"] {font-weight:720}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>QuantMind AI <span style="font-size:15px;font-weight:500">/ 1.0</span></h1><p>Research first. Predict responsibly. Validate by time, account for costs, reproduce results.</p></div>',unsafe_allow_html=True)
st.caption("Research / portfolio demonstration — no broker connection, no live execution, no investment recommendation. Historical AAPL dates have been viewed in earlier versions.")

with st.sidebar:
    st.header("Research setup")
    source = st.radio("Data source", ["Synthetic demo (offline)", "Yahoo Finance (historical)", "CSV snapshot (offline)"],
                      help="Synthetic data are artificial. Yahoo data are subject to provider terms. Keep exported snapshots private if needed.")
    if source.startswith("Yahoo"):
        tickers = st.multiselect("Research universe (max 4)", ["AAPL","MSFT","NVDA","SPY","GOOG","AMZN","7203.T","6758.T","9984.T"], default=["AAPL"], max_selections=4)
        start_date = st.date_input("History from", value=pd.Timestamp("2017-01-01"))
    elif source.startswith("CSV"):
        tickers = [st.text_input("Symbol label", "LOCAL").strip().upper() or "LOCAL"]
        upload = st.file_uploader("OHLCV snapshot (Date,Open,High,Low,Close,Volume)", type=["csv"])
    else:
        demo_seed = st.number_input("Demo random seed", min_value=0, max_value=999999, value=42)
        tickers = ["SYNTHETIC"]
    st.divider()
    st.subheader("Research protocol")
    chosen = st.multiselect("Candidate models", list(MODELS), default=["Logistic Regression", "Random Forest"],
                            help="Select the model with lowest validation Brier. Final-period labels never select a model.")
    retrain_every = st.select_slider("Retrain interval (sessions)", options=[20, 40, 60, 120, 240], value=60)
    window_label = st.selectbox("Training window", ["756 sessions (~3y)", "504 sessions (~2y)", "252 sessions (~1y)", "Expanding"], index=0)
    window = None if window_label == "Expanding" else int(window_label.split()[0])
    threshold = st.slider("Long entry P(up) threshold", min_value=0.50, max_value=0.80, value=0.55, step=0.01)
    fee = st.slider("Fee per position change (bps)", min_value=0, max_value=40, value=10)
    slip = st.slider("Illustrative slippage (bps)", min_value=0, max_value=30, value=5)
    with st.expander("Split and bootstrap"):
        train_fraction = st.slider("Initial train share", 0.50, 0.70, 0.60, 0.05)
        val_fraction = st.slider("Validation share", 0.10, 0.25, 0.20, 0.05)
        repeats = st.select_slider("Moving-block bootstrap repetitions", options=[0,100,300,600], value=300)
    run = st.button("Run research protocol", type="primary", use_container_width=True)
    st.caption("At close t: features known. Synthetic fill at close t+1; target return close t+1→t+2. Training embargo: 2 rows. No short positions.")
    st.divider()
    st.caption("Legacy research pages (v0.3 kept intact)")
    st.page_link("pages/1_Model_Comparison.py", label="Model Comparison (legacy)", icon="🧪")
    st.page_link("pages/2_Research_Audit.py", label="Research Audit (legacy)", icon="🔬")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(symbol: str, begin: str):
    return market_prices(symbol, begin)

if run:
    if not chosen:
        st.error("Choose at least one candidate model.")
    elif not tickers:
        st.error("Choose at least one ticker.")
    elif source.startswith("CSV") and upload is None:
        st.error("Upload a valid OHLCV CSV snapshot first.")
    else:
        cfg = StudyConfig(train_fraction=train_fraction, validation_fraction=val_fraction,
                          retrain_every=retrain_every, train_window=window,
                          threshold=threshold, fee_bps=fee, slippage_bps=slip,
                          bootstrap_repeats=repeats)
        computed, failures = {}, {}
        bar = st.progress(0, text="Starting reproducible study...")
        for i, ticker in enumerate(tickers):
            try:
                with st.spinner(f"Researching {ticker} with validation-only selection..."):
                    if source.startswith("Synthetic"):
                        prices = demo_prices(n=1500, seed=int(demo_seed))
                    elif source.startswith("CSV"):
                        prices = load_price_snapshot(upload.getvalue())
                    else:
                        prices = fetch_prices(ticker, str(start_date))
                    computed[ticker] = run_study(prices, config=cfg, candidates=tuple(chosen),
                                                  ticker=ticker, source=source)
            except Exception as exc:
                failures[ticker] = str(exc)
            bar.progress((i+1)/len(tickers), text=f"Processed {i+1} / {len(tickers)} symbols")
        st.session_state["quantmind_v1_results"] = computed
        st.session_state["quantmind_v1_failures"] = failures
        st.session_state["quantmind_v1_source"] = source
        bar.empty()

results = st.session_state.get("quantmind_v1_results", {})
failures = st.session_state.get("quantmind_v1_failures", {})
for symbol, msg in failures.items():
    st.error(f"{symbol}: {msg}")
if not results:
    st.info("Choose a source and run the research protocol. **Synthetic demo** works entirely offline; real historical data require an internet connection. Earlier v0.3 analysis remains available on the legacy pages.")
    st.markdown("""### What this 1.0 release can demonstrate
**Scientific workflow:** chronological training → validation-only model selection → subsequent final-period evaluation.  
**Engineering:** rolling retraining, deterministic pipelines, input fingerprints, archived experiment manifests, CI tests and a read-only API.  
**Business judgement:** benchmarks, hypothetical transaction costs, confidence intervals and explicit limitations.

For actual forward-only evidence, keep future paper-prediction logs *before* the future prices are known. Past AAPL data cannot become a pristine holdout again.""")
    st.stop()

if st.session_state.get("quantmind_v1_source", "").startswith("Synthetic"):
    st.warning("SYNTHETIC DATA — artificial prices. Treat the results as a software demonstration only.")
else:
    st.warning("EXPLORATORY HISTORICAL RESEARCH — the time split is rigorous in code, but some historical periods were already inspected by the project team. No live-edge claim.")

summary = []
for name, study in results.items():
    m = study["test_metrics"]
    ret = study["scorecard"].loc["ML strategy"]
    benchmark = study["scorecard"].loc["Buy & Hold"]
    summary.append({"Ticker":name, "Selected by validation":study["champion"],
                    "Final period":study["split"]["test_start"] + " → " + study["split"]["test_end"],
                    "Final accuracy":m["accuracy"],"Final Brier":m["brier"],
                    "Prior Brier":m["prior_brier"],"ML net return":ret["total_return"],
                    "Buy & Hold net":benchmark["total_return"],"ML turnover":ret["position_changes"]})
st.subheader("Research universe / same protocol, separate markets")
st.dataframe(pd.DataFrame(summary).style.format({"Final accuracy":"{:.1%}","Final Brier":"{:.4f}",
 "Prior Brier":"{:.4f}","ML net return":"{:.1%}","Buy & Hold net":"{:.1%}"}),
             use_container_width=True, hide_index=True)
st.caption("Different countries have different currencies and trading days. Results are NOT pooled into a single investable portfolio.")

symbol = st.selectbox("Inspect detailed study", list(results.keys()))
study = results[symbol]
metrics, test, split = study["test_metrics"], study["test"], study["split"]
tabs = st.tabs(["Overview", "Research design", "Strategy & costs", "Stability", "Artifacts"])
with tabs[0]:
    a,b,c,d = st.columns(4)
    a.metric("Validation-selected model", study["champion"])
    b.metric("Final-period accuracy", f"{metrics['accuracy']:.1%}", delta=f"{metrics['accuracy']-metrics['always_up_accuracy']:+.1%} vs always-up")
    c.metric("Final-period Brier ↓", f"{metrics['brier']:.4f}", delta=f"{metrics['brier']-metrics['prior_brier']:+.4f} vs prior", delta_color="inverse")
    d.metric("Test observations", f"{metrics['observations']:,}")
    st.caption("The score is not a verified probability of future profits. Accuracy baseline depends on the sample's positive-rate, not necessarily 50%.")
    fig = go.Figure()
    for col, name, color in [("ml_wealth","Selected ML (net)","#06b6d4"),
                              ("momentum_wealth","5-day momentum (net)","#f59e0b"),
                              ("buyhold_wealth","Buy & Hold (net)","#8b5cf6"),
                              ("cash_wealth","Cash", "#9ca3af")]:
        fig.add_trace(go.Scatter(x=test.index,y=test[col],mode="lines",name=name,line=dict(color=color,width=2)))
    fig.update_layout(title="Final-period hypothetical wealth (each starts at 1)",
                      yaxis_title="Wealth multiple",height=430,margin=dict(t=50,b=25),legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(study["scorecard"].style.format({
        "total_return":"{:+.2%}","annualized_return":"{:+.2%}","sharpe":"{:.2f}",
        "max_drawdown":"{:.2%}","volatility":"{:.2%}","exposure":"{:.1%}"}),use_container_width=True)
with tabs[1]:
    st.markdown("**Model selection** is based exclusively on validation Brier Score. The trading probability threshold is fixed before either validation or final scoring.")
    st.dataframe(study["validation"].style.format({
        "accuracy":"{:.1%}","balanced_accuracy":"{:.1%}","brier":"{:.4f}",
        "prior_brier":"{:.4f}","brier_skill":"{:.2%}","up_frequency":"{:.1%}",
        "always_up_accuracy":"{:.1%}","net_return":"{:+.1%}"}),use_container_width=True,hide_index=True)
    st.markdown("**Chronological split (no random shuffle)**")
    st.json(split, expanded=True)
    st.caption("At prediction row i, last training row ≤ i−3, even when retraining inside validation or final periods. Previous observed validation labels may enter later training, never future labels. Retrain interval and model candidates were chosen by the human researcher.")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test.index,y=test["prob_up"],name="Model P(up)",line=dict(color="#14b8a6")))
    fig2.add_trace(go.Scatter(x=test.index,y=test["prior_up"],name="Training-only positive-rate",line=dict(color="#f97316")))
    fig2.add_hline(y=study["config"]["threshold"],line_dash="dash",annotation_text="Entry threshold")
    fig2.update_layout(title="Frozen model probabilities and retraining-only prior", height=360, yaxis_title="P(up)")
    st.plotly_chart(fig2,use_container_width=True)
with tabs[2]:
    st.markdown("**Fees and slippage are charged on each complete position change** in this simplified close-to-close simulation. Not a real fill model.")
    cost_table = []
    cfg = StudyConfig(**study["config"])
    for fee_test in [0,5,10,20,30]:
        p = apply_policy(test, cfg.threshold, replace(cfg, fee_bps=fee_test))
        score = policy_scorecard(p).loc["ML strategy"]
        cost_table.append({"Fee bps":fee_test,"Slippage bps":cfg.slippage_bps,
                           "ML cumulative net":score["total_return"],"ML Sharpe":score["sharpe"],
                           "Position changes":score["position_changes"]})
    st.dataframe(pd.DataFrame(cost_table).style.format({"ML cumulative net":"{:+.2%}","ML Sharpe":"{:.2f}"}),
                 use_container_width=True,hide_index=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=[str(r["Fee bps"]) for r in cost_table],y=[r["ML cumulative net"]*100 for r in cost_table]))
    fig3.update_layout(title="Net cumulative return sensitivity to assumed fee",xaxis_title="Fee (bps) + fixed slippage",yaxis_title="Return (%)",height=320)
    st.plotly_chart(fig3,use_container_width=True)
    st.caption("Higher fee and slippage sensitivity is not evidence of causal predictability. The strategy remains hypothetical.")
with tabs[3]:
    st.markdown("**Market context is measured with trailing indicators available at the decision date.**")
    st.dataframe(study["regimes"].style.format({"accuracy":"{:.1%}","brier":"{:.4f}",
      "mean_net_daily_return":"{:+.4%}","annualized_net_daily_mean":"{:+.1%}"}),use_container_width=True,hide_index=True)
    info=study["bootstrap"]
    st.markdown("**Moving-block bootstrap: arithmetic annualized daily return differential (ML minus Buy & Hold)**")
    if info["samples"]:
        a,b,c = st.columns(3)
        a.metric("Observed difference",f"{info['observed']:+.2%}")
        b.metric("Descriptive 95% lower",f"{info['ci_low']:+.2%}")
        c.metric("Descriptive 95% upper",f"{info['ci_high']:+.2%}")
        st.caption(f"{info['samples']} block resamples, block size {info['block']}. NOT a confirmatory p-value, future guarantee, or correction for multiple testing.")
    else:
        st.info("Bootstrap disabled in experiment config.")
with tabs[4]:
    st.markdown("""### Reproducibility pack
The archive includes validation rankings, final-period predictions, policy results, regime diagnostics, model settings, environment versions, and optionally a private source-data snapshot. **Do not commit provider price history to a public GitHub repository without data rights.**""")
    include_prices=st.checkbox("Include the original OHLCV snapshot in my private download",value=True)
    bundle=study_bundle(study,include_prices=include_prices)
    st.download_button("Download reproducibility bundle (.zip)",data=bundle,
        file_name=f"quantmind_v1_{symbol.lower()}_research.zip",mime="application/zip",type="primary")
    st.download_button("Download final-period predictions (.csv)",data=test.to_csv(index=True).encode(),
        file_name=f"quantmind_v1_{symbol.lower()}_predictions.csv",mime="text/csv")
    st.json({"ticker":symbol,"data_sha256":study["price_sha256"],"champion":study["champion"],
             "split":split,"configuration":study["config"]},expanded=False)
    st.info("For truly new evidence, pre-register settings and log predictions before observing future returns. Historical AAPL performance in this app remains exploratory.")
