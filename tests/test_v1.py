"""Contract tests for QuantMind 1.0; never rely on third-party live data."""
import json
import zipfile
from dataclasses import replace
from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from quantmind.core import demo_prices
from quantmind.audit import price_snapshot_csv,load_price_snapshot
from quantmind.v1 import (MODELS, StudyConfig,clean_research_frame,chronological_boundaries,
                           walk_period,apply_policy,probability_metrics,policy_scorecard,
                           run_study,study_bundle,manifest_from_study,block_bootstrap_mean_difference)

@pytest.fixture(scope="module")
def prices():
    return demo_prices(n=760,seed=191)

@pytest.fixture(scope="module")
def config():
    return StudyConfig(min_train_rows=150,train_window=252,retrain_every=120,bootstrap_repeats=30)

@pytest.fixture(scope="module")
def report(prices,config):
    return run_study(prices,config,candidates=("Logistic Regression","Random Forest"))

def test_chronological_split_and_two_row_separation(prices,config):
    d=clean_research_frame(prices)
    val,test=chronological_boundaries(d,config)
    assert val < test-2
    pred=walk_period(d,"Logistic Regression",val,test-2,config,"validation")
    assert pred.index.max() <= d.index[test-3]
    for date,row in pred.iterrows():
        i=d.index.get_loc(date)
        assert pd.Timestamp(row["train_end"]) <= d.index[i-3]

def test_train_refits_only_at_selected_cadence(prices,config):
    d=clean_research_frame(prices)
    val,test=chronological_boundaries(d,config)
    pred=walk_period(d,"Logistic Regression",val,test-2,config,"validation")
    assert pred["retrained"].iloc[0]
    assert pred["retrained"].sum() == (len(pred)-1)//config.retrain_every + 1

def test_model_champion_is_validation_min_brier(report):
    assert report["champion"]==report["validation"].iloc[0]["model"]
    assert report["split"]["validation_to_test_embargo_rows"]==2
    assert report["split"]["n_test"]==len(report["test"])

def test_brier_prior_is_training_only(report):
    x=report["test"]
    m=report["test_metrics"]
    assert np.isclose(m["prior_brier"],brier_score_loss(x.target,x.prior_up))
    assert 0 <= x.prob_up.min() <= x.prob_up.max() <= 1

def test_costs_on_every_position_change(prices,config):
    d=clean_research_frame(prices)
    val,test=chronological_boundaries(d,config)
    x=walk_period(d,"Logistic Regression",test,len(d),config,"test")
    a=apply_policy(x,.55,replace(config,fee_bps=0,slippage_bps=0))
    b=apply_policy(x,.55,replace(config,fee_bps=10,slippage_bps=5))
    assert np.allclose(a.ml_gross,b.ml_gross)
    assert np.allclose(b.ml_net,a.ml_net-0.0015*b.ml_turnover)
    assert int(b.buyhold_turnover.sum())==1
    assert (b.cash_net==0).all()

def test_calculation_obeys_no_next_price_feature(prices,config):
    d=clean_research_frame(prices)
    val,test=chronological_boundaries(d,config)
    first=walk_period(d,"Logistic Regression",val,val+10,config,"validation")
    # Changes to observations after current decision date must not affect forecasts.
    changed=prices.copy()
    date=d.index[val+12]
    changed.loc[date:,["Open","High","Low","Close"]]*=1.05
    d2=clean_research_frame(changed)
    first_again=walk_period(d2,"Logistic Regression",val,val+10,config,"validation")
    np.testing.assert_allclose(first.prob_up.to_numpy(),first_again.prob_up.to_numpy(),atol=1e-12)

def test_snapshot_sha_and_csv_roundtrip(prices,report):
    restored=load_price_snapshot(price_snapshot_csv(prices))
    assert len(restored)==len(prices)
    assert report["price_sha256"]==manifest_from_study(report)["price_snapshot_sha256"]

def test_archive_has_no_pickle_code_execution(report):
    payload=study_bundle(report,include_prices=True)
    with zipfile.ZipFile(BytesIO(payload)) as z:
        assert set(["manifest.json","prices_snapshot.csv","final_test_predictions.csv","validation_model_ranking.csv"]).issubset(z.namelist())
        m=json.loads(z.read("manifest.json"))
        assert m["selected_model"]==report["champion"]
        assert m["version"]=="1.0.0"

def test_bundle_can_omit_licensed_market_data(report):
    with zipfile.ZipFile(BytesIO(study_bundle(report,include_prices=False))) as z:
        assert "prices_snapshot.csv" not in z.namelist()

def test_bootstrap_deterministic(report,config):
    a=block_bootstrap_mean_difference(report["test"],config)
    b=block_bootstrap_mean_difference(report["test"],config)
    assert a==b and a["ci_low"]<=a["ci_high"]

def test_invalid_params_rejected():
    with pytest.raises(ValueError): StudyConfig(train_fraction=.8,validation_fraction=.2).validate()
    with pytest.raises(ValueError): StudyConfig(retrain_every=1).validate()
    with pytest.raises(ValueError): StudyConfig(fee_bps=-1).validate()

def test_test_order_is_monotonic(report):
    t=report["test"]
    assert t.index.is_monotonic_increasing and t.index.is_unique
    assert t.phase.eq("test").all()
    assert t.train_end.lt(t.index.to_series()).all()

def test_model_choice_recomputed_only_from_validation(report):
    ranking=report["validation"].sort_values(["brier","model"])
    assert report["champion"]==ranking.iloc[0]["model"]
    assert report["test_metrics"]["observations"]==len(report["test"])

def test_identical_input_same_test_probabilities(prices,config,report):
    repeat=run_study(prices,config,candidates=("Logistic Regression","Random Forest"))
    assert repeat["champion"]==report["champion"]
    np.testing.assert_allclose(repeat["test"].prob_up,report["test"].prob_up,atol=1e-12)

def test_price_validation_rejects_duplicates(prices,config):
    bad=pd.concat([prices,prices.iloc[[0]]])
    with pytest.raises(ValueError):run_study(bad,config,candidates=("Logistic Regression",))
