import io
import json
import zipfile
import pytest
import pandas as pd
from quantmind.core import demo_prices, features_and_labels
from quantmind.diagnostics import compare_models
from quantmind.audit import (price_snapshot_csv, load_price_snapshot, price_fingerprint,
                             fold_audit, fee_decomposition, experiment_archive)

@pytest.fixture(scope="module")
def experiment():
    prices=demo_prices(n=470,seed=21)
    data=features_and_labels(prices)
    table,frames,baselines=compare_models(data,n_splits=3,cost_bps=10)
    return prices,data,table,frames,baselines

def test_price_snapshot_roundtrip(experiment):
    prices=experiment[0]
    payload=price_snapshot_csv(prices)
    recovered=load_price_snapshot(payload)
    assert recovered.index.equals(prices.index.rename("Date"))
    assert price_fingerprint(recovered)==price_fingerprint(prices)
    assert recovered.shape==prices.shape

def test_reject_bad_csv():
    with pytest.raises(ValueError):
        load_price_snapshot(b'Date,Open,High,Low,Close\n2020-01-01,1,2,1,2\n')

def test_reject_duplicate_dates(experiment):
    prices=experiment[0]
    altered=pd.concat([prices,prices.iloc[[-1]]])
    with pytest.raises(ValueError,match="Duplicate"):
        price_snapshot_csv(altered)

def test_folds_report_training_only_prior(experiment):
    _,data,_,frames,_=experiment
    folds=fold_audit(data,frames,n_splits=3)
    assert len(folds)==6
    assert folds.groupby('model')['observations'].sum().eq(len(next(iter(frames.values())))).all()
    assert (folds['train_end'] < folds['test_start']).all()
    assert (folds['model_brier']>=0).all()

def test_fee_decomposition(experiment):
    *_,frames,_ = experiment
    impact=fee_decomposition(frames,cost_bps=10)
    assert (impact['gross_total_return']>=impact['net_total_return']).all()
    for name in impact.index:
        assert impact.loc[name,'position_changes']==frames[name]['model_turnover'].sum()

def test_bundle_contains_price_hash_and_predictions(experiment):
    prices,data,table,frames,base=experiment
    folds=fold_audit(data,frames,n_splits=3)
    impact=fee_decomposition(frames,10)
    zipped=experiment_archive(prices,frames,table,base,folds,impact,source='synthetic',ticker='SYN',
             start_date='2019-01-01',n_splits=3,threshold=.55,cost_bps=10)
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['snapshot_sha256']==price_fingerprint(prices)
        assert 'per_fold_audit.csv' in z.namelist()
        assert 'prices_snapshot.csv' in z.namelist()
        assert load_price_snapshot(z.read('prices_snapshot.csv')).shape==prices.shape

def test_offline_replay(tmp_path, experiment):
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'scripts'))
    from reproduce import replay
    prices,data,table,frames,base=experiment
    folds=fold_audit(data,frames,n_splits=3)
    fees=fee_decomposition(frames,10)
    blob=experiment_archive(prices,frames,table,base,folds,fees,
             source='synthetic',ticker='SYN',start_date='2019-01-01',
             n_splits=3,threshold=.55,cost_bps=10)
    f=tmp_path/'replay.zip'
    f.write_bytes(blob)
    assert len(replay(str(f)))==2
