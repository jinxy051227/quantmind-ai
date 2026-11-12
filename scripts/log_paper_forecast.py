#!/usr/bin/env python3
"""Log one as-of probability for later paper-only follow-up (NO trading).

This records a prediction timestamp but CANNOT independently establish that an
old snapshot had not been observed; genuine forward testing requires timely use.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from quantmind.audit import load_price_snapshot, price_fingerprint
from quantmind.core import features_and_labels, forward_probability
from quantmind.v1 import StudyConfig, run_study, MODELS


def create_record(snapshot: bytes, ticker: str, models: tuple[str,...], as_of_utc: datetime | None = None) -> dict:
    prices=load_price_snapshot(snapshot)
    # Select with historical validation; use all fully labeled rows to fit last known model.
    study=run_study(prices,StudyConfig(bootstrap_repeats=0),models,ticker=ticker,source="CSV paper journal")
    asof, probability=forward_probability(features_and_labels(prices),study["champion"])
    created=as_of_utc or datetime.now(timezone.utc)
    return {"recorded_at_utc":created.isoformat(),"ticker":ticker.upper(),
            "last_available_price_date":asof.date().isoformat(),
            "predicted_event":"close(t+2) / close(t+1) > 1 (from features at t)",
            "selected_model":study["champion"],"up_probability":probability,
            "snapshot_sha256":price_fingerprint(prices),
            "warning":"Paper forecast only, snapshot could be historical or backfilled. No trades."}


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            prior=json.loads(line)
            if (prior.get("ticker"), prior.get("last_available_price_date")) == (record["ticker"],record["last_available_price_date"]):
                raise ValueError("A paper forecast already exists for this ticker and as-of date")
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--csv",required=True)
    a.add_argument("--ticker",required=True)
    a.add_argument("--journal",default="data/paper_forecasts.jsonl")
    a.add_argument("--models",nargs="+",choices=MODELS,default=["Logistic Regression"])
    opts=a.parse_args()
    record=create_record(Path(opts.csv).read_bytes(),opts.ticker,tuple(opts.models))
    append_record(Path(opts.journal),record)
    print(json.dumps(record,indent=2))

if __name__ == "__main__":main()
