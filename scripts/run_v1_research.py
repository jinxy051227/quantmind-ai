#!/usr/bin/env python3
"""Offline reproducibility CLI. Example: PYTHONPATH=src python scripts/run_v1_research.py --source demo --output data/out.zip"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from quantmind.core import demo_prices
from quantmind.audit import load_price_snapshot
from quantmind.v1 import StudyConfig, run_study, study_bundle, MODELS, json_safe


def main():
    p = argparse.ArgumentParser(description="QuantMind v1.0 controlled historical study")
    p.add_argument("--source",choices=["demo","csv"],required=True)
    p.add_argument("--csv", help="Required with --source csv; local OHLCV snapshot")
    p.add_argument("--ticker",default="DEMO")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--rows",type=int,default=1500)
    p.add_argument("--models",nargs="+",choices=MODELS,default=["Logistic Regression","Random Forest"])
    p.add_argument("--retrain",type=int,default=60)
    p.add_argument("--window",type=int,default=756,help="0 for expanding training window")
    p.add_argument("--threshold",type=float,default=0.55)
    p.add_argument("--fee-bps",type=float,default=10)
    p.add_argument("--slippage-bps",type=float,default=5)
    p.add_argument("--output",type=Path,default=Path("data/quantmind_v1_research.zip"))
    p.add_argument("--exclude-snapshot",action="store_true",help="Do not include licensed market data in bundle")
    args=p.parse_args()
    if args.source == "csv":
        if not args.csv:
            p.error("--csv is required when source=csv")
        prices=load_price_snapshot(Path(args.csv).read_bytes())
    else:
        prices=demo_prices(n=args.rows,seed=args.seed)
    cfg=StudyConfig(retrain_every=args.retrain,train_window=args.window or None,
                    threshold=args.threshold,fee_bps=args.fee_bps,slippage_bps=args.slippage_bps)
    study=run_study(prices,cfg,tuple(args.models),ticker=args.ticker,source=args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(study_bundle(study, include_prices=not args.exclude_snapshot))
    safe={"generated_archive":str(args.output),"ticker":study["ticker"],
          "selected_model":study["champion"],"split":study["split"],
          "test_metrics":study["test_metrics"],
          "test_scorecard":study["scorecard"].reset_index().to_dict(orient="records"),
          "snapshot_sha256":study["price_sha256"]}
    print(json.dumps(json_safe(safe),indent=2,default=str,allow_nan=False))

if __name__ == "__main__":
    main()
