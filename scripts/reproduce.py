"""Offline replay of a QuantMind research bundle, without fetching web data.

Usage: PYTHONPATH=src python scripts/reproduce.py path/to/quantmind_v0.3_aapl_audit.zip
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantmind.audit import load_price_snapshot, price_fingerprint
from quantmind.core import features_and_labels
from quantmind.diagnostics import compare_models


def replay(archive_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as z:
        cfg = json.loads(z.read("manifest.json"))
        raw = z.read("prices_snapshot.csv")
        expected = cfg["snapshot_sha256"]
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Input hash mismatch: {actual} != {expected}")
        prices = load_price_snapshot(raw)
        if price_fingerprint(prices) != expected:
            raise ValueError("CSV roundtrip does not reproduce the original input hash")
        saved = pd.read_csv(io.BytesIO(z.read("model_scorecard.csv")), index_col=0)
    data = features_and_labels(prices)
    actual_table, _, _ = compare_models(data, n_splits=cfg["n_splits"],
                                      threshold=cfg["threshold"],
                                      cost_bps=cfg["fee_per_position_change_bps"])
    if set(saved.index) != set(actual_table.index):
        raise AssertionError("Different models in replay")
    fields = saved.select_dtypes(include=["number"]).columns.intersection(actual_table.columns)
    if not fields.size:
        raise AssertionError("Missing numeric scores in archive")
    # Same hardware/packages should ordinarily be exact; small floating differences
    # may occur across package versions or native linear algebra implementations.
    for model in actual_table.index:
        np.testing.assert_allclose(actual_table.loc[model, fields].astype(float),
                                   saved.loc[model, fields].astype(float),
                                   rtol=1e-7, atol=1e-9, equal_nan=True)
    return actual_table


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/reproduce.py path/to/research_bundle.zip")
    scores = replay(sys.argv[1])
    print("PASS: input SHA-256 and archived metrics verified against offline recomputation")
    print(scores[["direction_accuracy", "brier", "total_return_net"]].to_string())
