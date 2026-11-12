# Reproducibility checklist

- [ ] Commit the source and note the commit SHA.
- [ ] Save the input OHLCV snapshot privately and its SHA-256 fingerprint.
- [ ] Keep the exact experiment settings; both manifest and CLI arguments are preserved.
- [ ] Record Python, NumPy, pandas and scikit-learn versions from the exported manifest.
- [ ] Distinguish numeric tolerance from bit-for-bit identical probability arrays across environments.
- [ ] Verify the validation model ranking is unchanged by test-era label changes.
- [ ] Verify returns deduct an initial entry fee and each subsequent position change.
- [ ] Re-run on imported `prices_snapshot.csv` offline and compare metrics.
- [ ] Confirm all GitHub CI tests pass on the published commit.
- [ ] Do not commit the source prices or private data without distribution rights.
- [ ] Label already-inspected AAPL history as exploratory.
- [ ] For genuinely unseen evaluation, save the forecast **before** later prices become available.

## Data-file portability

Canonical snapshot bytes are `Date,Open,High,Low,Close,Volume` CSV, UTF-8, date-only index, 17-digit float precision and Unix line endings. Input hash identifies those exact normalized bytes, not the trained model, code version, or environment. Differences between cross-platform ML libraries may cause slight probability changes even with identical inputs and seeds. Save `pip freeze` as an additional private artifact for exact provenance.

## Re-run on the old v0.3 audit

1. Extract `prices_snapshot.csv` from your `quantmind_v0.3_aapl_audit.zip` downloaded from the original Research Audit.
2. Launch v1.0 dashboard, switch to **CSV snapshot (offline)**.
3. Use the same parameters recorded in v1.0 manifest and run the study.
4. Download the v1.0 reproducibility ZIP (optionally exclude source data).
5. Compare `final_test_predictions.csv`, `validation_model_ranking.csv`, and `manifest.json` with the archived v1.0 run. **Do not** expect the different v0.3 and v1.0 protocols to produce matching predictions; they intentionally differ.

## Troubleshooting

- `No module named streamlit`: ensure the `.venv` is activated and `pip install -r requirements-dev.txt` succeeds.
- `FileNotFoundError: requirements.txt`: `cd` into the extracted project root before installing.
- `No OHLCV rows`: verify symbol, network and provider access, or switch to offline mode.
- `Not enough initial history`: choose an earlier date, longer snapshot or smaller training window within allowed limits.
- `Connection refused`: verify the app is still running in the foreground and port 8501 is not blocked.
