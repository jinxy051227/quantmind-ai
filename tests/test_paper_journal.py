import json
import pytest
from datetime import datetime,timezone
from pathlib import Path
from scripts.log_paper_forecast import append_record

def test_dedup_paper_log(tmp_path):
    p=tmp_path/"log.jsonl"
    r={"ticker":"AAPL","last_available_price_date":"2026-11-10","up_probability":0.5}
    append_record(p,r)
    assert json.loads(p.read_text().strip())==r
    with pytest.raises(ValueError):append_record(p,r)
