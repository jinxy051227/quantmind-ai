"""Read-only bounded demo API. No execution, trading or portfolio credentials."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantmind import __version__
from quantmind.core import demo_prices
from quantmind.v1 import StudyConfig, run_study, json_safe

app = FastAPI(title="QuantMind AI Research API",version=__version__,docs_url="/docs")

class DemoRequest(BaseModel):
    seed: int = Field(default=42,ge=0,le=100000)
    rows: int = Field(default=1100,ge=600,le=2200)
    retrain_every: int = Field(default=120,ge=20,le=240)

@app.get("/health")
def health():
    return {"status":"ok","version":__version__,"mode":"research-demo-only"}

@app.post("/v1/demo/study")
def run_demo(body: DemoRequest):
    s = run_study(demo_prices(n=body.rows,seed=body.seed),
                  StudyConfig(retrain_every=body.retrain_every,min_train_rows=160,
                              train_window=504,bootstrap_repeats=0),
                  candidates=("Logistic Regression",),source="synthetic-api",ticker="DEMO")
    return json_safe({"warning":"Synthetic retrospective demo only; not financial advice",
            "ticker":s["ticker"],"champion":s["champion"],
            "split":s["split"],"test_metrics":s["test_metrics"],
            "policy_scorecard":s["scorecard"].reset_index().to_dict(orient="records")})
