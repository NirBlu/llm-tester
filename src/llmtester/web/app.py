"""Lightweight web UI: run suites, list runs, compare results."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yaml
from .. import db
from ..suite import parse_suite, load_suite
from ..runner import run_suite, write_outputs
from ..providers import get_provider

SUITES_DIR = Path(os.getenv("LLMTESTER_SUITES", "suites"))
RESULTS_DIR = Path(os.getenv("LLMTESTER_RESULTS", "results"))
app = FastAPI(title="llmtester")
app.state.provider_factory = get_provider
_conn = None


def conn():
    global _conn
    if _conn is None:
        _conn = db.connect(os.getenv("LLMTESTER_DB", str(RESULTS_DIR / "llmtester.db")))
    return _conn


class RunRequest(BaseModel):
    models: list[str]
    suite_file: str | None = None
    suite_yaml: str | None = None
    judge_model: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/suites")
def suites():
    return sorted(p.name for p in SUITES_DIR.glob("*.y*ml")) if SUITES_DIR.exists() else []


@app.get("/api/runs")
def runs():
    return db.list_runs(conn())


@app.get("/api/runs/{run_id}")
def run(run_id: int):
    r = db.get_run(conn(), run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r


@app.post("/api/runs")
def create_run(req: RunRequest):
    if req.suite_yaml:
        suite = parse_suite(yaml.safe_load(req.suite_yaml))
    elif req.suite_file:
        path = (SUITES_DIR / Path(req.suite_file).name)
        if not path.exists():
            raise HTTPException(404, "suite not found")
        suite = load_suite(path)
    else:
        raise HTTPException(400, "suite_file or suite_yaml required")
    r = run_suite(suite, req.models, provider_factory=app.state.provider_factory, judge_model=req.judge_model)
    write_outputs(r, RESULTS_DIR)
    rid = db.save_run(conn(), r)
    return {"id": rid, "summary": r["summary"]}
