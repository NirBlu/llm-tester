import json
from pathlib import Path
from llmtester.cli import main
from llmtester.suite import load_suite
from llmtester.runner import run_suite
from llmtester import db

SUITE = Path(__file__).parent.parent / "suites" / "sample.yaml"


def test_run_suite(factory):
    r = run_suite(load_suite(SUITE), ["fake:good", "fake:bad", "fake:broken"], provider_factory=factory,
                  judge_model="fake:judge")
    s = {x["model"]: x for x in r["summary"]}
    assert s["fake:good"]["avg_score"] > s["fake:bad"]["avg_score"]
    assert s["fake:broken"]["errors"] == len(load_suite(SUITE).cases)
    assert s["fake:good"]["avg_tokens_per_s"] > 0


def test_cli_and_db(tmp_path, factory, capsys):
    dbp = tmp_path / "t.db"
    main(["run", "--suite", str(SUITE), "--models", "fake:good,fake:bad", "--out", str(tmp_path),
          "--db", str(dbp), "--judge-model", "fake:j"], provider_factory=factory)
    out = capsys.readouterr().out
    assert "avg_score" in out and "fake:good" in out
    assert list(tmp_path.glob("*.json")) and list(tmp_path.glob("*.csv"))
    c = db.connect(dbp)
    runs = db.list_runs(c)
    assert len(runs) == 1 and len(db.get_run(c, runs[0]["id"])["results"]) == 12
