from __future__ import annotations
import argparse, os, sys
from .suite import load_suite
from .runner import run_suite, write_outputs, format_table
from . import db


def main(argv=None, provider_factory=None):
    p = argparse.ArgumentParser(prog="llmtester")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite")
    r.add_argument("--suite", required=True)
    r.add_argument("--models", required=True, help="comma-separated provider:model list")
    r.add_argument("--out", default="results")
    r.add_argument("--db", default=os.getenv("LLMTESTER_DB", "results/llmtester.db"))
    r.add_argument("--judge-model", default=None)
    s = sub.add_parser("serve", help="start web UI")
    s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("llmtester.web.app:app", host=a.host, port=a.port)
        return 0
    suite = load_suite(a.suite)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    kw = {"provider_factory": provider_factory} if provider_factory else {}
    run = run_suite(suite, models, judge_model=a.judge_model, **kw)
    jp, cp = write_outputs(run, a.out)
    rid = db.save_run(db.connect(a.db), run)
    print(format_table(run["summary"]))
    print(f"\nrun #{rid} saved -> {jp}, {cp}, {a.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
