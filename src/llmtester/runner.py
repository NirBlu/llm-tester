from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from .providers import get_provider, Provider
from .scoring import score
from .suite import Suite


def run_suite(suite: Suite, models: list[str], provider_factory=get_provider, judge_model: str | None = None) -> dict:
    judge_spec = judge_model or suite.judge_model
    judge: Provider | None = provider_factory(judge_spec) if judge_spec else None
    results = []
    for spec in models:
        prov = provider_factory(spec)
        for case in suite.cases:
            scorer = case.scorer or suite.scorer
            row = {"model": spec, "case_id": case.id, "scorer": scorer, "score": 0.0, "latency_s": None,
                   "prompt_tokens": None, "completion_tokens": None, "tokens_per_s": None, "output": "", "error": None}
            try:
                c = prov.complete(case.prompt, system=case.system or suite.system, temperature=suite.temperature)
                row.update(output=c.text, latency_s=round(c.latency_s, 4), prompt_tokens=c.prompt_tokens,
                           completion_tokens=c.completion_tokens,
                           tokens_per_s=round(c.tokens_per_s, 2) if c.tokens_per_s else None)
                row["score"] = round(score(scorer, c.text, case.expected, prompt=case.prompt,
                                           rubric=case.rubric, judge=judge), 4)
            except Exception as e:  # keep going; record the error
                row["error"] = f"{type(e).__name__}: {e}"
            results.append(row)
    return {"suite": suite.name, "models": models, "started_at": datetime.now(timezone.utc).isoformat(),
            "results": results, "summary": summarize(results, models)}


def summarize(results: list[dict], models: list[str]) -> list[dict]:
    out = []
    for m in models:
        rs = [r for r in results if r["model"] == m]
        ok = [r for r in rs if not r["error"]]
        lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        tps = [r["tokens_per_s"] for r in ok if r["tokens_per_s"]]
        out.append({"model": m, "cases": len(rs), "errors": len(rs) - len(ok),
                    "avg_score": round(mean(r["score"] for r in rs), 4) if rs else 0.0,
                    "avg_latency_s": round(mean(lat), 3) if lat else None,
                    "total_tokens": sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in ok),
                    "avg_tokens_per_s": round(mean(tps), 2) if tps else None})
    return out


def write_outputs(run: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = run["started_at"].replace(":", "").replace("-", "")[:15]
    base = out / f"{run['suite']}_{stamp}"
    jp, cp = base.with_suffix(".json"), base.with_suffix(".csv")
    jp.write_text(json.dumps(run, indent=2))
    with cp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run["results"][0].keys()) if run["results"] else ["model"])
        w.writeheader(); w.writerows(run["results"])
    return jp, cp


def format_table(summary: list[dict]) -> str:
    cols = ["model", "avg_score", "avg_latency_s", "avg_tokens_per_s", "total_tokens", "errors", "cases"]
    rows = [[("" if s[c] is None else str(s[c])) for c in cols] for s in summary]
    w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda r: " | ".join(v.ljust(w[i]) for i, v in enumerate(r))
    return "\n".join([line(cols), "-+-".join("-" * x for x in w), *map(line, rows)])
