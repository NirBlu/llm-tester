from __future__ import annotations
import json, sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, suite TEXT, models TEXT,
  started_at TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, model TEXT,
  case_id TEXT, scorer TEXT, score REAL, latency_s REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
  tokens_per_s REAL, output TEXT, error TEXT, FOREIGN KEY(run_id) REFERENCES runs(id));
"""

def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c

def save_run(conn, run: dict) -> int:
    cur = conn.execute("INSERT INTO runs(suite, models, started_at, summary) VALUES(?,?,?,?)",
                       (run["suite"], json.dumps(run["models"]), run["started_at"], json.dumps(run["summary"])))
    rid = cur.lastrowid
    conn.executemany(
        "INSERT INTO results(run_id, model, case_id, scorer, score, latency_s, prompt_tokens, completion_tokens,"
        " tokens_per_s, output, error) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        [(rid, r["model"], r["case_id"], r["scorer"], r["score"], r["latency_s"], r["prompt_tokens"],
          r["completion_tokens"], r["tokens_per_s"], r["output"], r["error"]) for r in run["results"]])
    conn.commit()
    return rid

def list_runs(conn):
    return [dict(r, models=json.loads(r["models"]), summary=json.loads(r["summary"]))
            for r in conn.execute("SELECT * FROM runs ORDER BY id DESC")]

def get_run(conn, run_id: int):
    r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    d = dict(r, models=json.loads(r["models"]), summary=json.loads(r["summary"]))
    d["results"] = [dict(x) for x in conn.execute("SELECT * FROM results WHERE run_id=? ORDER BY id", (run_id,))]
    return d
