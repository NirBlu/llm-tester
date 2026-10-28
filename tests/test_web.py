import importlib
from pathlib import Path
from fastapi.testclient import TestClient


def test_web(tmp_path, monkeypatch, factory):
    monkeypatch.setenv("LLMTESTER_DB", str(tmp_path / "w.db"))
    monkeypatch.setenv("LLMTESTER_RESULTS", str(tmp_path))
    monkeypatch.setenv("LLMTESTER_SUITES", str(Path(__file__).parent.parent / "suites"))
    import llmtester.web.app as m
    m = importlib.reload(m)
    m.app.state.provider_factory = factory
    c = TestClient(m.app)
    assert "llmtester" in c.get("/").text
    assert "sample.yaml" in c.get("/api/suites").json()
    r = c.post("/api/runs", json={"suite_file": "sample.yaml", "models": ["fake:good"], "judge_model": "fake:j"})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert c.get("/api/runs").json()[0]["id"] == rid
    assert len(c.get(f"/api/runs/{rid}").json()["results"]) == 6
    y = "name: inline\ncases:\n  - prompt: capital?\n    expected: Paris\n"
    assert c.post("/api/runs", json={"suite_yaml": y, "models": ["fake:good"]}).json()["summary"][0]["avg_score"] == 1.0
    assert c.get("/api/runs/999").status_code == 404
