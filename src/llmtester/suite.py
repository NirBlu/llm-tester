from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Case:
    id: str
    prompt: str
    expected: str | list[str] | None = None
    scorer: str | None = None
    rubric: str | None = None
    system: str | None = None


@dataclass
class Suite:
    name: str
    cases: list[Case]
    scorer: str = "contains"
    system: str | None = None
    judge_model: str | None = None
    temperature: float = 0.0
    description: str = ""


def parse_suite(data: dict) -> Suite:
    cases = []
    for i, c in enumerate(data.get("cases") or []):
        if "prompt" not in c:
            raise ValueError(f"case #{i} missing 'prompt'")
        cases.append(Case(id=str(c.get("id", i + 1)), prompt=c["prompt"], expected=c.get("expected"),
                          scorer=c.get("scorer"), rubric=c.get("rubric"), system=c.get("system")))
    if not cases:
        raise ValueError("suite has no cases")
    return Suite(name=data.get("name", "suite"), cases=cases, scorer=data.get("scorer", "contains"),
                 system=data.get("system"), judge_model=data.get("judge_model"),
                 temperature=float(data.get("temperature", 0.0)), description=data.get("description", ""))


def load_suite(path: str | Path) -> Suite:
    return parse_suite(yaml.safe_load(Path(path).read_text()))
