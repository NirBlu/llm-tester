"""Scorers return a float in [0, 1]."""
from __future__ import annotations
import re

def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())

def _alts(expected) -> list[str]:
    if expected is None:
        return []
    return [str(e) for e in expected] if isinstance(expected, list) else [str(expected)]

def exact(output, expected, **_):
    return float(any(_norm(output) == _norm(e) for e in _alts(expected)))

def contains(output, expected, **_):
    return float(any(_norm(e) in _norm(output) for e in _alts(expected)))

def regex(output, expected, **_):
    return float(any(re.search(e, output, re.I | re.S) for e in _alts(expected)))

def _tokens(s):
    return re.findall(r"\w+", s.lower())

def token_f1(output, expected, **_):
    best = 0.0
    o = _tokens(output)
    for e in _alts(expected):
        r = _tokens(e)
        if not o or not r:
            continue
        common = sum(min(o.count(t), r.count(t)) for t in set(r))
        if common:
            p, rc = common / len(o), common / len(r)
            best = max(best, 2 * p * rc / (p + rc))
    return best

def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_l(output, expected, **_):
    best = 0.0
    o = _tokens(output)
    for e in _alts(expected):
        r = _tokens(e)
        if not o or not r:
            continue
        l = _lcs(o, r)
        if l:
            p, rc = l / len(o), l / len(r)
            best = max(best, 2 * p * rc / (p + rc))
    return best

JUDGE_PROMPT = """You are grading a model answer. Score it from 0 to 10.
Question:
{prompt}

Reference answer (may be empty):
{expected}

Rubric:
{rubric}

Model answer:
{output}

Reply with only: SCORE: <number 0-10>"""

def llm_judge(output, expected, prompt="", rubric=None, judge=None, **_):
    if judge is None:
        raise ValueError("llm_judge scorer requires judge_model")
    c = judge.complete(JUDGE_PROMPT.format(prompt=prompt, expected="; ".join(_alts(expected)),
                                           rubric=rubric or "Correctness and helpfulness.", output=output))
    m = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", c.text) or re.search(r"([0-9]+(?:\.[0-9]+)?)", c.text)
    return max(0.0, min(1.0, float(m.group(1)) / 10)) if m else 0.0

SCORERS = {"exact": exact, "contains": contains, "regex": regex, "token_f1": token_f1,
           "rouge_l": rouge_l, "llm_judge": llm_judge}

def score(name, output, expected, **kw) -> float:
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}; choose from {sorted(SCORERS)}")
    return SCORERS[name](output, expected, **kw)
