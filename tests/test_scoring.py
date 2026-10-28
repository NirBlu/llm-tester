import pytest
from llmtester.scoring import score
from conftest import FakeProvider

def test_exact():
    assert score("exact", " paris ", "Paris") == 1.0
    assert score("exact", "Paris!", "Paris") == 0.0

def test_contains_list():
    assert score("contains", "I feel glad", ["joyful", "glad"]) == 1.0

def test_regex():
    assert score("regex", "2026-10-28", r"^\d{4}-\d{2}-\d{2}$") == 1.0

def test_token_f1_and_rouge():
    assert score("token_f1", "the cat sat", "the cat sat") == 1.0
    assert 0 < score("rouge_l", "the cat sat on mat", "the cat is on the mat") < 1
    assert score("rouge_l", "", "x") == 0.0

def test_llm_judge():
    assert score("llm_judge", "x", None, prompt="q", judge=FakeProvider("j")) == 0.8
    with pytest.raises(ValueError):
        score("llm_judge", "x", None)

def test_unknown():
    with pytest.raises(ValueError):
        score("nope", "a", "b")
