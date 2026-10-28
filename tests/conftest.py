import pytest
from llmtester.providers import Completion, Provider


class FakeProvider(Provider):
    def __init__(self, model, answers=None):
        self.model = model
        self.answers = answers or {}

    def complete(self, prompt, system=None, temperature=0.0):
        if self.model == "broken":
            raise RuntimeError("boom")
        for k, v in self.answers.items():
            if k in prompt:
                return Completion(v, 0.5, 10, 20)
        if "grading" in prompt:
            return Completion("SCORE: 8", 0.1, 5, 3)
        return Completion("I don't know", 0.5, 10, 5)


ANSWERS = {"capital": "Paris", "17 * 3": "The answer is 51.", "haiku": "leaves fall"}


@pytest.fixture
def factory():
    def f(spec):
        _, model = spec.split(":", 1)
        return FakeProvider(model, ANSWERS if model == "good" else {})
    return f
