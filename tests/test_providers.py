import json, httpx, pytest
from llmtester.providers import get_provider, OllamaProvider, XAIProvider, AnthropicProvider


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama():
    def h(req):
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "hi"}, "prompt_eval_count": 3, "eval_count": 4})
    c = OllamaProvider("llama3", base_url="http://ollama:11434", client=client(h)).complete("x")
    assert c.text == "hi" and c.completion_tokens == 4 and c.tokens_per_s > 0


def test_xai_openai_compat(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "k")
    def h(req):
        assert str(req.url) == "https://api.x.ai/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer k"
        assert json.loads(req.content)["model"] == "grok-3-mini"
        return httpx.Response(200, json={"choices": [{"message": {"content": "yo"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    p = get_provider("xai:grok-3-mini", client=client(h))
    assert isinstance(p, XAIProvider) and p.complete("x").text == "yo"


def test_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://local:1234/v1")
    def h(req):
        assert req.url.host == "local"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    assert get_provider("openai:m", client=client(h)).complete("x").text == "ok"


def test_anthropic():
    def h(req):
        assert req.headers["x-api-key"] == "a"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "claude"}],
                                         "usage": {"input_tokens": 2, "output_tokens": 3}})
    c = AnthropicProvider("claude-x", api_key="a", client=client(h)).complete("x", system="s")
    assert c.text == "claude" and c.prompt_tokens == 2


def test_bad_spec():
    with pytest.raises(ValueError): get_provider("llama3")
    with pytest.raises(ValueError): get_provider("foo:bar")
