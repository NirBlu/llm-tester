"""Provider adapters. Model spec format: `provider:model`, e.g. `ollama:llama3`."""
from __future__ import annotations
import os, time
from dataclasses import dataclass
import httpx


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def tokens_per_s(self) -> float | None:
        if self.completion_tokens and self.latency_s > 0:
            return self.completion_tokens / self.latency_s
        return None


class Provider:
    name = "base"

    def __init__(self, model: str, client: httpx.Client | None = None, timeout: float = 120):
        self.model = model
        self.client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> Completion:
        raise NotImplementedError

    def _post(self, url, json, headers=None):
        r = self.client.post(url, json=json, headers=headers or {})
        r.raise_for_status()
        return r.json()


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model, base_url=None, **kw):
        super().__init__(model, **kw)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def complete(self, prompt, system=None, temperature=0.0):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t = time.perf_counter()
        d = self._post(f"{self.base_url}/api/chat", {"model": self.model, "messages": msgs, "stream": False,
                                                       "options": {"temperature": temperature}})
        lat = time.perf_counter() - t
        return Completion(d["message"]["content"], lat, d.get("prompt_eval_count"), d.get("eval_count"))


class OpenAICompatProvider(Provider):
    name = "openai"
    default_base = "https://api.openai.com/v1"
    key_env, base_env = "OPENAI_API_KEY", "OPENAI_BASE_URL"

    def __init__(self, model, base_url=None, api_key=None, **kw):
        super().__init__(model, **kw)
        self.base_url = (base_url or os.getenv(self.base_env) or self.default_base).rstrip("/")
        self.api_key = api_key or os.getenv(self.key_env, "")

    def complete(self, prompt, system=None, temperature=0.0):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t = time.perf_counter()
        d = self._post(f"{self.base_url}/chat/completions",
                       {"model": self.model, "messages": msgs, "temperature": temperature},
                       {"Authorization": f"Bearer {self.api_key}"})
        lat = time.perf_counter() - t
        u = d.get("usage") or {}
        return Completion(d["choices"][0]["message"]["content"] or "", lat,
                          u.get("prompt_tokens"), u.get("completion_tokens"))


class XAIProvider(OpenAICompatProvider):
    name = "xai"
    default_base = "https://api.x.ai/v1"
    key_env, base_env = "XAI_API_KEY", "XAI_BASE_URL"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model, api_key=None, base_url=None, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")

    def complete(self, prompt, system=None, temperature=0.0):
        body = {"model": self.model, "max_tokens": 1024, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        t = time.perf_counter()
        d = self._post(f"{self.base_url}/v1/messages", body,
                       {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        lat = time.perf_counter() - t
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        return Completion(text, lat, u.get("input_tokens"), u.get("output_tokens"))


PROVIDERS: dict[str, type[Provider]] = {
    "ollama": OllamaProvider, "openai": OpenAICompatProvider, "xai": XAIProvider,
    "grok": XAIProvider, "anthropic": AnthropicProvider,
}


def get_provider(spec: str, **kw) -> Provider:
    if ":" not in spec:
        raise ValueError(f"Model spec must be provider:model, got {spec!r}")
    prov, model = spec.split(":", 1)
    if prov not in PROVIDERS:
        raise ValueError(f"Unknown provider {prov!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[prov](model, **kw)
