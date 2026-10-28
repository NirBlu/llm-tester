# llm-tester

Home-runnable LLM evaluation: run YAML benchmark suites against local (Ollama) and remote
(OpenAI-compatible, xAI Grok, Anthropic) models, score answers, and compare latency, tokens,
tokens/sec and score — from a CLI or a small web UI. Results go to JSON, CSV and SQLite.

## Quickstart (local)
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env && set -a && . ./.env && set +a   # add API keys
llmtester run --suite suites/sample.yaml --models ollama:llama3,openai:gpt-4o-mini,xai:grok-3-mini \
  --judge-model openai:gpt-4o-mini
llmtester serve            # web UI on http://localhost:8000
pytest
```

## Quickstart (Docker)
```bash
cp .env.example .env
docker compose up --build                     # web UI, uses Ollama on the host
docker compose --profile ollama up --build    # also start a bundled Ollama
# (bundled: set OLLAMA_BASE_URL_DOCKER=http://ollama:11434 and `docker compose exec ollama ollama pull llama3`)
docker compose run --rm web llmtester run --suite suites/sample.yaml --models ollama:llama3
```

## Models
`provider:model` — providers: `ollama`, `openai` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`),
`xai` / `grok` (https://api.x.ai/v1, `XAI_API_KEY`), `anthropic` (`ANTHROPIC_API_KEY`).

## Suite format
```yaml
name: my-suite
scorer: contains            # default: exact | contains | regex | token_f1 | rouge_l | llm_judge
system: Answer concisely.
judge_model: openai:gpt-4o-mini   # for llm_judge
cases:
  - id: q1
    prompt: Capital of France?
    expected: Paris           # string or list of acceptable answers
    scorer: exact
  - id: q2
    prompt: Write a haiku.
    scorer: llm_judge
    rubric: 5-7-5, about autumn.
```

## Output
- `results/<suite>_<timestamp>.json|.csv` — per-case rows (score, latency_s, tokens, tokens_per_s, output, error)
- `results/llmtester.db` — SQLite `runs` / `results` tables (shared by CLI and web UI)
- Console comparison table: avg score, avg latency, avg tokens/sec, total tokens, errors.

## Web API
`GET /api/suites`, `GET /api/runs`, `GET /api/runs/{id}`, `POST /api/runs {suite_file|suite_yaml, models, judge_model}`.

License: MIT
