# AI Gateway with a Verified Semantic Cache

An OpenAI-compatible gateway that sits in front of an LLM and gives each tenant its own API key, rate limit and isolated semantic cache. Repeated or reworded questions are answered from Redis in about 0.14 s instead of calling the model, and a second model checks borderline matches before they are served.

The most interesting part is not the plumbing but the measurement: an evaluation of how well embedding distance alone separates paraphrases from near-miss questions, and what a verifier adds. See [Evaluation](#evaluation).

## Results at a glance

- **Latency.** In one manual run against Gemini (through its OpenAI-compatible endpoint), a cache miss took about 3.3 s and a hit about 0.14 s. Another miss in the same session took 17.9 s because of provider latency. This is a single run, not a benchmark. With the built-in mock LLM (fixed 1 s delay), misses took about 1.2 to 1.5 s against about 0.14 s for hits.
- **Accuracy.** On 100 hand-written question pairs, distance alone could not separate paraphrases from one-word intent changes. Adding a verifier roughly doubled reuse at a similar error rate, but one failure remains that no cutoff fixes.

## How it works

```
Client --> FastAPI gateway --> [auth -> rate limit -> policy -> cache lookup]
                                      |                              |
                                      | HIT                          | MISS
                                      v                              v
                                serve cached answer            LLM provider
                                                                     |
                                                          store answer in cache
```

1. The tenant is identified from its bearer API key. Only a SHA-256 hash of the key is stored.
2. A per-tenant rate limit and model allow-list are applied.
3. Only safe requests are cached: single-turn, no tools, temperature at or below 0.7. Everything else is marked `BYPASS`.
4. The cache is searched with a filter on `tenant_id` and a hash of the model and system prompt, so tenants never see each other's answers.
5. Matching uses two zones: a close match is served directly, and a borderline match is served only if a verifier model agrees (see below).
6. On a miss the gateway calls the provider (with retry and optional fallback models), stores the answer, and logs the request.

Responses carry an `X-Cache: HIT | MISS | BYPASS` header.

## Features

- OpenAI-compatible `/v1/chat/completions`, streaming and non-streaming
- Multi-tenant isolation with tests for isolation paths
- Per-tenant config: rate limit, allowed models, cache on or off, match threshold
- Two-zone semantic matching with a cross-encoder verifier
- Provider retry with backoff, optional fallback models, and clear 502 errors
- Streamed cache hits replayed as chunks; streamed misses cached only if the stream finished cleanly
- Request log in SQLite and a `stats.py` summary of hit rate and latency per tenant
- Works with a mock LLM (no API key needed) or any OpenAI-compatible endpoint

## Evaluation

**Setup.** 100 hand-written question pairs (`eval/pairs.csv`, `eval/pairs_extra.csv`): 50 paraphrases that should match and 50 near-misses that should not (for example "cancel my subscription" vs "pause my subscription"). Embeddings use `redis/langcache-embed-v1`; the verifier is `cross-encoder/quora-roberta-base`. Reproduce with `python eval/run_eval_v2.py`.

![Verifier vs distance only](eval/verifier_curves.png)

| Rule | Paraphrases served (of 50) | Wrong answers served (of 50) |
|---|---|---|
| Distance ≤ 0.10 | 10 | 0 |
| Distance ≤ 0.15 | 25 | 4 |
| Distance ≤ 0.08, or ≤ 0.30 and verifier ≥ 0.8 (**used**) | 19 | 1 |

AUC: 0.806 for distance alone, 0.870 for the verifier.

**Findings**

- Distance alone cannot cleanly separate paraphrases from one-word intent changes. Near-misses such as "pay" vs "register" for the semester fee sit at the same distance as real paraphrases.
- The verifier fixes some of these (it rejects pay/register and library open/close) but not "cancel vs pause my subscription", which it scores at 0.93. This is recorded as a known-failure test (`xfail`) so it cannot be forgotten.
- The verifier is conservative: only half of the paraphrases score 0.5 or higher, probably because it was trained on Quora questions and these are short support-style questions.

**Limitations of the evaluation.** The cutoffs were chosen on the same 100 pairs they are reported on, so the results are optimistic. The pairs are hand-written, and the near-misses are adversarial by design, so real traffic would probably show fewer wrong answers. One pair sits at distance 0.1005, so a plain 0.10 cutoff only avoids it by a hair. A held-out set has not been run yet.

## Quickstart (Windows cmd; adapt for macOS or Linux)

Requires Python (developed on 3.12) and Docker.

```
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python create_tenant.py tenant_a "Tenant A"      # prints an API key once; save it
uvicorn app.main:app --reload
```

In a second window (venv active), using the key you saved:

```
set TEST_MODEL=test-model
python try_it.py gw_YOUR_KEY
```

The default `LLM_MODE=mock` needs no API key. To use a real provider, set these in the window that runs `uvicorn`, before starting it:

```
set LLM_MODE=real
set LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
set LLM_API_KEY=your-key
set LLM_FALLBACK_MODELS=optional,comma,separated,models
```

Then set `TEST_MODEL` to a model name your provider currently serves. Never commit keys.

**Tenant config** (stored per tenant, set with `python set_config.py <tenant> <key> <value>`): `cache_enabled`, `rate_limit_per_min`, `allowed_models`, and `distance_threshold` (overrides the auto-accept zone only).

**Tests** (need Redis running):

```
python -m pytest -v
```

## Project layout

```
app/          gateway code: main.py, cache.py, db.py, providers.py, ratelimit.py
tests/        isolation tests and verifier tests
eval/         evaluation data, script and results
docs/         design document with decisions and limitations
*.py (root)   helper scripts: create_tenant, set_config, stats, try_it, try_stream
```

## Design and limitations

The full list of design decisions and known limitations is in [docs/design.md](docs/design.md). The main ones: only single-turn requests are cached; there is no cache TTL or invalidation endpoint yet; no PII detection; streaming requests do not retry or fall back; answers from a fallback model are cached under the originally requested model; the verifier's latency has not been measured; and SQLite limits write concurrency. Tested against a mock provider and a small number of manual runs against one real provider, with no load test.
