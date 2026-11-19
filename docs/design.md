# AI Gateway: Design Document

## 1. Goal

An OpenAI-compatible gateway that sits between tenants (apps or teams) and an LLM
provider. It authenticates each tenant, enforces per-tenant policy, and serves
repeated or semantically similar requests from a semantic cache instead of calling
the LLM again, to cut latency and cost.

Non-goals (for now): multi-turn caching, tool-call caching, billing, a UI.

## 2. Architecture

```
Client --> FastAPI gateway --> [auth -> rate limit -> policy -> cache lookup]
                                      |                              |
                                      | HIT                          | MISS
                                      v                              v
                                serve cached answer            LLM provider
                                                                     |
                                                          store answer in cache
```

Components:

- **FastAPI**: the `/v1/chat/completions` endpoint, streaming and non-streaming.
- **Redis Stack (Docker)**: semantic cache (vector index) and rate-limit counters.
- **RedisVL `SemanticCache`**: embeddings plus vector search. The default
  embedding model is `redis/langcache-embed-v1`, run locally.
- **Cross-encoder verifier**: `cross-encoder/quora-roberta-base`, run locally,
  used only on borderline matches (section 4.6).
- **SQLite**: tenants (hashed API keys, JSON config) and a per-request log.
- **Provider module**: a mock LLM (fixed delay) or any OpenAI-compatible endpoint,
  with retry and optional fallback models (section 4.11).

## 3. Request flow

1. Authenticate: the tenant is derived from the bearer API key.
2. Check the rate limit, then the model allow-list.
3. Decide whether the request is cacheable (section 4.5).
4. If cacheable, search the cache scoped to this tenant and scope hash, using the
   two-zone rule in section 4.6.
5. HIT: return the stored answer (as JSON, or replayed as a stream).
6. MISS or BYPASS: call the provider. On a MISS, store the answer.
7. Log tenant, model, HIT/MISS/BYPASS, latency, and token counts.

Responses carry `X-Cache: HIT | MISS | BYPASS` so behaviour is visible to clients.

## 4. Decisions

### 4.1 Tenant identity comes from the API key, never the request body

The request body is untrusted. If tenant identity came from it, any caller could
read another tenant's cached answers. The gateway looks up the tenant from the
key hash and injects `tenant_id` into every cache search and store itself.

### 4.2 API keys are stored only as SHA-256 hashes

Keys are 256-bit random values, so a fast unsalted hash is adequate (unlike
user-chosen passwords). The raw key is shown once at creation. A database leak
does not expose usable keys. Known gap: no key rotation yet.

### 4.3 Isolation is a metadata filter on one shared index

Each entry is tagged with `tenant_id` and `scope_hash`, and every lookup filters
on both. This is simple and efficient, but isolation depends on the filter being
applied on every path, so it is covered by tests (`tests/test_isolation.py`):
same-tenant paraphrase hits, other-tenant misses, different-system-prompt misses.
Alternative considered: one index per tenant (stronger physical separation,
higher operational cost as tenants grow).

### 4.4 Cache scope includes model and system prompt

`scope_hash = sha256(model + system prompt)`. Without it, two apps of the same
tenant with different system prompts, or different models, would share answers
that are wrong for each other.

### 4.5 Only safe requests are cached

A request is cached only if all of these hold:

- the tenant has caching enabled;
- it is single-turn (one user message, no assistant turns);
- it has no tools or functions;
- temperature is 0.7 or lower.

Reasons: in a multi-turn chat, a message like "what about the second one?" is
meaningless without context and would match unrelated conversations. Tool calls
depend on external state. High-temperature requests expect varied output.
Everything else is marked BYPASS. Trade-off: lower hit rate in exchange for
fewer wrong answers.

### 4.6 Two-zone matching: distance first, then a verifier

Distances are RedisVL cosine **distances** (0 to 2, lower is stricter). This is
the opposite direction of LangCache's similarity score, so any comparison between
backends must convert scales. Lookups fetch up to 3 nearest candidates for the
tenant and scope, then apply:

| Zone | Rule |
|---|---|
| distance <= `auto_accept` (0.08) | serve the nearest candidate |
| `auto_accept` < distance <= `verify_max` (0.30) | score each candidate with the verifier (duplicate probability, 0 to 1); serve the best only if its score is at least `verify_min_score` (0.8) |
| distance > 0.30 | miss |

**Why.** Distance alone could not separate paraphrases from one-word intent
changes (section 7). The history of the default: it started at 0.15, chosen
without data; a 40-pair evaluation led to 0.10; a 100-pair evaluation showed
0.10 sat right at the edge of a near-miss (distance 0.1005) and led to the
two-zone rule, with the auto-accept zone set to 0.08 for margin.

**Tenant override.** A tenant's `distance_threshold` overrides only the
auto-accept zone. The verification zone and verifier cutoff are global.

**Cost.** A borderline lookup adds one verifier call (up to 3 candidates scored
in one batch). Its latency has not been measured yet. The model also adds memory
(hundreds of MB) and startup time.

### 4.7 Cache backend is behind a small interface

The gateway calls `lookup` and `store` on a backend class. RedisVL is the first
implementation. A managed LangCache backend can be added without changing the
gateway. RedisVL calls are synchronous, so they run in worker threads
(`asyncio.to_thread`) to avoid blocking the event loop.

### 4.8 Rate limiting: fixed-window counter in Redis

`INCR` on `rl:{tenant}:{minute}` with a short expiry. Simple, fast, and
per-tenant configurable. Weakness: a burst across a window boundary can briefly
reach about 2x the limit. A sliding window or token bucket would fix this.

### 4.9 Streaming

- On a HIT, the cached answer is replayed as OpenAI-style chunks.
- On a MISS, chunks are passed through to the client as they arrive while the
  text is collected; the answer is stored only if the provider finished cleanly
  (`finish_reason: stop`). Disconnects, provider errors, or truncated answers
  are never cached.
- The first provider chunk is fetched before the response begins, so a provider
  failure returns a real HTTP 502 instead of a broken 200 stream.
- The streaming path does not retry or fall back (section 4.11 covers
  non-streaming requests only).

### 4.10 Observability by request log

Every request is logged to SQLite with its cache status and latency. `stats.py`
derives per-tenant hit rate and average hit/miss latency from it. Prometheus and
Grafana are planned (section 8).

### 4.11 Provider retries, fallback and error handling

For non-streaming requests, transient failures (HTTP 429, 5xx, network errors)
are retried up to 3 times with 1 s and 2 s waits. If the requested model still
fails, each model in `LLM_FALLBACK_MODELS` is tried in order. Permanent errors
(such as 400 or 404) are raised immediately, since retrying or switching models
would not fix a bad request. Upstream failures reach the client as HTTP 502 with
the provider's message, not as an opaque 500.

This design came from real failures during testing: a retired model returned 404
and Google's service returned 503.

Known issue: an answer produced by a fallback model is cached under the scope of
the model the client asked for, so a later request for the primary model can be
served an answer written by the fallback. Options are to skip caching fallback
answers or to cache them under the model that produced them.

## 5. Security notes

- Tenant IDs are restricted to `[a-z0-9_]` (safe in tag filters).
- Cache poisoning is limited to the tenant's own scope, since entries are never
  shared across tenants.
- Cached responses may contain user data; entries are tenant-scoped, but there
  is no PII detection yet (see limitations).
- Provider keys are read from environment variables and never stored in files or
  the repository.

## 6. Known limitations

- Multi-turn conversations are not cached (see 4.5).
- The verifier fails on some one-word intent changes, for example "cancel" vs
  "pause" my subscription (score 0.93). This is recorded as an expected-failure
  test. No cutoff fixes it without discarding most valid paraphrases.
- The verifier is conservative on short support-style questions, probably
  because it was trained on Quora questions; about half of the evaluation
  paraphrases score below 0.5.
- No cache TTL or invalidation endpoint yet, so stale answers can persist.
- No PII detection; sensitive prompts are cached like any other.
- Token counts are zero for streamed responses unless the provider reports usage.
- No cost estimation or dollars-saved metric yet.
- Client disconnects mid-stream do not cache the partial answer.
- Streaming requests do not retry or fall back.
- Fallback-model answers are cached under the requested model's scope (4.11).
- SQLite is a single-writer store; fine for a demo, not for high concurrency.
- Tested against a mock provider and a small number of manual runs against one
  real provider. No load test.

## 7. Evaluation

**Method.** `eval/run_eval_v2.py` embeds labelled prompt pairs with the same model
the cache uses, computes the cosine distance for each pair, scores each pair with
the verifier, and compares decision rules. Dataset: 100 hand-written pairs
(`eval/pairs.csv`, `eval/pairs_extra.csv`), 50 paraphrases that should match and
50 near-misses that should not. Per-pair scores are in `eval/pair_scores.csv`;
the plot is `eval/verifier_curves.png`. An earlier 40-pair run
(`eval/results.csv`, `eval/threshold_curve.png`) motivated the first change of
default and is superseded by this one.

**Results.** AUC: 0.806 for distance alone, 0.870 for the verifier.

| Rule | Paraphrases served (of 50) | Wrong answers served (of 50) |
|---|---|---|
| Distance <= 0.10 | 10 | 0 |
| Distance <= 0.15 | 25 | 4 |
| Distance <= 0.08, or <= 0.30 and verifier >= 0.8 (used) | 19 | 1 |
| Distance <= 0.10, or <= 0.30 and verifier >= 0.7 | 21 | 1 |
| Distance <= 0.10, or <= 0.30 and verifier >= 0.5 | 26 | 2 |

For comparison, distance alone needs about 3 to 5 wrong answers to reach 18 to 29
paraphrase hits. Zero wrong answers out of 50 has a 95% upper bound of about 6%
on the true rate.

**Findings.**

- Distance alone cannot cleanly separate paraphrases from one-word intent
  changes. Near-misses such as "last date to pay the semester fee" vs "last date
  to register for the semester" (0.1005) and "what time does the library open"
  vs "close" (0.111) sit at the same distance as real paraphrases.
- The verifier rejects both of those (scores 0.02 and 0.07) but accepts
  "cancel my subscription" vs "pause my subscription" (distance 0.118, score
  0.93) and scores "contact customer support" vs "contact the sales team" at 0.68.
- The verifier is conservative: 25 of 50 paraphrases score at least 0.5.

**Limitations of this evaluation.**

- The cutoffs were chosen on the same 100 pairs they are reported on, so the
  numbers are optimistic. A held-out set has not been run yet.
- Differences of one or two pairs are within noise at this size.
- The pairs are hand-written, and the near-misses are adversarial by design, so
  the wrong-answer rate likely overstates the risk on typical traffic. This is an
  untested hypothesis.
- It evaluates pairs, not nearest-neighbour retrieval over a full cache, where a
  query can match the wrong stored entry.
- The 50/50 class balance is artificial; hit and wrong-answer rates are more
  portable than precision.

**Latency.** In one manual run against a real provider, a miss took about 3.3 s
and a hit about 0.14 s; another miss in the same session took 17.9 s (provider
latency). With the mock provider (1 s delay), misses took 1.2 to 1.5 s and hits
about 0.14 s. These are single observations. The verifier's added latency and a
proper load test are still to do.

## 8. Roadmap

1. Held-out evaluation set (written without tuning), scored once with the final
   settings.
2. Measure verifier latency and run a load test, cache on vs off.
3. Retry and fallback for streaming; fix caching under the wrong model scope.
4. Prometheus metrics and a Grafana dashboard.
5. Admin API: create tenants, rotate keys, usage, per-tenant cache invalidation
   (invalidation must always be scoped to a tenant).
6. TTL policy and bypass rules for time-sensitive prompts.
7. Optional: LangCache backend and a backend-vs-backend comparison.
