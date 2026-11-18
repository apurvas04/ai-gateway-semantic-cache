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
- **SQLite**: tenants (hashed API keys, JSON config) and a per-request log.
- **Provider module**: a mock LLM (fixed delay) or any OpenAI-compatible endpoint.

## 3. Request flow

1. Authenticate: the tenant is derived from the bearer API key.
2. Check the rate limit, then the model allow-list.
3. Decide whether the request is cacheable (section 4.5).
4. If cacheable, search the cache scoped to this tenant and scope hash.
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

### 4.6 Similarity threshold is configurable per tenant

Uses RedisVL's cosine **distance** (0 to 2, lower is stricter). Different tenants
tolerate different risk: a legal-FAQ bot wants strict matching, a casual
assistant can accept looser matches. This is the opposite direction of
LangCache's similarity score, so any comparison between backends must convert
scales.

**Default: 0.10.** The initial default was 0.15, chosen without data. The
evaluation in section 7 showed 0.15 was too loose (about 1 in 7 hits wrong on an
adversarial test set), so the default was lowered to 0.10, the strictest setting
that still caught some paraphrases and produced no false hits on that set.
Tenants can raise it with `distance_threshold` in their config.

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

### 4.10 Observability by request log

Every request is logged to SQLite with its cache status and latency. `stats.py`
derives per-tenant hit rate and average hit/miss latency from it. Prometheus and
Grafana are planned (section 8).

## 5. Security notes

- Tenant IDs are restricted to `[a-z0-9_]` (safe in tag filters).
- Cache poisoning is limited to the tenant's own scope, since entries are never
  shared across tenants.
- Cached responses may contain user data; entries are tenant-scoped, but there
  is no PII detection yet (see limitations).

## 6. Known limitations

- Multi-turn conversations are not cached (see 4.5).
- A distance threshold alone cannot reliably separate paraphrases from
  one-word intent changes (see section 7).
- No cache TTL or invalidation yet, so stale answers can persist.
- No PII detection; sensitive prompts are cached like any other.
- Token counts are zero for streamed responses unless the provider reports usage.
- No cost estimation or dollars-saved metric yet.
- Client disconnects mid-stream do not cache the partial answer.
- SQLite is a single-writer store; fine for a demo, not for high concurrency.
- Tested so far against a mock provider only.

## 7. Evaluation

**Method.** `eval/run_eval.py` embeds labelled prompt pairs with the same model
the cache uses, computes the cosine distance for each pair, and sweeps the
threshold. Dataset (`eval/pairs.csv`): 20 paraphrase pairs that should match and
20 near-miss pairs that should not, written by hand. Each near-miss differs by a
meaning-changing word.

**Results.**

| Distance threshold | Paraphrase hit rate | False-hit rate | Precision |
|---|---|---|---|
| 0.10 | 15% | 0% | 100% |
| 0.12 | 25% | 10% | 71% |
| 0.16 | 45% | 15% | 75% |
| 0.20 | 55% | 25% | 69% |
| 0.30 | 75% | 60% | 56% |

(Full sweep in `eval/results.csv`; plot in `eval/threshold_curve.png`.)

**Findings.**

- The distance distributions of paraphrases and near-misses overlap heavily. No
  threshold gives a high hit rate with a low false-hit rate on this set.
- Failure mode: one-word differences that change intent sit at distance
  0.10 to 0.18, the same range where real paraphrases sit. Closest near-misses:
  - 0.100: "last date to **pay** the semester fee" vs "last date to **register**
    for the semester"
  - 0.118: "**cancel** my subscription" vs "**pause** my subscription"
  - 0.152: "**boiling** point of water" vs "**freezing** point of water"
- The initial default of 0.15 would have served a wrong answer for the first two
  pairs, so the default was lowered to 0.10.
- 0 false hits out of 20 near-misses at 0.10 is not a guarantee: the 95% upper
  bound on the true false-hit rate is about 14%.

**Limitations of this evaluation.**

- Small, hand-written dataset (40 pairs); treat results as a prototype.
- Near-misses are adversarial by design, so the false-hit rate likely
  overstates the risk on typical traffic. This is an untested hypothesis.
- It evaluates pairs, not nearest-neighbour retrieval over a full cache, where a
  query can match the wrong stored entry.
- Class balance is 50/50 by construction, so precision would differ on real
  traffic; the hit rate and false-hit rate are the more portable numbers.

**Still to do:** load test (cache on vs off) for latency and cost.

## 8. Roadmap

1. Grow the evaluation set to 100+ pairs, including a public duplicate-question
   dataset, and add an AUC summary.
2. Verification stage for borderline hits (for example a cross-encoder), then
   re-measure hit rate vs false-hit rate.
3. Real provider integration and fallback to a second provider.
4. Prometheus metrics and a Grafana dashboard.
5. Load test, cache on vs off.
6. Admin API: create tenants, rotate keys, usage, per-tenant cache invalidation
   (invalidation must always be scoped to a tenant).
7. TTL policy and bypass rules for time-sensitive prompts.
8. Optional: LangCache backend and a backend-vs-backend comparison.
