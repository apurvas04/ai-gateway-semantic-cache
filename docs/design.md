\# AI Gateway: Design Document



\## 1. Goal



An OpenAI-compatible gateway that sits between tenants (apps or teams) and an LLM

provider. It authenticates each tenant, enforces per-tenant policy, and serves

repeated or semantically similar requests from a semantic cache instead of calling

the LLM again, to cut latency and cost.



Non-goals (for now): multi-turn caching, tool-call caching, billing, a UI.



\## 2. Architecture



&#x20;   Client --> FastAPI gateway --> \[auth -> rate limit -> policy -> cache lookup]

&#x20;                                         |                              |

&#x20;                                         | HIT                          | MISS

&#x20;                                         v                              v

&#x20;                                   serve cached answer            LLM provider

&#x20;                                                                        |

&#x20;                                                             store answer in cache



Components:

\- \*\*FastAPI\*\*: the `/v1/chat/completions` endpoint, streaming and non-streaming.

\- \*\*Redis Stack (Docker)\*\*: semantic cache (vector index) and rate-limit counters.

\- \*\*RedisVL `SemanticCache`\*\*: embeddings plus vector search. The default

&#x20; embedding model is `redis/langcache-embed-v1`, run locally.

\- \*\*SQLite\*\*: tenants (hashed API keys, JSON config) and a per-request log.

\- \*\*Provider module\*\*: a mock LLM (fixed delay) or any OpenAI-compatible endpoint.



\## 3. Request flow



1\. Authenticate: the tenant is derived from the bearer API key.

2\. Check the rate limit, then the model allow-list.

3\. Decide whether the request is cacheable (section 5).

4\. If cacheable, search the cache scoped to this tenant and scope hash.

5\. HIT: return the stored answer (as JSON, or replayed as a stream).

6\. MISS or BYPASS: call the provider. On a MISS, store the answer.

7\. Log tenant, model, HIT/MISS/BYPASS, latency, and token counts.



Responses carry `X-Cache: HIT | MISS | BYPASS` so behaviour is visible to clients.



\## 4. Decisions



\### 4.1 Tenant identity comes from the API key, never the request body

The request body is untrusted. If tenant identity came from it, any caller could

read another tenant's cached answers. The gateway looks up the tenant from the

key hash and injects `tenant\_id` into every cache search and store itself.



\### 4.2 API keys are stored only as SHA-256 hashes

Keys are 256-bit random values, so a fast unsalted hash is adequate (unlike

user-chosen passwords). The raw key is shown once at creation. A database leak

does not expose usable keys. Known gap: no key rotation yet.



\### 4.3 Isolation is a metadata filter on one shared index

Each entry is tagged with `tenant\_id` and `scope\_hash`, and every lookup filters

on both. This is simple and efficient, but isolation depends on the filter being

applied on every path, so it is covered by tests (`tests/test\_isolation.py`):

same-tenant paraphrase hits, other-tenant misses, different-system-prompt misses.

Alternative considered: one index per tenant (stronger physical separation,

higher operational cost as tenants grow).



\### 4.4 Cache scope includes model and system prompt

`scope\_hash = sha256(model + system prompt)`. Without it, two apps of the same

tenant with different system prompts, or different models, would share answers

that are wrong for each other.



\### 4.5 Only safe requests are cached

A request is cached only if all of these hold:

\- the tenant has caching enabled;

\- it is single-turn (one user message, no assistant turns);

\- it has no tools or functions;

\- temperature is 0.7 or lower.



Reasons: in a multi-turn chat, a message like "what about the second one?" is

meaningless without context and would match unrelated conversations. Tool calls

depend on external state. High-temperature requests expect varied output.

Everything else is marked BYPASS. Trade-off: lower hit rate in exchange for

fewer wrong answers.



\### 4.6 Similarity threshold is configurable per tenant

Uses RedisVL's cosine \*\*distance\*\* (0 to 2, lower is stricter), default 0.15 in

this project. Different tenants tolerate different risk: a legal-FAQ bot wants

strict matching, a casual assistant can accept looser matches. Note this is the

opposite direction of LangCache's similarity score, so any comparison between

backends must convert scales. The chosen default is justified by the evaluation

in section 7.



\### 4.7 Cache backend is behind a small interface

The gateway calls `lookup` and `store` on a backend class. RedisVL is the first

implementation. A managed LangCache backend can be added without changing the

gateway. RedisVL calls are synchronous, so they run in worker threads

(`asyncio.to\_thread`) to avoid blocking the event loop.



\### 4.8 Rate limiting: fixed-window counter in Redis

`INCR` on `rl:{tenant}:{minute}` with a short expiry. Simple, fast, and

per-tenant configurable. Weakness: a burst across a window boundary can briefly

reach about 2x the limit. A sliding window or token bucket would fix this.



\### 4.9 Streaming

\- On a HIT, the cached answer is replayed as OpenAI-style chunks.

\- On a MISS, chunks are passed through to the client as they arrive while the

&#x20; text is collected; the answer is stored only if the provider finished cleanly

&#x20; (`finish\_reason: stop`). Disconnects, provider errors, or truncated answers

&#x20; are never cached.

\- The first provider chunk is fetched before the response begins, so a provider

&#x20; failure returns a real HTTP 502 instead of a broken 200 stream.



\### 4.10 Observability by request log

Every request is logged to SQLite with its cache status and latency. `stats.py`

derives per-tenant hit rate and average hit/miss latency from it. Prometheus and

Grafana are planned (section 8).



\## 5. Security notes



\- Tenant IDs are restricted to `\[a-z0-9\_]` (safe in tag filters).

\- Cache poisoning is limited to the tenant's own scope, since entries are never

&#x20; shared across tenants.

\- Cached responses may contain user data; entries are tenant-scoped, but there

&#x20; is no PII detection yet (see limitations).



\## 6. Known limitations



\- Multi-turn conversations are not cached (see 4.5).

\- No cache TTL or invalidation yet, so stale answers can persist.

\- No PII detection; sensitive prompts are cached like any other.

\- Token counts are zero for streamed responses unless the provider reports usage.

\- No cost estimation or dollars-saved metric yet.

\- Client disconnects mid-stream do not cache the partial answer.

\- SQLite is a single-writer store; fine for a demo, not for high concurrency.

\- Tested so far against a mock provider only.



\## 7. Evaluation (to be completed)



Planned: a labelled set of prompt pairs (paraphrases that should match, and

near-miss pairs that should not), swept across distance thresholds.



| Threshold (distance) | Hit rate | False-hit rate | Precision |

|---|---|---|---|

| 0.05 | TBD | TBD | TBD |

| 0.10 | TBD | TBD | TBD |

| 0.15 | TBD | TBD | TBD |

| 0.20 | TBD | TBD | TBD |

| 0.30 | TBD | TBD | TBD |



Recommended default threshold: TBD, with reasoning.



Load test (cache on vs off): TBD.



\## 8. Roadmap



1\. Real provider integration and fallback to a second provider.

2\. Prometheus metrics and a Grafana dashboard.

3\. Threshold evaluation and load test (section 7).

4\. Admin API: create tenants, rotate keys, usage, per-tenant cache invalidation

&#x20;  (invalidation must always be scoped to a tenant).

5\. TTL policy and bypass rules for time-sensitive prompts.

6\. Optional: LangCache backend and a backend-vs-backend comparison.

