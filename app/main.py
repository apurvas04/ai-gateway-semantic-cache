import asyncio
import json
import re
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.cache import RedisVLBackend, make_scope_hash
from app.db import get_tenant_by_key, init_db, log_request
from app.providers import call_llm, stream_llm
from app.ratelimit import check_rate_limit

app = FastAPI(title="AI Gateway")
init_db()
cache = RedisVLBackend(name="gw_cache")

_PIECES = re.compile(r"\S+\s*|\s+")
_SSE_HEADERS = {"Cache-Control": "no-cache"}


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _prepare(payload: dict, authorization: str | None) -> dict:
    """Auth, limits, validation and cache policy shared by both request types."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    tenant = get_tenant_by_key(authorization[7:])
    if tenant is None:
        raise HTTPException(401, "Invalid API key")
    tenant_id, cfg = tenant["tenant_id"], tenant["config"]

    limit = int(cfg.get("rate_limit_per_min", 60))
    if not await asyncio.to_thread(check_rate_limit, tenant_id, limit):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})

    model = payload.get("model", "default")
    allowed = cfg.get("allowed_models")
    if allowed and model not in allowed:
        raise HTTPException(403, f"Model '{model}' is not allowed for this tenant")

    messages = payload.get("messages") or []
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message")

    single_turn = len(user_msgs) == 1 and not any(m.get("role") == "assistant" for m in messages)
    use_cache = (
        cfg.get("cache_enabled", True)
        and single_turn
        and not payload.get("tools")
        and (payload.get("temperature") or 0) <= 0.7
    )
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    return {
        "tenant_id": tenant_id,
        "cfg": cfg,
        "model": model,
        "prompt": str(user_msgs[-1].get("content", "")),
        "scope": make_scope_hash(model, system),
        "use_cache": use_cache,
    }


async def _lookup(ctx: dict):
    return await asyncio.to_thread(
        cache.lookup, ctx["tenant_id"], ctx["prompt"], ctx["scope"],
        ctx["cfg"].get("distance_threshold"),
    )


async def _store(ctx: dict, text: str):
    await asyncio.to_thread(cache.store, ctx["tenant_id"], ctx["prompt"], text, ctx["scope"])


# ---------- streaming helpers ----------

async def _replay(text: str, ctx: dict, start: float):
    """Serve a cached answer as a stream of chunks."""
    base = {"id": "cached", "object": "chat.completion.chunk", "model": ctx["model"]}

    def chunk(delta, finish=None):
        return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    yield chunk({"role": "assistant"})
    for piece in _PIECES.findall(text):
        yield chunk({"content": piece})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"
    log_request(ctx["tenant_id"], ctx["model"], "HIT", _ms(start))


async def _chain(first, rest):
    yield first
    async for c in rest:
        yield c


async def _proxy(first, upstream, ctx: dict, status: str, start: float):
    """Pass provider chunks through, collect the text, store it if the stream completes."""
    parts, usage, finished = [], {}, False
    try:
        async for chunk in _chain(first, upstream):
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    parts.append(piece)
                if ch.get("finish_reason") == "stop":
                    finished = True
            if chunk.get("usage"):
                usage = chunk["usage"]
            yield _sse(chunk)
        yield "data: [DONE]\n\n"
    except Exception:
        # provider died mid-stream: tell the client, cache nothing
        yield _sse({"error": {"message": "Upstream stream failed"}})
        yield "data: [DONE]\n\n"
        log_request(ctx["tenant_id"], ctx["model"], "ERROR", _ms(start))
        return

    text = "".join(parts)
    if ctx["use_cache"] and finished and text:
        await _store(ctx, text)
    log_request(ctx["tenant_id"], ctx["model"], status, _ms(start),
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


async def _stream_response(payload: dict, ctx: dict, start: float):
    status = "BYPASS"
    if ctx["use_cache"]:
        cached = await _lookup(ctx)
        if cached is not None:
            return StreamingResponse(
                _replay(cached, ctx, start),
                media_type="text/event-stream",
                headers={**_SSE_HEADERS, "X-Cache": "HIT"},
            )
        status = "MISS"

    # Pull the first chunk BEFORE responding, so a provider failure becomes a
    # proper HTTP error instead of a broken 200 stream.
    upstream = stream_llm(payload)
    try:
        first = await upstream.__anext__()
    except StopAsyncIteration:
        raise HTTPException(502, "Provider returned an empty stream")
    except Exception:
        raise HTTPException(502, "Upstream provider error")

    return StreamingResponse(
        _proxy(first, upstream, ctx, status, start),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, "X-Cache": status},
    )


# ---------- endpoint ----------

@app.post("/v1/chat/completions")
async def chat(payload: dict, authorization: str | None = Header(default=None)):
    start = time.perf_counter()
    ctx = await _prepare(payload, authorization)

    if payload.get("stream"):
        return await _stream_response(payload, ctx, start)

    status = "BYPASS"
    if ctx["use_cache"]:
        cached = await _lookup(ctx)
        if cached is not None:
            body = {
                "id": "cached",
                "object": "chat.completion",
                "model": ctx["model"],
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": cached},
                             "finish_reason": "stop"}],
            }
            log_request(ctx["tenant_id"], ctx["model"], "HIT", _ms(start))
            return JSONResponse(body, headers={"X-Cache": "HIT", "X-Latency-Ms": str(_ms(start))})
        status = "MISS"

    result = await call_llm(payload)
    if ctx["use_cache"]:
        content = result["choices"][0]["message"]["content"]
        if content:
            await _store(ctx, content)
    usage = result.get("usage") or {}
    ms = _ms(start)
    log_request(ctx["tenant_id"], ctx["model"], status, ms,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    return JSONResponse(result, headers={"X-Cache": status, "X-Latency-Ms": str(ms)})