import asyncio
import json
import os
import re

import httpx

_PIECES = re.compile(r"\S+\s*|\s+")
RETRY_STATUS = {429, 500, 502, 503, 504}


async def _post_with_retry(client, url, payload, headers, attempts=3):
    """POST with exponential backoff on transient errors (429, 5xx, network)."""
    last = None
    for i in range(attempts):
        try:
            r = await client.post(url, json=payload, headers=headers)
        except httpx.TransportError as e:
            last = e
        else:
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()  # permanent errors (400, 404...) raise immediately
                return r.json()
            last = httpx.HTTPStatusError(
                f"Upstream returned {r.status_code}", request=r.request, response=r
            )
        if i < attempts - 1:
            await asyncio.sleep(2**i)  # wait 1s, then 2s
    raise last


async def call_llm(payload: dict) -> dict:
    if os.getenv("LLM_MODE", "mock") == "mock":
        await asyncio.sleep(1.0)  # simulate a slow LLM so cache hits are visible
        last_msg = str(payload["messages"][-1]["content"])
        return {
            "id": "mock-1",
            "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"[mock answer] {last_msg}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    base = os.environ["LLM_BASE_URL"].rstrip("/")
    key = os.environ["LLM_API_KEY"]
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    fallbacks = [
        m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
    ]
    models = [payload.get("model")] + fallbacks

    last = None
    async with httpx.AsyncClient(timeout=60) as client:
        for m in models:
            try:
                return await _post_with_retry(client, url, {**payload, "model": m}, headers)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in RETRY_STATUS:
                    raise  # a permanent error: a fallback model won't fix a bad request
                last = e
            except httpx.TransportError as e:
                last = e
    raise last


def _chunk(model: str, delta: dict, finish=None) -> dict:
    return {
        "id": "mock-stream",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def stream_llm(payload: dict):
    """Yield OpenAI-style chunk dicts until the provider finishes."""
    model = payload.get("model", "mock")

    if os.getenv("LLM_MODE", "mock") == "mock":
        await asyncio.sleep(0.5)  # time to first token
        text = f"[mock answer] {payload['messages'][-1]['content']}"
        yield _chunk(model, {"role": "assistant"})
        for piece in _PIECES.findall(text):
            await asyncio.sleep(0.1)
            yield _chunk(model, {"content": piece})
        yield _chunk(model, {}, finish="stop")
        return

    base = os.environ["LLM_BASE_URL"].rstrip("/")
    key = os.environ["LLM_API_KEY"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=120)) as client:
        async with client.stream(
            "POST",
            f"{base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        ) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"Upstream {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue