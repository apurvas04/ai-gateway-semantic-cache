import asyncio
import json
import os
import re

import httpx

_PIECES = re.compile(r"\S+\s*|\s+")


async def call_llm(payload: dict) -> dict:
    if os.getenv("LLM_MODE", "mock") == "mock":
        await asyncio.sleep(1.0)  # simulate a slow LLM so cache hits are visible
        last = str(payload["messages"][-1]["content"])
        return {
            "id": "mock-1",
            "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"[mock answer] {last}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    base = os.environ["LLM_BASE_URL"].rstrip("/")
    key = os.environ["LLM_API_KEY"]
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        )
        r.raise_for_status()
        return r.json()


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