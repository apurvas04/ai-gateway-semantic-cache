import json
import sys
import time

import httpx

key, question = sys.argv[1], sys.argv[2]
start = time.perf_counter()
first, text = None, ""

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "test-model", "stream": True,
          "messages": [{"role": "user", "content": question}]},
    timeout=60,
) as r:
    print("status", r.status_code, "| X-Cache:", r.headers.get("X-Cache"))
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        for ch in json.loads(data).get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter() - start
                text += piece

total = time.perf_counter() - start
print(f"first token: {(first or 0) * 1000:.0f} ms | total: {total * 1000:.0f} ms")
print(text)