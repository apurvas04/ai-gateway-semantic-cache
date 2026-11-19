import os
import sys

import httpx

key = sys.argv[1]
model = os.environ["TEST_MODEL"]
questions = [
    "What is the capital of Spain?",
    "What is the capital of Spain?",
    "Which city is Spain's capital?",
    "What is the capital of Portugal?",
]
for q in questions:
    r = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": q}]},
        timeout=120,
    )
    if r.status_code != 200:
        print(r.status_code, r.text[:200])
        continue
    answer = r.json()["choices"][0]["message"]["content"]
    print(r.headers["X-Cache"], r.headers["X-Latency-Ms"], "ms |", q, "->", answer[:60].replace("\n", " "))