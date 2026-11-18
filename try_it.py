import sys
import httpx

key = sys.argv[1]
questions = [
    "What is the capital of France?",
    "What is the capital of France?",
    "Which city is France's capital?",
    "What is the capital of Germany?",
]
for q in questions:
    r = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "test-model", "messages": [{"role": "user", "content": q}]},
        timeout=60,
    )
    print(r.status_code, r.headers.get("X-Cache", "-"), r.headers.get("X-Latency-Ms", "-"), "ms |", q)