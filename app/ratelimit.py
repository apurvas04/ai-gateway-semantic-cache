import time

import redis

_r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def check_rate_limit(tenant_id: str, limit_per_min: int) -> bool:
    """Fixed-window counter: simple, and good enough to start."""
    window = int(time.time() // 60)
    key = f"rl:{tenant_id}:{window}"
    count = _r.incr(key)
    if count == 1:
        _r.expire(key, 70)
    return count <= limit_per_min