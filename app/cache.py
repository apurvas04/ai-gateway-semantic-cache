import hashlib
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag


def make_scope_hash(model: str, system_prompt: str) -> str:
    """Entries only match when model and system prompt are identical."""
    return hashlib.sha256(f"{model}||{system_prompt}".encode()).hexdigest()[:16]


class RedisVLBackend:
    def __init__(self, name="gateway_cache",
                 redis_url="redis://localhost:6379",
                 distance_threshold=0.15):
        self.cache = SemanticCache(
            name=name,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            filterable_fields=[
                {"name": "tenant_id", "type": "tag"},
                {"name": "scope_hash", "type": "tag"},
            ],
        )

    def lookup(self, tenant_id, prompt, scope_hash, distance_threshold=None):
        flt = (Tag("tenant_id") == tenant_id) & (Tag("scope_hash") == scope_hash)
        hits = self.cache.check(
            prompt=prompt,
            filter_expression=flt,
            num_results=1,
            distance_threshold=distance_threshold,
        )
        return hits[0]["response"] if hits else None

    def store(self, tenant_id, prompt, response, scope_hash):
        self.cache.store(
            prompt=prompt,
            response=response,
            filters={"tenant_id": tenant_id, "scope_hash": scope_hash},
        )