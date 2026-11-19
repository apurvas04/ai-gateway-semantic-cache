import hashlib

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag

_verifiers = {}


def _get_verifier(name):
    if name not in _verifiers:
        from sentence_transformers import CrossEncoder
        _verifiers[name] = CrossEncoder(name)
    return _verifiers[name]


def make_scope_hash(model: str, system_prompt: str) -> str:
    """Entries only match when model and system prompt are identical."""
    return hashlib.sha256(f"{model}||{system_prompt}".encode()).hexdigest()[:16]


class RedisVLBackend:
    def __init__(self, name="gateway_cache",
                 redis_url="redis://localhost:6379",
                 auto_accept=0.08, verify_max=0.30, verify_min_score=0.8,
                 use_verifier=True,
                 verifier_model="cross-encoder/quora-roberta-base"):
        self.auto_accept = auto_accept
        self.verify_max = verify_max
        self.verify_min_score = verify_min_score
        self.verifier = _get_verifier(verifier_model) if use_verifier else None
        self.cache = SemanticCache(
            name=name,
            redis_url=redis_url,
            distance_threshold=verify_max,
            filterable_fields=[
                {"name": "tenant_id", "type": "tag"},
                {"name": "scope_hash", "type": "tag"},
            ],
        )

    def lookup(self, tenant_id, prompt, scope_hash, distance_threshold=None):
        # A tenant's distance_threshold overrides the auto-accept zone only.
        auto = self.auto_accept if distance_threshold is None else distance_threshold
        gate = max(self.verify_max, auto)
        flt = (Tag("tenant_id") == tenant_id) & (Tag("scope_hash") == scope_hash)
        hits = self.cache.check(
            prompt=prompt, filter_expression=flt,
            num_results=3, distance_threshold=gate,
        )
        for h in hits:  # sorted nearest first
            if float(h["vector_distance"]) <= auto:
                return h["response"]
        if self.verifier is None or not hits:
            return None
        scores = self.verifier.predict([(prompt, h["prompt"]) for h in hits])
        best = max(range(len(hits)), key=lambda i: scores[i])
        return hits[best]["response"] if scores[best] >= self.verify_min_score else None

    def store(self, tenant_id, prompt, response, scope_hash):
        self.cache.store(
            prompt=prompt,
            response=response,
            filters={"tenant_id": tenant_id, "scope_hash": scope_hash},
        )