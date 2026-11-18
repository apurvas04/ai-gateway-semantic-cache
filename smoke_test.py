from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag

cache = SemanticCache(
    name="gateway_cache",
    redis_url="redis://localhost:6379",
    distance_threshold=0.15,
    filterable_fields=[{"name": "tenant_id", "type": "tag"}],
)

cache.store(
    prompt="What is the capital of France?",
    response="Paris.",
    filters={"tenant_id": "tenant_a"},
)

def lookup(prompt, tenant):
    return cache.check(
        prompt=prompt,
        filter_expression=Tag("tenant_id") == tenant,
        num_results=1,
    )

print(lookup("Which city is France's capital?", "tenant_a"))  # expect a hit
print(lookup("Which city is France's capital?", "tenant_b"))  # expect []