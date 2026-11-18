import uuid
import pytest
from app.cache import RedisVLBackend, make_scope_hash


@pytest.fixture
def backend():
    b = RedisVLBackend(name=f"test_{uuid.uuid4().hex[:8]}")
    yield b
    b.cache.delete()  # drops this test's index and data


def test_same_tenant_paraphrase_hits(backend):
    s = make_scope_hash("gpt-x", "You are helpful.")
    backend.store("tenant_a", "What is the capital of France?", "Paris.", s)
    assert backend.lookup("tenant_a", "Which city is France's capital?", s) == "Paris."


def test_other_tenant_misses(backend):
    s = make_scope_hash("gpt-x", "You are helpful.")
    backend.store("tenant_a", "What is the capital of France?", "Paris.", s)
    assert backend.lookup("tenant_b", "What is the capital of France?", s) is None


def test_different_system_prompt_misses(backend):
    s1 = make_scope_hash("gpt-x", "You are helpful.")
    s2 = make_scope_hash("gpt-x", "Answer only in French.")
    backend.store("tenant_a", "What is the capital of France?", "Paris.", s1)
    assert backend.lookup("tenant_a", "What is the capital of France?", s2) is None