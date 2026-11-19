import uuid

import pytest

from app.cache import RedisVLBackend, make_scope_hash

S = make_scope_hash("m", "sys")


@pytest.fixture
def backend():
    b = RedisVLBackend(name=f"test_{uuid.uuid4().hex[:8]}")
    yield b
    b.cache.delete()


def test_verifier_rejects_one_word_intent_change(backend):
    backend.store("tenant_a", "What is the last date to pay the semester fee?", "Fee deadline.", S)
    assert backend.lookup("tenant_a", "What is the last date to register for the semester?", S) is None


def test_verifier_accepts_borderline_paraphrase(backend):
    backend.store("tenant_a", "What documents are needed for admission?", "Docs list.", S)
    assert backend.lookup("tenant_a", "Which papers do I have to submit to get admitted?", S) == "Docs list."


@pytest.mark.xfail(strict=True, reason="Known failure: verifier scores cancel/pause at 0.93")
def test_known_failure_cancel_vs_pause(backend):
    backend.store("tenant_a", "How can I cancel my subscription?", "Cancel steps.", S)
    assert backend.lookup("tenant_a", "How can I pause my subscription?", S) is None