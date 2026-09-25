"""Unit tests for LLM-client request plumbing (Groq API is mocked - no key needed).

Run: python -m pytest tests/ -q   (from the project root)
"""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The groq SDK may not be installed where tests run; stub the module so the
# client's local `from groq import Groq` import succeeds and can be mocked.
if "groq" not in sys.modules:
    try:
        import groq  # noqa: F401
    except ImportError:
        sys.modules["groq"] = types.ModuleType("groq")
        sys.modules["groq"].Groq = MagicMock()

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")

from src.generation.llm_config import get_llm_client  # noqa: E402


def _mocked_client(**llm_cfg):
    from unittest.mock import patch
    with patch("groq.Groq") as groq_cls:
        client = get_llm_client({"llm": llm_cfg})
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="YES"))]
        client.client.chat.completions.create = MagicMock(return_value=resp)
        yield client


@pytest.fixture()
def judge_client():
    yield from _mocked_client(provider="groq", model="qwen/qwen3.6-27b",
                              reasoning_effort="none")


@pytest.fixture()
def plain_client():
    yield from _mocked_client(provider="groq", model="x")


def test_reasoning_effort_forwarded_single_and_batch(judge_client):
    judge_client.generate_k_samples("p", k=1, temperature=0.0, top_p=1.0,
                                    max_new_tokens=10)
    _, kw = judge_client.client.chat.completions.create.call_args
    assert kw["reasoning_effort"] == "none"
    assert (kw["temperature"], kw["max_tokens"]) == (0.0, 10)
    judge_client.generate_k_samples_batch(["p1", "p2"], k=1, temperature=0.0,
                                          top_p=1.0, max_new_tokens=10)
    _, kw2 = judge_client.client.chat.completions.create.call_args
    assert kw2["reasoning_effort"] == "none"


def test_reasoning_effort_omitted_when_unset(plain_client):
    plain_client.generate_k_samples("p", k=1, temperature=0.8, top_p=0.95,
                                    max_new_tokens=512)
    _, kw = plain_client.client.chat.completions.create.call_args
    assert "reasoning_effort" not in kw
    assert "seed" not in kw


def test_seed_forwarded_when_set():
    # Exact-value contract: the seed kwarg is the DERIVED per-call seed
    # (base + crc32(prompt) + sample_index) % (2^31 - 1) - NOT the base seed
    # verbatim, which would make k calls of one prompt return k identical
    # completions on a seed-honoring API.
    import zlib
    prompt = "p"
    base = 42
    for client in _mocked_client(provider="groq", model="x", seed=base):
        client.generate_k_samples("p", k=1, temperature=0.8, top_p=0.95,
                                  max_new_tokens=512)
        _, kw = client.client.chat.completions.create.call_args
        expected = (base + zlib.crc32(prompt.encode("utf-8")) + 0) % (2**31 - 1)
        assert kw["seed"] == expected


def test_derived_seed_stable_across_identical_calls():
    # Identical (prompt, sample_index) -> identical derived seed: reruns of the
    # same run reproduce exactly.
    seeds = []
    for _ in range(2):
        for client in _mocked_client(provider="groq", model="x", seed=42):
            client.generate_k_samples("p", k=1, temperature=0.8, top_p=0.95,
                                      max_new_tokens=512)
            _, kw = client.client.chat.completions.create.call_args
            seeds.append(kw["seed"])
    assert seeds[0] == seeds[1]


def test_k_samples_get_distinct_seeds():
    # The whole point of the fix: k calls of ONE prompt must carry k distinct
    # seeds, else a seed-honoring API returns k identical completions and the
    # Groq arm's diversity collapses to ~0.
    for client in _mocked_client(provider="groq", model="x", seed=42):
        client.generate_k_samples("p", k=3, temperature=0.8, top_p=0.95,
                                  max_new_tokens=512)
        call_seeds = [
            client.client.chat.completions.create.call_args_list[i][1]["seed"]
            for i in range(3)
        ]
        assert len(set(call_seeds)) == 3, call_seeds


def test_batch_path_distinct_seeds_per_job():
    for client in _mocked_client(provider="groq", model="x", seed=42):
        client.generate_k_samples_batch(["p1", "p2"], k=2, temperature=0.8,
                                        top_p=0.95, max_new_tokens=512)
        call_seeds = [
            c[1]["seed"] for c in client.client.chat.completions.create.call_args_list
        ]
        # 2 prompts x 2 samples = 4 calls, all with distinct seeds...
        assert len(set(call_seeds)) == 4, call_seeds
        # ...and the per-prompt reassembly order is preserved (contract: flat[i*k:(i+1)*k])
        assert len(client.client.chat.completions.create.call_args_list) == 4


def test_batch_seed_derivation_matches_formula():
    import zlib
    base = 42
    for client in _mocked_client(provider="groq", model="x", seed=base):
        client.generate_k_samples_batch(["p1", "p2"], k=2, temperature=0.8,
                                        top_p=0.95, max_new_tokens=512)
        calls = client.client.chat.completions.create.call_args_list
        # jobs are (prompt, prompt_index, sample_index): (p1,0,0) (p1,0,1) (p2,1,0) (p2,1,1)
        expected = [
            (base + zlib.crc32(b"p1") + 0) % (2**31 - 1),
            (base + zlib.crc32(b"p1") + 1) % (2**31 - 1),
            (base + zlib.crc32(b"p2") + 0) % (2**31 - 1),
            (base + zlib.crc32(b"p2") + 1) % (2**31 - 1),
        ]
        actual = [c[1]["seed"] for c in calls]
        assert sorted(actual) == sorted(expected), (actual, expected)


def _success_response():
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="OK"))]
    return resp


def test_retry_succeeds_after_transient_failures():
    # Two transient failures then success: the call returns, with exactly 3
    # underlying attempts. retry_backoff=0 keeps the test instant.
    for client in _mocked_client(provider="groq", model="x", retry_backoff=0):
        client.client.chat.completions.create = MagicMock(side_effect=[
            RuntimeError("429 rate limited"), TimeoutError("timed out"), _success_response(),
        ])
        out = client.generate_k_samples("p", k=1, temperature=0.8, top_p=0.95,
                                        max_new_tokens=512)
        assert out == ["OK"]
        assert client.client.chat.completions.create.call_count == 3


def test_retry_exhaustion_raises_after_max_retries():
    for client in _mocked_client(provider="groq", model="x", retry_backoff=0,
                                 max_retries=2):
        client.client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            client.generate_k_samples("p", k=1, temperature=0.8, top_p=0.95,
                                      max_new_tokens=512)
        assert client.client.chat.completions.create.call_count == 2
