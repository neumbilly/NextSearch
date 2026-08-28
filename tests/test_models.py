"""Registry + client-resolution tests for the models added by the LFM
experiment: the LFM2.5 entry and the Gemini judge. No network."""

import pytest

from nextsearch.models import Model, get_client, get_model
from nextsearch.models.openai_compat import OpenAICompatClient


def test_lfm_entry_is_registered_with_the_experiment_recipe():
    m = get_model("lfm2.5-2.6b")
    assert m.client == "vllm"
    assert m.model_id == "LiquidAI/LFM2.5-2.6B"
    # The Stage-1 recipe, and explicitly NOT NextSearch-1's temperature 0.7.
    assert m.sampling["temperature"] == 0.1
    assert m.sampling["temperature"] != 0.7
    assert m.sampling["max_tokens"] == 8192
    assert m.sampling["extra_body"] == {"top_k": 50, "repetition_penalty": 1.1}


def test_lfm_endpoint_follows_nextsearch_base_url(monkeypatch):
    # A vllm entry's endpoint is overridable, so one registry line serves a
    # laptop, a pod, or Colab.
    m = get_model("lfm2.5-2.6b", base_url="http://localhost:9001/v1")
    assert m.base_url == "http://localhost:9001/v1"


def test_gemini_judge_is_registered_but_not_the_default():
    from nextsearch.models.registry import DEFAULT_JUDGE
    m = get_model("gemini-3.6-flash")
    assert m.client == "gemini"
    assert m.model_id == "gemini-3.6-flash"
    # Reproducibility: the reported judge must stay put even though Gemini is
    # available as an alternative.
    assert DEFAULT_JUDGE == "gpt-5.6-luna-low"


def test_gemini_client_targets_the_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = get_client(get_model("gemini-3.6-flash"))
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url.endswith("/v1beta/openai/")
    assert client.api_key_env == "GEMINI_API_KEY"


def test_vllm_model_requires_a_base_url():
    with pytest.raises(ValueError):
        get_client(Model(name="x", client="vllm", model_id="x", base_url=None))
