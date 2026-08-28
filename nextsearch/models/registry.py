"""Model registry — pure data, one entry per model.

Three kinds of entry live here:

  * the NextSearch-1 releases, served locally through any OpenAI-compatible
    server (vLLM is what docs/serving.md describes). Their `base_url` comes
    from `NEXTSEARCH_BASE_URL` or the CLI's `--base-url`, so one entry works
    for a laptop, a pod, or a cluster.
  * the default judge, which every reported number was graded with.
  * a couple of hosted reference models, useful as a sanity check that the
    harness works end to end before you serve anything yourself.

Anything not registered still works: a name containing "/" is treated as a
raw OpenRouter model id, so any model that provider serves can be evaluated
without editing this file.

The sampling settings on the NextSearch-1 entries are the ones the models
were trained and evaluated under. Changing them changes the numbers.
"""

import os

from . import Model

# One knob for every locally served release. Point it at your own server.
LOCAL_BASE_URL = os.environ.get("NEXTSEARCH_BASE_URL",
                                "http://localhost:8000/v1")

# Shared serving configuration for the released checkpoints: 16k completion
# cap, 64k context. Thinking is ON — these are reasoning models, and the
# research policy was trained with it enabled.
_SERVE = {"temperature": 0.7, "max_tokens": 16384}

MODELS = {m.name: m for m in [
    # ---- NextSearch-1 releases (serve them yourself; see docs/serving.md)
    Model(name="nextsearch-1-m", client="vllm", base_url=LOCAL_BASE_URL,
          model_id="NextTokenAI/NextSearch-1-M",
          # The M release runs at a continuous reasoning-effort setting of
          # 0.7, which is what it was trained at. See docs/serving.md for how
          # to pass this through your server.
          sampling={**_SERVE, "extra_body": {"reasoning_effort": 0.7}}),
    Model(name="nextsearch-1-s", client="vllm", base_url=LOCAL_BASE_URL,
          model_id="NextTokenAI/NextSearch-1-S",
          sampling=dict(_SERVE)),
    Model(name="nextsearch-1-xs", client="vllm", base_url=LOCAL_BASE_URL,
          model_id="NextTokenAI/NextSearch-1-XS",
          sampling=dict(_SERVE)),

    # ---- experiment: LFM2.5-2.6B, served locally through vLLM (see
    # docs/lfm2-step1.md). Deliberately NOT built from `_SERVE`: LFM2.5 is a
    # different model family and must not inherit NextSearch-1's temperature
    # 0.7. Its sampling is the Stage-1 web-research recipe — low temperature
    # for stable tool-calling, an 8k completion cap, and the vLLM-only
    # top_k / repetition_penalty passed through `extra_body`.
    Model(name="lfm2.5-2.6b", client="vllm", base_url=LOCAL_BASE_URL,
          model_id="LiquidAI/LFM2.5-2.6B",
          sampling={"temperature": 0.1, "max_tokens": 8192,
                    "extra_body": {"top_k": 50, "repetition_penalty": 1.1}}),

    # ---- graders
    # The default judge for every benchmark. All reported NextSearch-1 numbers
    # were graded with this model; changing it changes the scores, so treat a
    # different judge as a different evaluation.
    Model(name="gpt-5.6-luna-low", client="openrouter",
          model_id="openai/gpt-5.6-luna",
          sampling={"reasoning_effort": "low", "max_tokens": 8192}),

    # An alternative judge for users with a Google AI Studio key
    # (GEMINI_API_KEY) instead of OPENROUTER_API_KEY — served through Gemini's
    # OpenAI-compatible endpoint. This is NOT the judge the reported
    # NextSearch-1 numbers were graded with, so scores produced with it are a
    # self-consistent evaluation of their own, not comparable to the headline
    # table. Pass it explicitly with `--judge gemini-3.6-flash`.
    Model(name="gemini-3.6-flash", client="gemini",
          model_id="gemini-3.6-flash",
          sampling={"temperature": 0.0, "max_tokens": 8192}),

    # ---- hosted reference models, for smoke-testing the harness
    Model(name="gpt-5.6-luna-med", client="openrouter",
          model_id="openai/gpt-5.6-luna",
          sampling={"reasoning_effort": "medium", "max_tokens": 16384}),
    Model(name="deepseek-v4-flash", client="openrouter",
          model_id="deepseek/deepseek-v4-flash",
          sampling={"max_tokens": 16384}),
]}

DEFAULT_JUDGE = "gpt-5.6-luna-low"
# The orchestrated harness's default research subagent, matching
# harnesses.SubagentDefaults.
DEFAULT_SUBAGENT = "deepseek-v4-flash"


def get_model(name, base_url=None) -> Model:
    """Registry lookup. A name containing "/" is an ad-hoc OpenRouter id, so
    unregistered models work without a code change. `base_url` overrides the
    endpoint for self-hosted entries."""
    from dataclasses import replace
    if name in MODELS:
        m = MODELS[name]
        return replace(m, base_url=base_url) \
            if base_url and m.client == "vllm" else m
    if "/" in name:
        return Model(name=name.replace("/", "--"), client="openrouter",
                     model_id=name)
    raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)} "
                   "(or pass a 'vendor/model' OpenRouter id)")
