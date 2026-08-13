"""The OpenAI-compatible chat-completions client.

One client, three base URLs: OpenAI, OpenRouter, and any self-hosted
OpenAI-compatible server (vLLM, SGLang). Sampling dicts pass through as
`create()` kwargs verbatim — temperature, max_tokens, reasoning_effort,
extra_body — with no translation layer, so what a model accepts is data on its
registry entry rather than code here.

Transient failures are retried by the SDK; anything that still raises is the
harness's problem, which captures it per-episode rather than crashing a run.
"""

import os

from .. import types
from . import Client


class ProviderNoChoices(RuntimeError):
    """The provider answered without a choices array — for example an error
    body delivered as a well-formed 200 response, which happens when a prompt
    trips provider-side moderation.

    A retry re-sends the same prompt and gets the same block, so callers that
    can degrade per-item (the graders) catch THIS class specifically. Ordinary
    transport errors stay ordinary exceptions and still fail the run.
    """


def attach_reasoning_details(messages, converted):
    """Re-attach provider `reasoning_details` (encrypted thought signatures)
    to the wire-format assistant messages.

    Some providers require these back on every subsequent turn of a tool loop;
    dropping them makes long tool chains degrade into empty completions. The
    canonical converter strips unknown keys, so the round-trip lives here, in
    the wire layer, per the client contract.
    """
    if len(messages) != len(converted):  # converter changed shape: don't guess
        return converted
    for src, dst in zip(messages, converted):
        if src.get("role") == "assistant" and src.get("reasoning_details"):
            dst["reasoning_details"] = src["reasoning_details"]
    return converted


class OpenAICompatClient(Client):
    def __init__(self, base_url=None, api_key_env="OPENAI_API_KEY",
                 api_key_default=None, max_retries=5):
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key_default = api_key_default
        self.max_retries = max_retries
        self._client = None

    def _get(self):
        if self._client is None:
            import openai
            key = os.environ.get(self.api_key_env) or self.api_key_default
            if not key:
                raise RuntimeError(
                    f"{self.api_key_env} not set (a .env file in the working "
                    "directory is loaded automatically)")
            self._client = openai.AsyncOpenAI(
                base_url=self.base_url, api_key=key,
                max_retries=self.max_retries)
        return self._client

    async def generate(self, model_id, messages, tools, sampling):
        kwargs = dict(sampling)
        if tools:
            kwargs["tools"] = types.tools_to_openai(tools)
        resp = await self._get().chat.completions.create(
            model=model_id,
            messages=attach_reasoning_details(
                messages, types.to_openai_messages(messages)),
            **kwargs,
        )
        if not resp.choices:
            # Never index resp.choices[0] on this shape: the TypeError it
            # raises hides the actual provider error payload.
            raise ProviderNoChoices(
                f"{model_id}: provider returned no choices "
                f"(error={getattr(resp, 'error', None)!r})")
        # model_dump keeps provider extras such as vLLM's reasoning_content
        raw = resp.choices[0].message.model_dump(exclude_none=True)
        # Some providers ship thinking as `reasoning`; normalize to the
        # canonical field, because storage never strips thinking.
        if not raw.get("reasoning_content") and raw.get("reasoning"):
            raw["reasoning_content"] = raw["reasoning"]
        msg = types.from_openai_message(raw)
        # Keep provider thought signatures on the canonical message so the
        # next turn can send them back (see attach_reasoning_details).
        if raw.get("reasoning_details"):
            msg["reasoning_details"] = raw["reasoning_details"]
        usage = resp.usage.model_dump(exclude_none=True) if resp.usage else {}
        return msg, usage
