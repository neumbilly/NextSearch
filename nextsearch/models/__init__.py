"""Models: provider clients (code) plus a registry (data).

A `Client` implements the ONE canonical call the harness makes. All wire
conversion — canonical to provider shape, thinking capture, tool-call format —
lives in client files and nowhere else. A `Model` is a pure-data registry
entry, so adding a model is one registry line and adding a provider is one
client file.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Model:
    name: str                      # registry key; used in run paths and results
    client: str                    # "openai" | "openrouter" | "vllm" | "gemini"
    model_id: str                  # the id sent on the wire
    base_url: "str | None" = None  # for self-hosted endpoints
    sampling: dict = field(default_factory=dict)  # temperature, max_tokens, ...
    pricing: dict = field(default_factory=dict)   # {"in": $/Mtok, "out": $/Mtok}
    concurrency: "int | None" = None  # per-model in-flight cap for providers
    # with their own request-rate ceiling; None = only the global cap

    def to_config(self):
        # `concurrency` is an execution knob, not model identity: it cannot
        # affect outputs, so it stays out of the frozen manifest and changing
        # it never invalidates a resume.
        d = asdict(self)
        d.pop("concurrency", None)
        return d


class Client:
    """async generate(model_id, messages, tools, sampling) -> (Message, usage).

    Messages and tools are canonical (see `nextsearch.types`); the return is
    ONE canonical assistant message — content, reasoning_content, tool_calls —
    plus the provider's raw usage dict, which carries prompt and completion
    token counts at minimum.
    """

    async def generate(self, model_id, messages, tools, sampling):
        raise NotImplementedError


_CLIENT_CACHE = {}


def get_client(model: Model) -> Client:
    """Resolve a Model's client, cached per (kind, base_url)."""
    key = (model.client, model.base_url)
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = _make_client(model)
    return _CLIENT_CACHE[key]


def _make_client(model: Model) -> Client:
    from .openai_compat import OpenAICompatClient
    if model.client == "openai":
        return OpenAICompatClient(api_key_env="OPENAI_API_KEY")
    if model.client == "openrouter":
        return OpenAICompatClient(base_url="https://openrouter.ai/api/v1",
                                  api_key_env="OPENROUTER_API_KEY")
    if model.client == "gemini":
        # Google's OpenAI-compatible endpoint, so the same client works with a
        # Google AI Studio key. Useful as a judge for users who have
        # GEMINI_API_KEY rather than OPENROUTER_API_KEY (see the LFM notebook).
        return OpenAICompatClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY")
    if model.client == "vllm":
        if not model.base_url:
            raise ValueError(f"vllm model {model.name!r} needs base_url "
                             "(set NEXTSEARCH_BASE_URL or pass --base-url)")
        return OpenAICompatClient(base_url=model.base_url,
                                  api_key_env="VLLM_API_KEY",
                                  api_key_default="EMPTY")
    raise ValueError(f"unknown client kind: {model.client!r}")


from .registry import DEFAULT_JUDGE, MODELS, get_model  # noqa: E402

__all__ = ["Model", "Client", "get_client", "get_model", "MODELS",
           "DEFAULT_JUDGE"]
