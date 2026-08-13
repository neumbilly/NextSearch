"""Tool registry: canonical spec + async executor pairs.

These implementations are the same ones used to generate training rollouts, so
they are kept dependency-light and free of side effects beyond the API call
itself. A tool EXECUTION error is environment behavior, not a crash: the
harness feeds the error text back to the model as the tool result and records
it, and the episode carries on.
"""

from dataclasses import dataclass, field


@dataclass
class Tool:
    spec: dict                     # {name, description, parameters}
    execute: object                # async (args: dict) -> (content: str, info: dict)
    config: dict = field(default_factory=dict)  # provider/params, frozen in the manifest
    max_calls: "int | None" = None  # per-EPISODE call budget enforced by the
    # harness: over-budget calls get a "budget exhausted" tool message, cost
    # $0, and are never executed. None = unbounded. The cost guard for
    # expensive tools.
    wants_context: bool = False    # execute(args, context={episode_id, call_id,
    # bench}) — for tools that record per-call artifacts, such as the
    # research subagent's nested rollouts


# Every search backend rate-limits under evaluation concurrency, and a 429 fed
# to the model as a tool error is an infrastructure artifact masquerading as
# search quality — it is research the agent silently did without. Retry 429
# and 5xx with exponential backoff before surfacing anything. Assume no
# backend is exempt.
BACKOFF_BASE_S = 1.0


async def with_backoff(call, *args, attempts=5):
    """await call(*args), retrying 429/5xx up to `attempts` times with
    exponential backoff plus jitter. Returns (result, n_attempts)."""
    import asyncio
    import random

    import httpx
    for i in range(attempts):
        try:
            return await call(*args), i + 1
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            retryable = status == 429 or status >= 500
            if not retryable or i == attempts - 1:
                raise
            await asyncio.sleep(BACKOFF_BASE_S * 2 ** i
                                + random.uniform(0, BACKOFF_BASE_S))


# The --search-mode value space. Bare modes are Parallel tiers;
# '<provider>-<mode>' dispatches to that provider's tool module.
SEARCH_MODES = ("turbo", "basic", "advanced",
                "exa-auto", "exa-fast", "exa-deep",
                "tavily-basic", "tavily-advanced")


def _search_tool(search_mode):
    from . import exa, parallel, tavily
    if not search_mode:
        return parallel.tool()          # the default backend and tier
    provider, _, mode = search_mode.partition("-")
    if provider == "exa":
        return exa.tool(mode)
    if provider == "tavily":
        return tavily.tool(mode)
    return parallel.tool(search_mode)


def get_tools(names, search_mode=None) -> list:
    """`search_mode` selects the backend and tier for the `search` tool;
    values are listed in SEARCH_MODES, and None means Parallel turbo."""
    from . import fetch
    registry = {"search": lambda: _search_tool(search_mode),
                "fetch": fetch.tool}
    out = []
    for n in names:
        if n not in registry:
            raise KeyError(
                f"unknown tool {n!r}; available: {sorted(registry)}")
        out.append(registry[n]())
    return out
