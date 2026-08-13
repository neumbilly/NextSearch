"""The Parallel Search API as the `search` tool — the default backend.

One call is one web search: a natural-language objective plus keyword queries
in, ranked results with url/title/excerpts out, formatted as numbered text for
the tool message. Latency and per-call cost go into the info dict the harness
records.

The `mode` parameter selects a speed/quality tier. Turbo is the default: on a
paired comparison it matched the next tier up on accuracy while costing 5x
less, returning roughly 40% shorter excerpts (so cheaper model input too) and
answering about 3x faster.

Note that the MODEL-facing tool spec is mode-independent. Backend and tier are
harness configuration the model never sees, which is what makes a
search-backend comparison a controlled one.

Parameter shape and wording follow the provider's own tool-definition
recommendation: search APIs are tuned for different query formulations, and
matching each one's documented guidance is part of using it fairly.
"""

import os
import time

from ..types import tool_spec

ENDPOINT = "https://api.parallel.ai/v1/search"
DEFAULT_MODE = "turbo"
MAX_RESULTS = 10
MAX_CHARS_PER_RESULT = 1500
COST_PER_CALL_USD = {"turbo": 0.001, "basic": 0.005, "advanced": 0.005}

# The tool NAME is `search` across every backend: the registry key, the
# rollout records, and comparability across providers all hang off it, and the
# name itself carries no guidance.
SPEC = tool_spec(
    "search",
    "Searches the web for current and factual information, returning relevant "
    "results with titles, URLs, and content snippets. If results are "
    "insufficient, search again with different query angles rather than "
    "repeating the same wording.",
    {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "A concise, self-contained description of the web "
                    "research goal. Must include the key entity or topic being "
                    "searched for; include source or freshness guidance when "
                    "relevant (e.g. 'prefer official sources', 'as of 2026')."},
            "search_queries": {
                "type": "array", "items": {"type": "string"},
                "minItems": 3, "maxItems": 3,
                "description": "Exactly 3 keyword search queries, each 3-6 words. "
                    "Must be diverse — vary entity names, synonyms, and angles. "
                    "Each query must include the key entity or topic. NEVER "
                    "write sentences, instructions, or use site: operators."},
        },
        "required": ["objective", "search_queries"],
    },
)


def make_execute(mode):
    async def execute(args):
        import httpx
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError("PARALLEL_API_KEY not set")
        objective = (args.get("objective") or "").strip()
        # Models occasionally emit tool calls with empty arguments, and the
        # API rejects blank input. Raise a clear message instead — a tool
        # error is feedback the model can act on.
        if not objective:
            raise ValueError("search requires a non-empty 'objective' "
                             "describing what to find")
        queries = [q for q in (args.get("search_queries") or [])
                   if isinstance(q, str) and q.strip()][:5] or [objective]
        body = {"objective": objective, "search_queries": queries, "mode": mode}

        async def call():
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(ENDPOINT, json=body,
                                         headers={"x-api-key": key})
                resp.raise_for_status()
                return resp.json()

        from . import with_backoff
        t0 = time.monotonic()
        data, attempts = await with_backoff(call)
        latency = round(time.monotonic() - t0, 3)  # includes backoff sleeps
        lines = []
        for i, r in enumerate(data.get("results", [])[:MAX_RESULTS], 1):
            head = f"[{i}] {r.get('title') or '(untitled)'} — {r.get('url', '')}"
            excerpts = "\n".join(e.strip() for e in r.get("excerpts", [])
                                 if e.strip())
            excerpts = excerpts[:MAX_CHARS_PER_RESULT]
            lines.append(f"{head}\n{excerpts}" if excerpts else head)
        content = "\n\n".join(lines) or "(no results)"
        info = {"tool": "search", "mode": mode, "latency_s": latency,
                "n_results": len(data.get("results", [])),
                "cost_usd": COST_PER_CALL_USD[mode]}
        if attempts > 1:
            info["retries"] = attempts - 1
        return content, info
    return execute


def tool(mode=DEFAULT_MODE):
    from . import Tool
    if mode not in COST_PER_CALL_USD:
        raise KeyError(f"unknown search mode {mode!r}; "
                       f"available: {sorted(COST_PER_CALL_USD)}")
    return Tool(spec=SPEC, execute=make_execute(mode),
                config={"provider": "parallel", "endpoint": ENDPOINT,
                        "mode": mode, "max_results": MAX_RESULTS,
                        "max_chars_per_result": MAX_CHARS_PER_RESULT,
                        "cost_per_call_usd": COST_PER_CALL_USD[mode]})
