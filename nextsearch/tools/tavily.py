"""Tavily /search as an alternative `search` backend.

Same model-facing role as `parallel.py` — one call is one web search, numbered
results with title/url/snippet — but the request interface follows Tavily's
best-practice guidance: one concise query under 400 characters, one topic per
call, with multi-topic questions split into separate searches. Snippets are
Tavily's own relevance chunks.

The tool spec never names the provider: backend choice is harness
configuration the model cannot see.

Modes map to `search_depth`: basic or advanced. Two of Tavily's options are
deliberately unused. `auto_parameters` would let the API choose depth per
call, making cost and behavior nondeterministic and breaking a fixed-config
comparison. `include_answer` would smuggle a second model into the evaluation
— the agent is the synthesizer here, not the search API.

Per-call cost comes from the response's `usage.credits` when present;
`info["cost_source"]` records whether the static table was used instead.
"""

import os
import time

from ..types import tool_spec

ENDPOINT = "https://api.tavily.com/search"
DEFAULT_MODE = "basic"
MAX_RESULTS = 10          # set explicitly — Tavily's own default is 5
MAX_CHARS_PER_RESULT = 1500
USD_PER_CREDIT = 0.008
CREDITS_PER_CALL = {"basic": 1, "advanced": 2}

# First and last sentences shared verbatim with the Parallel spec — the
# backend-neutral surface. The middle is per-provider, written from Tavily's
# best-practice guidance.
SPEC = tool_spec(
    "search",
    "Searches the web for current and factual information, returning relevant "
    "results with titles, URLs, and content snippets. If results are "
    "insufficient, search again with different query angles rather than "
    "repeating the same wording.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise search query, under 400 characters, "
                    "focused on a single topic. Split multi-topic questions "
                    "into separate focused searches rather than combining "
                    "them into one long query."},
        },
        "required": ["query"],
    },
)


async def _search_api(body, key):
    import httpx
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(ENDPOINT, json=body,
                                 headers={"Authorization": f"Bearer {key}"})
        resp.raise_for_status()
        return resp.json()


def make_execute(mode):
    async def execute(args):
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY not set")
        query = (args.get("query") or "").strip()
        if not query:
            raise ValueError("search requires a non-empty 'query' "
                             "describing what to find")
        body = {"query": query, "search_depth": mode,
                "max_results": MAX_RESULTS, "include_answer": False,
                "include_raw_content": False, "include_usage": True}
        from . import with_backoff
        t0 = time.monotonic()
        data, attempts = await with_backoff(_search_api, body, key)
        latency = round(time.monotonic() - t0, 3)  # includes backoff sleeps
        lines = []
        for i, r in enumerate(data.get("results", [])[:MAX_RESULTS], 1):
            head = f"[{i}] {r.get('title') or '(untitled)'} — {r.get('url', '')}"
            snippet = (r.get("content") or "").strip()[:MAX_CHARS_PER_RESULT]
            lines.append(f"{head}\n{snippet}" if snippet else head)
        content = "\n\n".join(lines) or "(no results)"
        credits = (data.get("usage") or {}).get("credits")
        cost, cost_source = (
            (round(float(credits) * USD_PER_CREDIT, 6), "api")
            if isinstance(credits, (int, float)) else
            (CREDITS_PER_CALL[mode] * USD_PER_CREDIT, "static"))
        info = {"tool": "search", "mode": mode, "latency_s": latency,
                "n_results": len(data.get("results", [])),
                "cost_usd": cost, "cost_source": cost_source}
        if attempts > 1:
            info["retries"] = attempts - 1
        return content, info
    return execute


def tool(mode=DEFAULT_MODE):
    from . import Tool
    if mode not in CREDITS_PER_CALL:
        raise KeyError(f"unknown tavily search mode {mode!r}; "
                       f"available: {sorted(CREDITS_PER_CALL)}")
    return Tool(spec=SPEC, execute=make_execute(mode),
                config={"provider": "tavily", "endpoint": ENDPOINT,
                        "mode": mode, "max_results": MAX_RESULTS,
                        "max_chars_per_result": MAX_CHARS_PER_RESULT,
                        "cost_per_call_usd":
                            CREDITS_PER_CALL[mode] * USD_PER_CREDIT})
