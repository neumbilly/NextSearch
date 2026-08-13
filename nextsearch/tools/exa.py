"""Exa /search as an alternative `search` backend.

Same model-facing role as `parallel.py` — one call is one web search, numbered
results with title/url/snippet — but the request interface follows Exa's own
guidance: a single natural-language query phrased as a complete request. Exa
does its own query engineering, and `site:` operators are explicitly
discouraged (domain filtering is an API parameter we deliberately do not
expose to the model). Snippets are Exa highlights — LLM-selected relevant
passages, the closest analogue to Parallel's excerpts — falling back to raw
text when highlights are absent.

The tool spec never names the provider: backend choice is harness
configuration the model cannot see.

Modes map to Exa's search `type`: auto, fast, deep. Per-call cost comes from
the response's own `costDollars.total` where present; the static table below
is only a fallback, and `info["cost_source"]` records which was used.
"""

import os
import time

from ..types import tool_spec

ENDPOINT = "https://api.exa.ai/search"
DEFAULT_MODE = "auto"
MAX_RESULTS = 10
MAX_CHARS_PER_RESULT = 1500
COST_PER_CALL_USD = {"auto": 0.007, "fast": 0.005, "deep": 0.012}

# The first and last sentences are shared verbatim with the Parallel spec —
# that is the backend-neutral surface. Only the middle, which describes query
# phrasing, is per-provider, and it is written from Exa's own guidance.
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
                "description": "A self-contained natural-language description "
                    "of what to find. Phrase it as a complete request, not "
                    "bare keywords (e.g. 'the university where the physicist "
                    "who proposed X taught in the 1930s'). NEVER use search "
                    "operators such as site:."},
        },
        "required": ["query"],
    },
)


async def _search_api(body, key):
    import httpx
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(ENDPOINT, json=body,
                                 headers={"x-api-key": key})
        resp.raise_for_status()
        return resp.json()


def make_execute(mode):
    async def execute(args):
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise RuntimeError("EXA_API_KEY not set")
        query = (args.get("query") or "").strip()
        if not query:
            raise ValueError("search requires a non-empty 'query' "
                             "describing what to find")
        body = {"query": query, "type": mode, "numResults": MAX_RESULTS,
                "contents": {"highlights":
                             {"maxCharacters": MAX_CHARS_PER_RESULT}}}
        from . import with_backoff
        t0 = time.monotonic()
        data, attempts = await with_backoff(_search_api, body, key)
        latency = round(time.monotonic() - t0, 3)  # includes backoff sleeps
        lines = []
        for i, r in enumerate(data.get("results", [])[:MAX_RESULTS], 1):
            head = f"[{i}] {r.get('title') or '(untitled)'} — {r.get('url', '')}"
            snippet = "\n".join(h.strip() for h in (r.get("highlights") or [])
                                if h.strip())
            snippet = (snippet
                       or (r.get("text") or "").strip())[:MAX_CHARS_PER_RESULT]
            lines.append(f"{head}\n{snippet}" if snippet else head)
        content = "\n\n".join(lines) or "(no results)"
        api_cost = (data.get("costDollars") or {}).get("total")
        cost, cost_source = (
            (round(float(api_cost), 6), "api")
            if isinstance(api_cost, (int, float)) else
            (COST_PER_CALL_USD[mode], "static"))
        info = {"tool": "search", "mode": mode, "latency_s": latency,
                "n_results": len(data.get("results", [])),
                "cost_usd": cost, "cost_source": cost_source}
        if attempts > 1:
            info["retries"] = attempts - 1
        return content, info
    return execute


def tool(mode=DEFAULT_MODE):
    from . import Tool
    if mode not in COST_PER_CALL_USD:
        raise KeyError(f"unknown exa search mode {mode!r}; "
                       f"available: {sorted(COST_PER_CALL_USD)}")
    return Tool(spec=SPEC, execute=make_execute(mode),
                config={"provider": "exa", "endpoint": ENDPOINT,
                        "mode": mode, "max_results": MAX_RESULTS,
                        "max_chars_per_result": MAX_CHARS_PER_RESULT,
                        "cost_per_call_usd": COST_PER_CALL_USD[mode]})
