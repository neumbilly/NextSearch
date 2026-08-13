"""The Parallel Extract API as the `fetch` tool: read specific pages.

One call reads a set of URLs against an objective and returns the relevant
content as markdown excerpts, with title and publish date where available.
Excerpts rather than whole pages, on purpose: full page dumps blow up a small
model's context, and objective-scoped excerpts keep results the same shape and
size class as `search` results. JavaScript-heavy pages and PDFs are handled
server-side.

Adding this tool alongside `search` was one of the larger harness wins we
measured — reading a selected page replaces several rounds of re-searching,
which improves set-answer accuracy and lowers cost at the same time.

**Windowing and paging.** Each result shows a window of its extracted content
with a truncation marker carrying the next offset; calling again with that
`offset` (and the identical urls and objective) re-slices the SAME response.
That requires the raw API response to be stable, so it is cached on disk with
a one-week TTL. Slicing sits ABOVE the cache — `offset` is not part of the
raw-cache key — so every window size reads the same bytes, and a repeated
fetch of a hot page is billed once. Cost accounting stays honest: hits report
`cached: True` and cost 0.

A 4,000-character window gave the best aggregate score and latency across a
4k-16k sweep. Larger windows improved recall on some comprehensive set-answer
tasks but increased freshness errors and produced long latency tails.
"""

import os
import time

from ..paths import CACHE_DIR
from ..types import tool_spec

ENDPOINT = "https://api.parallel.ai/v1/extract"
MAX_URLS = 3
WINDOW_CHARS = 4000           # fetch is targeted; more than search's 1500
CACHE_TTL_S = 7 * 24 * 3600   # the freshness/cost tradeoff for page reads
COST_PER_URL_USD = 0.001


def _spec(window):
    return tool_spec(
        "fetch",
        "Fetches specific web pages and returns the content relevant to your "
        "objective as markdown excerpts. Use it to read a page whose search "
        "excerpt was truncated or ambiguous, or to verify a critical fact "
        "against the primary source. Handles JavaScript-heavy pages and PDFs. "
        f"Long content is windowed: each result shows at most {window} "
        "characters per call, and a truncated result names the offset to pass "
        "to read the next chunk. Only offset advances the window — rewording "
        "the objective restarts a fresh extract from the beginning.",
        {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "maxItems": MAX_URLS,
                    "description": f"Up to {MAX_URLS} URLs to fetch, e.g. from "
                        "earlier search results."},
                "objective": {
                    "type": "string",
                    "description": "What you want to learn from these pages — "
                        "excerpts are selected to match it."},
                "offset": {
                    "type": "integer", "minimum": 0,
                    "description": "Character offset into each page's extracted "
                        f"content (default 0); each call returns the next "
                        f"{window} characters from here. When a result is "
                        "marked truncated, call fetch again with the SAME urls "
                        "and objective and the offset from the marker to read "
                        "the next chunk — served from cache, no extra cost."},
            },
            "required": ["urls", "objective"],
        },
    )


SPEC = _spec(WINDOW_CHARS)


async def _extract_api(args):
    """The raw Extract call. The content is the response's results and errors
    as JSON, not display text — the cacheable unit that paging re-slices."""
    import httpx
    key = os.environ.get("PARALLEL_API_KEY")
    if not key:
        raise RuntimeError("PARALLEL_API_KEY not set")
    body = {"urls": args["urls"], "objective": args.get("objective") or ""}

    async def call():
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(ENDPOINT, json=body,
                                     headers={"x-api-key": key})
            resp.raise_for_status()
            return resp.json()

    from . import with_backoff
    t0 = time.monotonic()
    data, _attempts = await with_backoff(call)
    return ({"results": data.get("results") or [],
             "errors": data.get("errors") or []},
            {"latency_s": round(time.monotonic() - t0, 3),
             "cost_usd": round(COST_PER_URL_USD * len(args["urls"]), 6)})


_CACHE = None       # module singleton: single-flight across concurrent episodes
_RAW = None         # the cache-wrapped raw extract


def _cached_extract():
    global _CACHE, _RAW
    if _RAW is None:
        from . import Tool
        from .cache import ToolCache
        _CACHE = ToolCache(path=CACHE_DIR / "fetch.jsonl", ttl_s=CACHE_TTL_S)
        _RAW = _CACHE.wrap(Tool(spec={"name": "fetch_extract"},
                                execute=_extract_api,
                                config={"endpoint": ENDPOINT}))
    return _RAW


def _text(r) -> str:
    excerpts = r.get("excerpts")
    if isinstance(excerpts, list):
        return "\n".join(e.strip() for e in excerpts if e.strip())
    return str(excerpts or "").strip()


async def execute(args, window=WINDOW_CHARS):
    urls = [u for u in (args.get("urls") or []) if u][:MAX_URLS]
    if not urls:
        raise ValueError("fetch requires at least one url — pass a url from "
                         "an earlier search result")
    offset = max(0, int(args.get("offset") or 0))
    raw, info = await _cached_extract().execute(
        {"urls": urls, "objective": args.get("objective") or ""})
    lines = []
    # An offset is only meaningful against the extract that produced its
    # truncation marker. If this {urls, objective} pair has no cached extract,
    # the offset came from a DIFFERENT call — honoring it would page silently
    # wrong content, or land past the end and waste the turn on nothing. Serve
    # from the start and say why, which salvages the turn and teaches the
    # contract at the same time.
    offset_ignored = bool(offset) and not info.get("cached")
    if offset_ignored:
        lines.append(f"[note: offset={offset} ignored — this exact "
                     "urls+objective pair has no earlier fetch to continue "
                     "(content selection is objective-specific), so the "
                     "extract starts fresh. Showing it from the beginning; "
                     "to page an earlier result, repeat its exact urls and "
                     "objective.]")
        offset = 0
    today = time.strftime("%Y-%m-%d", time.gmtime())
    for i, r in enumerate(raw["results"], 1):
        head = f"[{i}] {r.get('title') or '(untitled)'} — {r.get('url', '')}"
        # Extraction services report the CRAWL date as the publish date for
        # wiki-class pages. A "published today" stamp is almost always the
        # crawl, and on as-of questions it is poison — so a same-day stamp is
        # dropped rather than shown.
        if r.get("publish_date") \
                and not str(r["publish_date"]).startswith(today):
            head += f" (published {r['publish_date']})"
        text = _text(r)
        if not text:
            lines.append(head)
            continue
        chunk = text[offset:offset + window]
        if not chunk:
            lines.append(f"{head}\n(no content at offset {offset} — this "
                         f"page's extract is {len(text)} chars)")
            continue
        end = offset + len(chunk)
        # The marker is what the model reads at decision time, so it must
        # spell out the exact-repeat contract. Without that, continuation
        # calls drift to a new {urls, objective} pair and page a fresh,
        # re-billed, unstable extract instead of this cached one.
        marker = (f"\n[truncated: chars {offset}–{end} of {len(text)} shown "
                  f"— call fetch again with the same urls and objective and "
                  f"offset={end} to continue]"
                  if end < len(text) else "")
        lines.append(f"{head}\n{chunk}{marker}")
    for err in raw["errors"]:
        lines.append(f"[error] {err.get('url', '')}: "
                     f"{err.get('message') or err.get('error') or err}")
    content = "\n\n".join(lines) or "(no content)"
    out = {"tool": "fetch", "latency_s": info.get("latency_s", 0.0),
           "n_urls": len(urls), "n_results": len(raw["results"]),
           "cost_usd": info.get("cost_usd", 0.0)}
    if info.get("cached"):
        out["cached"] = True
    if offset:
        out["offset"] = offset
    if offset_ignored:
        out["offset_ignored"] = True
    return content, out


def tool(window_chars=WINDOW_CHARS):
    """`window_chars` changes the spec text (the model must know its window),
    the slicing, and the frozen tool config — but NOT the raw-response cache,
    which sits below the slicing, so every window reads the same bytes."""
    from . import Tool

    async def _execute(args):
        return await execute(args, window=window_chars)

    return Tool(spec=_spec(window_chars), execute=_execute,
                config={"provider": "parallel", "endpoint": ENDPOINT,
                        "max_urls": MAX_URLS,
                        "window_chars": window_chars,
                        "cache_ttl_s": CACHE_TTL_S,
                        "cost_per_url_usd": COST_PER_URL_USD})
