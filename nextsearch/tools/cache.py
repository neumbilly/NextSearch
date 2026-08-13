"""Single-flight tool-result cache.

Search is deliberately NOT cached during evaluation: a persistent search cache
would quietly serve a stale world under a dated prompt, which is exactly the
failure the task-date layer exists to prevent. What does use this class is the
`fetch` tool's raw-response cache, with a one-week TTL — page reads tolerate a
week of staleness, and paging through a long page must re-slice the SAME
response or the window would drift under the model's feet.

The cache key is a sha256 over {tool name, tool config, normalized args} — the
same inputs that determine the API response. It is single-flight: concurrent
identical calls share ONE in-flight execution via a future, so a burst of the
same request pays once and reads one consistent world. Failures are never
cached; a transient API error must not become the permanent answer. Hits
replay the stored content with cost zeroed and `cached: True` in the info
dict, so cost accounting stays honest.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path

from .. import io
from . import Tool


class ToolCache:
    """Wrap() every tool that should hit this cache."""

    def __init__(self, path=None, ttl_s=None):
        """`ttl_s`: entries older than this re-execute. Records carry a
        timestamp, and with a TTL set, records without one count as expired.
        None never expires."""
        self.path = Path(path) if path else None
        self.ttl_s = ttl_s
        self._done = {}      # key -> {"content", "info", "ts"}
        self._inflight = {}  # key -> asyncio.Future (single-flight)
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            for rec in io.read_jsonl(self.path):
                if self._fresh(rec.get("ts")):
                    self._done[rec["key"]] = {"content": rec["content"],
                                              "info": rec["info"],
                                              "ts": rec.get("ts")}

    def _fresh(self, ts) -> bool:
        return self.ttl_s is None or (
            ts is not None and time.time() - ts < self.ttl_s)

    @staticmethod
    def key(tool, args) -> str:
        return hashlib.sha256(json.dumps(
            {"tool": tool.spec["name"], "config": tool.config, "args": args},
            sort_keys=True, ensure_ascii=False,
            default=str).encode()).hexdigest()

    def _put(self, key, tool, args, content, info):
        ts = time.time()
        self._done[key] = {"content": content, "info": info, "ts": ts}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(io.dump_line({"key": key, "ts": ts,
                                      "tool": tool.spec["name"],
                                      "args": args, "content": content,
                                      "info": info}) + "\n")

    def wrap(self, tool: Tool) -> Tool:
        async def execute(args):
            key = self.key(tool, args)
            hit = self._done.get(key)
            if hit is not None and not self._fresh(hit.get("ts")):
                hit = None
                self._done.pop(key, None)   # expired mid-process: re-execute
            if hit is not None:
                self.hits += 1
                return hit["content"], {**hit["info"], "cached": True,
                                        "cost_usd": 0.0, "latency_s": 0.0}
            if key in self._inflight:       # someone is already fetching this
                content, info = await asyncio.shield(self._inflight[key])
                self.hits += 1
                return content, {**info, "cached": True, "cost_usd": 0.0}
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut
            try:
                content, info = await tool.execute(args)
            except BaseException as e:
                fut.set_exception(e)
                # Errors are never cached; waiters see the same failure once.
                fut.exception()             # mark retrieved (suppress warning)
                self._inflight.pop(key, None)
                raise
            self.misses += 1
            self._put(key, tool, args, content, info)
            fut.set_result((content, info))
            self._inflight.pop(key, None)
            return content, info

        return Tool(spec=tool.spec, execute=execute,
                    config={**tool.config, "cache": True,
                            **({"cache_ttl_s": self.ttl_s}
                               if self.ttl_s else {})})

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self._done)}
