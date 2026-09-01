"""Full episode traces: save the raw rollout and render the whole conversation.

A stop reason and a turn count do not prove an agent called its tools
correctly. To trust a smoke test you want to *see* the trace: every assistant
turn, every tool call with its parsed arguments, every tool result, any
reasoning, and the final answer. This module does two things:

  * `save_trace` writes the raw rollout JSON so a single episode can be reopened
    in full later; `save_as_rollouts` also drops it in a `rollouts.jsonl` so the
    live viewer and the telemetry CLI pick it up like any other run.
  * `format_trace` / `print_trace` render the conversation as readable text,
    with a per-tool call summary so "did it call `search`?" is answered at a
    glance.

Pure functions over a Rollout or a raw rollout dict — no plotting or notebook
dependency — so they work in a script, a cell, or a log.
"""

import json
from pathlib import Path

from ..io import dump_line


def _as_dict(rollout):
    return rollout.to_json() if hasattr(rollout, "to_json") else rollout


def _short(text, limit):
    text = "" if text is None else str(text)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def _parsed_args(tc):
    """Parsed tool-call arguments, tolerant of a malformed payload (which is
    itself worth seeing in a trace)."""
    fn = tc.get("function") if isinstance(tc, dict) else None
    raw = fn.get("arguments") if isinstance(fn, dict) else None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:  # noqa: BLE001 — show the raw string when it will not parse
        return raw


def tool_call_summary(rollout) -> dict:
    """Counts that answer 'did tools get called, and how': per-tool call counts
    from the assistant turns, plus turns that issued more than one call."""
    r = _as_dict(rollout)
    by_tool, parallel_turns, total = {}, 0, 0
    for m in r.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if len(calls) > 1:
            parallel_turns += 1
        for tc in calls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else "<malformed>"
            by_tool[name] = by_tool.get(name, 0) + 1
            total += 1
    return {"by_tool": by_tool, "total_calls": total,
            "parallel_turns": parallel_turns}


def format_trace(rollout, *, result_chars=800, answer_chars=None,
                 system_chars=300, reasoning_chars=600) -> str:
    """Render one episode as a readable transcript.

    Truncation keeps a long research trace legible: tool results and the system
    prompt are clipped by default, while the final answer is shown in full
    (`answer_chars=None`). Set any limit to `None` to show that part whole.
    """
    r = _as_dict(rollout)
    timing = r.get("timing") or {}
    meta = r.get("meta") or {}
    summary = tool_call_summary(r)
    out = []

    out.append("=" * 78)
    out.append(f"episode {meta.get('episode_id') or r.get('sample_id')}  "
               f"[{meta.get('bench')}]  model={r.get('model')}")
    out.append(f"stop={meta.get('stop_reason')}  truncated={r.get('truncated')}"
               f"  turns={r.get('n_turns')}  tokens={r.get('n_tokens')}")
    out.append(f"wall={timing.get('wall_s')}s  llm={timing.get('llm_s')}s  "
               f"model_$={timing.get('llm_cost_usd')}  "
               f"tool_$={timing.get('tool_cost_usd')}")
    out.append(f"tool calls: {summary['by_tool'] or '{}'} "
               f"(total {summary['total_calls']}, "
               f"{summary['parallel_turns']} parallel turns)")
    if r.get("error"):
        out.append(f"ERROR: {r['error']}")
    out.append("=" * 78)

    tool_by_id = {m.get("tool_call_id"): m for m in (r.get("messages") or [])
                  if m.get("role") == "tool"}
    turn = 0
    for m in r.get("messages") or []:
        role = m.get("role")
        if role == "system":
            out.append(f"\n[system]\n{_short(m.get('content'), system_chars)}")
        elif role == "user":
            # A harness note is injected as a user message; label it as such.
            content = m.get("content") or ""
            tag = "user/harness-note" if content.startswith("[harness note") \
                else "user"
            out.append(f"\n[{tag}]\n{_short(content, result_chars)}")
        elif role == "assistant":
            turn += 1
            out.append(f"\n[assistant · turn {turn}]")
            if m.get("reasoning_content"):
                out.append(f"  (reasoning) "
                           f"{_short(m['reasoning_content'], reasoning_chars)}")
            if (m.get("content") or "").strip():
                out.append(f"  {_short(m['content'], answer_chars)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else {}
                name = fn.get("name") if isinstance(fn, dict) else "<malformed>"
                args = _parsed_args(tc)
                out.append(f"  → CALL {name}({json.dumps(args, ensure_ascii=False)}"
                           f")  id={tc.get('id') if isinstance(tc, dict) else None}")
        elif role == "tool":
            out.append(f"    ← result id={m.get('tool_call_id')}: "
                       f"{_short(m.get('content'), result_chars)}")
    out.append("")
    return "\n".join(out)


def print_trace(rollout, **kwargs):
    print(format_trace(rollout, **kwargs))


def save_trace(rollout, path) -> Path:
    """Write the raw rollout JSON (the complete, re-loadable trace) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_dict(rollout), indent=2, default=str) + "\n")
    return path


def save_as_rollouts(rollout, path) -> Path:
    """Append the rollout to a `rollouts.jsonl` so the viewer and
    `nextsearch-telemetry` treat this smoke episode like any run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(dump_line(_as_dict(rollout)) + "\n")
    return path


def load_trace(path):
    """Read a raw trace back as a plain dict."""
    return json.loads(Path(path).read_text())
