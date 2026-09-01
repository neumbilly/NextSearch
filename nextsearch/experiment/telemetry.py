"""Reusable telemetry extraction from rollout JSONL.

The harness already records everything an experiment needs — per-call usage,
latency, cost, stop reasons, tool metadata — inside each `Rollout`'s `timing`
and `meta`. This module reads those artifacts back and derives two views:

  * `episode_metrics(rollout, ...)` — one flat record per episode, covering
    behavior (turns, tool calls, parallelism), tokens (prompt / completion /
    reasoning), latency (model / tool / wall), throughput, and cost.
  * `aggregate(episodes, ...)` — run-level rates, percentiles, throughput, and
    cost totals over a list of those records.

Both are pure functions over plain dicts, so they work on any rollouts.jsonl —
live during a run or long after — and the same records feed the CLI (JSON and
CSV) and the live Colab viewer. Hardware/software context (GPU type, vLLM
version) is not in a rollout, so it is passed in and stamped onto every record.

The CLI (`nextsearch-telemetry`) reads one or more rollout files (or an eval
directory), joins any grade sidecars for judge cost, and emits the aggregate as
JSON plus an optional per-episode CSV.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Tool-result info flags that mean "this call did not actually execute", so it
# should not count as a real search/fetch call or contribute latency.
_NON_EXECUTED = ("deduped", "budget_exhausted", "turn_cap_skipped",
                 "not_executed", "wall_cut")


def _percentiles(values, qs=(0.5, 0.9, 0.99)):
    """Nearest-rank percentiles, no interpolation — every reported value is a
    real episode. Empty input yields None per quantile."""
    if not values:
        return [None] * len(qs)
    s = sorted(values)
    return [s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] for q in qs]


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _domain(url):
    try:
        host = urlsplit(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:  # noqa: BLE001 — a malformed url has no domain
        return None


def _as_dict(rollout):
    """Accept a Rollout dataclass or a raw JSONL dict."""
    if hasattr(rollout, "to_json"):
        return rollout.to_json()
    return rollout


def _tool_turns(messages):
    """Per assistant turn, how many tool calls it issued (only turns that made
    at least one call)."""
    return [len(m["tool_calls"]) for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")]


def _executed_tool_calls(tool_calls):
    """Tool-log entries that actually hit a backend (no dedupe/budget/cut
    flags)."""
    return [c for c in tool_calls
            if not any(c.get(flag) for flag in _NON_EXECUTED)]


def _fetch_urls(messages):
    """Every url the model passed to a `fetch` call, across the transcript."""
    urls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict) or fn.get("name") != "fetch":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                continue
            urls.extend(u for u in (args.get("urls") or []) if isinstance(u, str))
    return urls


def episode_metrics(rollout, grade=None, *, gpu=None, vllm_version=None,
                    checkpoint=None, gpu_hourly_usd=None) -> dict:
    """One flat telemetry record for a single episode.

    `rollout` is a Rollout or a rollouts.jsonl dict. `grade` is the matching
    grade-sidecar dict when available (its `judge_usage.cost` is the only
    source of judge cost). `gpu`, `vllm_version`, and `checkpoint` are stamped
    verbatim — they describe the serving run, not the rollout.
    """
    r = _as_dict(rollout)
    timing = r.get("timing") or {}
    meta = r.get("meta") or {}
    messages = r.get("messages") or []
    tool_calls = timing.get("tool_calls") or []

    stop_reason = meta.get("stop_reason")
    errored = r.get("error") is not None
    forced = meta.get("forced_answer") or {}
    final_answer = (stop_reason == "final") or bool(forced.get("ok"))

    # Behavior over the transcript.
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    turns = _tool_turns(messages)
    executed = _executed_tool_calls(tool_calls)
    n_search = sum(1 for c in executed if c.get("tool") == "search")
    n_fetch = sum(1 for c in executed if c.get("tool") == "fetch")
    parallel_turns = sum(1 for n in turns if n > 1)
    fetch_urls = _fetch_urls(messages)
    domains = {d for d in (_domain(u) for u in fetch_urls) if d}

    # Reasoning: characters are always countable from stored thinking; tokens
    # only when the server reported them in usage details.
    reasoning_chars = sum(len(m.get("reasoning_content") or "")
                          for m in assistant_msgs)
    reasoning_tokens = 0
    have_reasoning_tokens = False
    for u in timing.get("usage") or []:
        details = (u or {}).get("completion_tokens_details") or {}
        if details.get("reasoning_tokens") is not None:
            reasoning_tokens += details["reasoning_tokens"]
            have_reasoning_tokens = True

    prompt_tokens = timing.get("prompt_tokens")
    completion_tokens = timing.get("completion_tokens")
    model_latency = timing.get("llm_s")
    lat = [c.get("latency_s") for c in tool_calls]
    tool_latency = round(sum(l for l in lat if l is not None), 3) \
        if lat and all(l is not None for l in lat) else None
    wall_s = timing.get("wall_s")
    # Effective output throughput: completion tokens over MODEL time, not wall
    # time — wall includes tool latency the model was not generating during.
    output_tps = (round(completion_tokens / model_latency, 2)
                  if completion_tokens and model_latency else None)

    def _tool_cost(name):
        vals = [c.get("cost_usd") for c in tool_calls if c.get("tool") == name]
        return round(sum(v for v in vals if v is not None), 6) if vals else 0.0

    search_usd = _tool_cost("search")
    fetch_usd = _tool_cost("fetch")
    judge_usd = ((grade or {}).get("judge_usage") or {}).get("cost")

    gpu_usd = None
    if gpu_hourly_usd is not None and wall_s is not None:
        gpu_usd = round(gpu_hourly_usd * wall_s / 3600.0, 6)

    return {
        "episode_id": meta.get("episode_id") or r.get("sample_id"),
        "benchmark": meta.get("bench"),
        "task_id": r.get("sample_id"),
        "model": r.get("model"),
        "checkpoint": checkpoint or r.get("model"),
        "gpu": gpu,
        "vllm_version": vllm_version,
        "sampling": r.get("sampling") or {},
        "stop_reason": stop_reason,
        "error": errored,
        "truncated": bool(r.get("truncated")),
        "final_answer": final_answer,
        "n_turns": r.get("n_turns"),
        "n_assistant_messages": len(assistant_msgs),
        "reasoning_chars": reasoning_chars,
        "reasoning_tokens": reasoning_tokens if have_reasoning_tokens else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": r.get("n_tokens"),
        "n_search_calls": n_search,
        "n_fetch_calls": n_fetch,
        "n_tool_calls": n_search + n_fetch,
        "max_calls_per_turn": max(turns) if turns else 0,
        "mean_calls_per_turn": round(sum(turns) / len(turns), 3) if turns else 0,
        "n_parallel_turns": parallel_turns,
        "has_parallel_calls": parallel_turns > 0,
        "n_unique_fetch_urls": len(set(fetch_urls)),
        "n_unique_domains": len(domains),
        "model_latency_s": model_latency,
        "tool_latency_s": tool_latency,
        "wall_s": wall_s,
        "started_at": timing.get("started_at"),
        "ended_at": timing.get("ended_at"),
        "output_tps": output_tps,
        "model_usd": timing.get("llm_cost_usd"),
        "search_usd": search_usd,
        "fetch_usd": fetch_usd,
        "judge_usd": judge_usd,
    }


def _rate(count, total):
    return round(count / total, 4) if total else None


def aggregate(episodes, *, gpu_hourly_usd=None) -> dict:
    """Run-level summary over a list of `episode_metrics` records.

    Rates are over all episodes. Latency percentiles and throughput cover only
    episodes that produced an answer (errored episodes are aborted work, not
    answer latency). `episodes_per_hour` and the aggregate GPU-cost estimate
    use the run's real makespan (max ended − min started), because episodes
    overlap under concurrency, so summing per-episode wall time would be wrong.
    """
    n = len(episodes)
    if not n:
        return {"n_episodes": 0}
    answered = [e for e in episodes if not e["error"]]

    walls = [e["wall_s"] for e in answered if e["wall_s"] is not None]
    p50, p90, p99 = _percentiles(walls)
    turns = [e["n_turns"] for e in episodes if e["n_turns"] is not None]
    tp50, tp90 = _percentiles(turns, (0.5, 0.9))

    # Parallel-call frequency at the episode level, which is unambiguous: what
    # fraction of episodes issued more than one tool call in a single turn.
    parallel_episode_rate = _rate(
        sum(1 for e in episodes if e["has_parallel_calls"]), n)

    completion = [e["completion_tokens"] for e in answered
                  if e["completion_tokens"]]
    model_time = [e["model_latency_s"] for e in answered
                  if e["model_latency_s"]]
    agg_tps = (round(sum(completion) / sum(model_time), 2)
               if completion and model_time and sum(model_time) > 0 else None)
    single_tps = _percentiles(
        [e["output_tps"] for e in answered if e["output_tps"]], (0.5,))[0]

    started = [e["started_at"] for e in episodes if e["started_at"] is not None]
    ended = [e["ended_at"] for e in episodes if e["ended_at"] is not None]
    makespan_s = (max(ended) - min(started)) if started and ended else None
    episodes_per_hour = (round(n / (makespan_s / 3600.0), 2)
                         if makespan_s and makespan_s > 0 else None)

    def _sum(field):
        vals = [e[field] for e in episodes if e.get(field) is not None]
        return round(sum(vals), 6) if vals else 0.0

    search_total = _sum("search_usd")
    fetch_total = _sum("fetch_usd")

    est_gpu_per_episode = None
    if gpu_hourly_usd is not None and makespan_s:
        est_gpu_per_episode = round(gpu_hourly_usd * makespan_s / 3600.0 / n, 6)

    return {
        "n_episodes": n,
        "n_answered": len(answered),
        "success_rate": _rate(
            sum(1 for e in episodes if e["final_answer"] and not e["error"]), n),
        "parsed_tool_call_rate": _rate(
            sum(1 for e in episodes if e["n_tool_calls"] > 0), n),
        "final_answer_rate": _rate(
            sum(1 for e in episodes if e["final_answer"]), n),
        "truncation_rate": _rate(sum(1 for e in episodes if e["truncated"]), n),
        "error_rate": _rate(sum(1 for e in episodes if e["error"]), n),
        "mean_turns": _mean([e["n_turns"] for e in episodes]),
        "p90_turns": tp90,
        "mean_reasoning_tokens": _mean(
            [e["reasoning_tokens"] for e in episodes]),
        "mean_reasoning_chars": _mean([e["reasoning_chars"] for e in episodes]),
        "mean_output_tokens": _mean(
            [e["completion_tokens"] for e in episodes]),
        "mean_search_calls": _mean([e["n_search_calls"] for e in episodes]),
        "mean_fetch_calls": _mean([e["n_fetch_calls"] for e in episodes]),
        "parallel_call_rate": parallel_episode_rate,
        "latency_p50_s": p50, "latency_p90_s": p90, "latency_p99_s": p99,
        "latency_max_s": max(walls) if walls else None,
        "aggregate_tps": agg_tps,
        "single_episode_tps_p50": single_tps,
        "makespan_s": round(makespan_s, 2) if makespan_s else None,
        "episodes_per_hour": episodes_per_hour,
        "est_gpu_usd_per_episode": est_gpu_per_episode,
        "total_search_usd": round(search_total + fetch_total, 6),
        "total_model_usd": _sum("model_usd"),
        "total_judge_usd": _sum("judge_usd"),
    }


def load_grades(rollouts_path) -> dict:
    """Map episode_id -> grade dict from the grade sidecars beside a
    rollouts.jsonl (…/<bench>/<model>/grades/*.jsonl). Empty when none exist;
    judge cost is simply absent then."""
    grades_dir = Path(rollouts_path).parent / "grades"
    out = {}
    if not grades_dir.exists():
        return out
    for gp in sorted(grades_dir.glob("*.jsonl")):
        for line in Path(gp).read_text().split("\n"):
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            eid = g.get("episode_id")
            if eid is not None:
                out[str(eid)] = g
    return out


def _accepted(rollouts_path):
    """The accepted rollout per sample (latest line wins), matching the
    harness's own resume semantics."""
    from ..io import read_jsonl
    accepted = {}
    for d in read_jsonl(rollouts_path):
        accepted[d.get("sample_id")] = d
    return list(accepted.values())


def rollout_telemetry(rollouts_path, *, gpu=None, vllm_version=None,
                      checkpoint=None, gpu_hourly_usd=None, join_grades=True):
    """Per-episode records for one rollouts.jsonl, joining grade sidecars for
    judge cost when present."""
    grades = load_grades(rollouts_path) if join_grades else {}
    out = []
    for r in _accepted(rollouts_path):
        eid = str((r.get("meta") or {}).get("episode_id") or r.get("sample_id"))
        out.append(episode_metrics(
            r, grade=grades.get(eid), gpu=gpu, vllm_version=vllm_version,
            checkpoint=checkpoint, gpu_hourly_usd=gpu_hourly_usd))
    return out


def _find_rollout_files(paths):
    """Expand files and directories into a de-duplicated, ordered list of
    rollouts.jsonl paths. A directory is searched recursively."""
    found = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found.extend(sorted(p.rglob("rollouts.jsonl")))
        elif p.name.endswith(".jsonl"):
            found.append(p)
        else:
            raise SystemExit(f"not a .jsonl file or directory: {p}")
    seen, ordered = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    # Serialize dict/list cells (sampling) as JSON so the CSV stays flat.
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (dict, list)) else v)
                        for k, v in row.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="nextsearch-telemetry",
        description="Extract experiment telemetry from rollout JSONL. Prints "
                    "the run-level aggregate as JSON; optionally writes a "
                    "per-episode CSV and a full JSON (episodes + aggregate).")
    ap.add_argument("paths", nargs="+",
                    help="rollouts.jsonl files, or directories searched "
                         "recursively for them (e.g. an eval run directory)")
    ap.add_argument("--gpu", default=None,
                    help="GPU label to stamp on every record, e.g. 'L4', "
                         "'A100-40GB'")
    ap.add_argument("--vllm-version", default=None,
                    help="vLLM version to stamp on every record")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint/adapter label overriding the rollout's "
                         "model name (for later SFT/RL stages)")
    ap.add_argument("--gpu-hourly-usd", type=float, default=None,
                    help="GPU $/hour, used to estimate GPU dollars per episode")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write {config, episodes, aggregate} to this JSON file")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="write the per-episode records to this CSV file")
    ap.add_argument("--no-grades", action="store_true",
                    help="do not join grade sidecars (skip judge cost)")
    args = ap.parse_args(argv)

    files = _find_rollout_files(args.paths)
    if not files:
        raise SystemExit("no rollouts.jsonl found under the given paths")
    episodes = []
    for f in files:
        episodes.extend(rollout_telemetry(
            f, gpu=args.gpu, vllm_version=args.vllm_version,
            checkpoint=args.checkpoint, gpu_hourly_usd=args.gpu_hourly_usd,
            join_grades=not args.no_grades))
    summary = aggregate(episodes, gpu_hourly_usd=args.gpu_hourly_usd)
    config = {"sources": [str(f) for f in files], "gpu": args.gpu,
              "vllm_version": args.vllm_version, "checkpoint": args.checkpoint,
              "gpu_hourly_usd": args.gpu_hourly_usd}

    if args.csv:
        _write_csv(args.csv, episodes)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"config": config, "aggregate": summary, "episodes": episodes},
            indent=2, ensure_ascii=False) + "\n")
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
