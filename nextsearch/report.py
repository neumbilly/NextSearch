"""Aggregate one evaluation into summary.json and a markdown table.

Result entries are flat and carry their own provenance, so multiple graders
and repeated runs never force a schema change.

Three things are reported next to every score, deliberately. A confidence
interval, because these benchmarks are small enough that a 3-point difference
is usually noise. Cost, split into model, search, subagent, and grader
components. And latency percentiles rather than a mean, because a research
agent's tail is what sets a job's wall clock.
"""

import json
import math
import random
import time
from pathlib import Path

from .grading import TABLE_FORMAT_FAILS, accepted_digest
from .harness import accepted_attempts
from .manifest import eval_dir, load as load_manifest

SCHEMA_VERSION = 1


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of the mean. Deterministic."""
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return [round(lo, 4), round(hi, 4)]


def _read_grades(path):
    # split("\n") rather than splitlines(): JSON strings legally carry raw
    # U+2028/U+2029, which judges emit when quoting web text, and splitlines()
    # would tear those records in half.
    return [json.loads(line) for line in Path(path).read_text().split("\n")
            if line.strip()]


def _known_sum(values):
    """(sum of known components, count of unknown ones). Unknowns count as $0
    in the sum, so the total is a LOWER BOUND whenever the count is non-zero —
    and the count travels with it, so nothing is silently fabricated. An
    episode that genuinely spent nothing contributes 0.0 and counts as known."""
    known = [v for v in values if v is not None]
    return round(sum(known), 6), len(values) - len(known)


def _cost_breakdown(rollouts, grades):
    """The components an evaluation actually spends money on, taken from
    write-time snapshots and never re-priced at report time.

    Grader cost attributes each verdict's judge call to this run even when it
    came from the cache: it is the cost of the verdicts, not of this
    invocation's marginal spend.
    """
    def _tool_split(calls):
        """One episode's tool spend as (search $, subagent-model $, whether
        the episode has unreported calls).

        A plain web-tool call is all search. A nested research call reports
        its search share as `search_usd`, the remainder being the subagent's
        own model spend. Known calls always count, so the sums stay tight
        lower bounds; a call with unreported cost flags the episode rather
        than discarding its siblings, because at high turn and subagent depths
        nearly every episode has one flaky call and episode-level poisoning
        would zero the columns entirely.
        """
        search = subagent = 0.0
        unknown = False
        for c in calls:
            if "search_usd" in c:
                s, total = c.get("search_usd"), c.get("cost_usd")
                if s is None or total is None:
                    unknown = True
                    continue
                search += s
                subagent += total - s
            else:
                v = c.get("cost_usd")
                if v is None:
                    unknown = True
                    continue
                search += v
        return round(search, 6), round(subagent, 6), unknown

    model, model_missing = _known_sum([
        r.timing.get("llm_cost_usd") if r.timing.get("usage") else 0.0
        for r in rollouts])                     # no LLM calls -> genuinely $0
    splits = [_tool_split(r.timing.get("tool_calls") or []) for r in rollouts]
    search = round(sum(s for s, _, _ in splits), 6)
    subagent = round(sum(g for _, g, _ in splits), 6)
    search_missing = subagent_missing = sum(1 for _, _, u in splits if u)
    grader, grader_missing = _known_sum([
        (g.get("judge_usage") or {}).get("cost") if g.get("judge_usage")
        else 0.0
        for g in grades])                       # no answer -> judge never called
    return {
        "model_usd": model,
        "search_usd": search,
        "subagent_usd": subagent,
        # A nested research call carries its own search count; a plain web-tool
        # call counts 1; a budget-skipped call executed nothing.
        "n_search_calls": sum(
            0 if c.get("budget_exhausted") else c.get("n_search_calls", 1)
            for r in rollouts for c in (r.timing.get("tool_calls") or [])),
        "grader_usd": grader,
        "total_usd": round(model + search + subagent + grader, 6),
        "missing": {"model": model_missing, "search": search_missing,
                    "subagent": subagent_missing, "grader": grader_missing},
    }


def percentiles(values, qs=(0.5, 0.9, 0.99)):
    """Nearest-rank percentiles, inclusive, with NO interpolation. Every
    reported value is an episode that actually happened. Empty input gives
    None per quantile, never 0."""
    if not values:
        return [None] * len(qs)
    s = sorted(values)
    return [s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] for q in qs]


def _r(v, nd=2):
    return None if v is None else round(v, nd)


def _tool_seconds(tool_calls):
    """Total per-call tool latency, or None if any call never reported one —
    the remainder would otherwise land silently in the model's column."""
    lat = [c.get("latency_s") for c in tool_calls]
    if any(v is None for v in lat):
        return None
    return round(sum(lat), 3)


def _unqueued_wall(rollout):
    """One episode's latency at unbounded subagent capacity, or None.

    A turn blocks on the SLOWEST call in its parallel batch, so the
    capacity-free episode is model time plus, per turn, the maximum service
    time in that turn's batch. Service time is the nested episode's own
    `sub_wall_s`; the call's `latency_s` is queue plus service, and the gap
    between them is the queue.

    Returns None when an episode has research calls that never recorded
    `sub_wall_s` — a partial reconstruction would understate the queue and
    read as a real speedup. Episodes with no research calls have no nested
    queue, so their `wall_s` is returned unchanged.
    """
    t = rollout.timing or {}
    calls = t.get("tool_calls") or []
    research = [c for c in calls if c.get("tool") == "research"]
    if not research:
        return t.get("wall_s")
    if any(c.get("sub_wall_s") is None for c in research):
        return None
    # Batch the calls back into turns from the transcript: one assistant
    # message with tool_calls is one turn's parallel batch, in call order.
    sizes = [len(m["tool_calls"]) for m in (rollout.messages or [])
             if m.get("role") == "assistant" and m.get("tool_calls")]
    total, i = 0.0, 0
    for n in sizes:
        batch, i = calls[i:i + n], i + n
        if batch:
            total += max((c.get("sub_wall_s") if c.get("tool") == "research"
                          else c.get("latency_s")) or 0.0 for c in batch)
    llm = t.get("llm_s")
    return round(total + (llm or 0.0), 3) if llm is not None else None


def _latency_block(rollouts):
    """End-to-end wall time over accepted episodes that produced an answer.

    Errored episodes are aborted work rather than answer latency, so they are
    excluded and counted. TRUNCATED episodes stay in — they really did take
    that long. `wall_s` is measured inside the run-global semaphore, so it
    excludes time an episode waited to start, but it still carries whatever
    provider-side queueing the run's concurrency provoked.

    **`wall_s` is not capacity-free under the orchestrated harness.** A parent
    sits inside a `research` call while its subagent waits for a slot on the
    nested semaphore, so that queue lands inside the parent's own `wall_s` and
    can dominate it. That makes an orchestrated wall number incomparable to a
    solo one, or even to the same configuration at a different subagent
    concurrency. Hence `unqueued_p50_s`: the same episode's latency at
    unbounded subagent capacity, which is what compares across harnesses and
    what belongs in any external latency claim.

    Percentiles rather than a mean, because the p99/p50 ratio on real cells
    runs several-fold and a mean describes neither the typical episode nor the
    stragglers that set a run's wall clock. At small n the p99 IS the max, and
    `n` rides along so the reader can see that.
    """
    answered = [r for r in rollouts if not r.error]
    walls, tool, non_tool, llm, tps = [], [], [], [], []
    out_tokens, started, ended, unqueued = 0, [], [], []
    for r in answered:
        t = r.timing or {}
        w = t.get("wall_s")
        if w is None:
            continue
        walls.append(w)
        uq = _unqueued_wall(r)
        if uq is not None:
            unqueued.append(uq)
        if t.get("llm_s") is not None:
            llm.append(t["llm_s"])
        if t.get("started_at") is not None:
            started.append(t["started_at"])
        if t.get("ended_at") is not None:
            ended.append(t["ended_at"])
        ts = _tool_seconds(t.get("tool_calls") or [])
        if ts is None:
            continue
        tool.append(ts)
        nt = max(w - ts, 1e-6)          # model plus harness overhead
        non_tool.append(round(nt, 3))
        ct = t.get("completion_tokens")
        if ct:
            out_tokens += ct
            tps.append(round(ct / nt, 2))
    p50, p90, p99 = percentiles(walls)
    return {
        "n": len(walls),
        "excluded_errors": len(rollouts) - len(answered),
        "missing": len(answered) - len(walls),   # wall_s never recorded
        "p50_s": _r(p50), "p90_s": _r(p90), "p99_s": _r(p99),
        "max_s": _r(max(walls)) if walls else None,
        "mean_s": _r(sum(walls) / len(walls)) if walls else None,
        "total_s": _r(sum(walls), 1) if walls else None,
        # n_unqueued below n means some episodes could not be reconstructed,
        # so that percentile is on a subset and says so rather than silently
        # mixing two definitions. split_n below n means episodes were dropped
        # from the model/tool split because a tool call reported no latency;
        # the wall percentiles still cover all n.
        "n_unqueued": len(unqueued),
        "unqueued_p50_s": _r(percentiles(unqueued, (0.5,))[0]),
        "unqueued_p90_s": _r(percentiles(unqueued, (0.9,))[0]),
        "split_n": len(tool),
        "tool_s_p50": _r(percentiles(tool, (0.5,))[0]),
        "llm_s_p50": _r(percentiles(llm, (0.5,))[0]),
        "non_tool_s_p50": _r(percentiles(non_tool, (0.5,))[0]),
        "output_tps_p50": _r(percentiles(tps, (0.5,))[0], 1),
        # Throughput inputs. Makespan is wall-clock, because episodes overlap,
        # so it is NOT total_s.
        "started_at": min(started) if started else None,
        "ended_at": max(ended) if ended else None,
        "output_tokens_total": out_tokens or None,
    }


def grade_status(grade_manifest, grades, rollouts, current_digest):
    """Classify one grade sidecar against the current accepted rollouts:
    'valid', 'stale' (the rollouts changed after grading), or 'invalid' (a
    structurally broken artifact). Never raises."""
    if grade_manifest is None:
        return "invalid", "missing grade manifest; rerun the grade stage"
    grades_by_episode = {str(g.get("episode_id")): g for g in grades}
    rollouts_by_episode = {str(r.meta.get("episode_id")): r for r in rollouts}
    if len(grades_by_episode) != len(grades) \
            or len(rollouts_by_episode) != len(rollouts):
        return "invalid", "duplicate episode_id in grades or rollouts"
    if grade_manifest.get("accepted_digest") != current_digest:
        return "stale", ("accepted rollouts changed after grading; rerun the "
                         "grade stage")
    if set(grades_by_episode) != set(rollouts_by_episode):
        return "invalid", "grade/rollout episode mismatch; rerun grading"
    return "valid", None


def compute_entry(bench, model, grader_name, grade_manifest, grades, rollouts,
                  current_digest):
    """Aggregate one valid (benchmark, model, grader) cell."""
    grades_by_episode = {str(g.get("episode_id")): g for g in grades}
    rollouts_by_episode = {str(r.meta.get("episode_id")): r for r in rollouts}
    joined = [(grades_by_episode[eid], rollouts_by_episode[eid])
              for eid in sorted(grades_by_episode)]
    grades = [g for g, _ in joined]
    rollouts = [r for _, r in joined]
    correctness = [g["correctness"] for g in grades]
    labels = {}
    for g in grades:
        labels[g["label"]] = labels.get(g["label"], 0) + 1
    costs = [r.timing.get("cost_usd") for r in rollouts
             if r.timing.get("cost_usd") is not None]
    tokens = [r.n_tokens for r in rollouts if r.n_tokens is not None]
    turns = [r.n_turns for r in rollouts if r.n_turns is not None]
    n = len(grades)
    metrics = {
        "accuracy": round(sum(correctness) / n, 4) if n else None,
        "ci95": bootstrap_ci(correctness),
        "n": n,
        "em": round(sum(g["em"] for g in grades) / n, 4) if n else None,
        "labels": labels,
        "parse_failures": labels.get("NO_ANSWER", 0)
        + labels.get("UNPARSEABLE", 0),
        "mean_turns": round(sum(turns) / len(turns), 2) if turns else None,
        "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "mean_cost_usd": round(sum(costs) / len(costs), 5) if costs else None,
        "costs": _cost_breakdown(rollouts, grades),
        "latency": _latency_block(rollouts),
        "truncated": sum(1 for r in rollouts if r.truncated),
        "errors": sum(1 for r in rollouts if r.error),
    }
    kind = (grade_manifest or {}).get("grade_kind")
    if kind not in ("set", "table"):
        for k in ("cover_em", "token_f1"):
            vals = [g[k] for g in grades if k in g]
            metrics[k] = round(sum(vals) / len(vals), 4) if vals else None
    if kind == "set":
        # `accuracy` is the primary per-prompt score (F1; a single-answer row
        # reduces to binary), and `em` is the fully-correct rate. Surface the
        # precision/recall decomposition so nothing is hidden by reusing the
        # single-answer metric slots.
        metrics["score_kind"] = "f1"
        metrics["mean_precision"] = (
            round(sum(g.get("precision", 0.0) for g in grades) / n, 4)
            if n else None)
        metrics["mean_recall"] = (
            round(sum(g.get("recall", 0.0) for g in grades) / n, 4)
            if n else None)
        metrics["fully_correct"] = metrics["em"]
    if kind == "table":
        # `accuracy` is mean row-F1, the discriminative headline, and `em` is
        # the strict whole-table success rate.
        metrics["score_kind"] = "row_f1"
        metrics["mean_precision"] = (
            round(sum(g.get("row_precision", 0.0) for g in grades) / n, 4)
            if n else None)
        metrics["mean_recall"] = (
            round(sum(g.get("row_recall", 0.0) for g in grades) / n, 4)
            if n else None)
        metrics["mean_item_f1"] = (
            round(sum(g.get("item_f1", 0.0) for g in grades) / n, 4)
            if n else None)
        metrics["success_rate"] = metrics["em"]
        # A table benchmark conflates two abilities: researching the answer,
        # and emitting a parseable table with the required header. A parse or
        # mapping failure scores the WHOLE episode zero, so a model with a
        # formatting tic can post a low headline while researching fine.
        # Split them: `format_fail_rate` is how often the answer never reached
        # scoring, `score_given_format` is mean row-F1 over the ones that did.
        unparsed = [g for g in grades if g.get("label") in TABLE_FORMAT_FAILS]
        scored = [g for g in grades
                  if g.get("label") not in TABLE_FORMAT_FAILS]
        metrics["format_fail_rate"] = round(len(unparsed) / n, 4) if n else None
        metrics["n_format_fail"] = len(unparsed)
        metrics["score_given_format"] = (
            round(sum(g.get("row_f1", 0.0) for g in scored) / len(scored), 4)
            if scored else None)
    return {"benchmark": bench, "model": model,
            "grader": grader_name,
            "grader_config": grade_manifest.get("grader"),
            "accepted_digest": current_digest,
            "metrics": metrics}


def _entry(bench, model, grades_path, rollouts):
    """File-reading wrapper for the report stage. Raises on any non-valid
    sidecar: a persisted summary must never contain stale numbers."""
    grades = _read_grades(grades_path)
    grade_manifest_path = grades_path.with_suffix(".manifest.json")
    grade_manifest = json.loads(grade_manifest_path.read_text()) \
        if grade_manifest_path.exists() else None
    current_digest = accepted_digest(rollouts)
    status, reason = grade_status(grade_manifest, grades, rollouts,
                                  current_digest)
    if status != "valid":
        raise ValueError(f"{status} grades for {grades_path}: {reason}")
    return compute_entry(bench, model, grades_path.stem, grade_manifest,
                         grades, rollouts, current_digest)


def summarize(eval_id):
    """Scan runs/<eval_id>/<bench>/<model>/ and write summary.json and
    summary.md."""
    d = eval_dir(eval_id)
    manifest = load_manifest(eval_id)
    entries = []
    for bench_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            rp = model_dir / "rollouts.jsonl"
            if not rp.exists():
                continue
            rollouts = list(accepted_attempts(rp).values())
            grades_dir = model_dir / "grades"
            for gp in (sorted(grades_dir.glob("*.jsonl"))
                       if grades_dir.exists() else []):
                e = _entry(bench_dir.name, model_dir.name, gp, rollouts)
                e["dataset_revision"] = (
                    (manifest["benches"].get(bench_dir.name) or {})
                    .get("dataset") or {}).get("revision")
                entries.append(e)
    doc = {"schema_version": SCHEMA_VERSION, "eval_id": eval_id,
           "manifest_hash": manifest.get("manifest_hash"),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "results": entries}
    (d / "summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    md = markdown_table(entries)
    (d / "summary.md").write_text(md)
    return doc, md


def markdown_table(entries):
    if not entries:
        return "(no graded runs)\n"
    lines = ["| benchmark | model | score | 95% CI | EM/full | token F1 "
             "| cover EM | precision | recall | n | turns | tokens "
             "| wall p50/p90/p99 | $ model | $ search | $ sub | $ judge "
             "| $ total | fmt✗ (score) | trunc | err |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
             "---|---|---|---|---|---|"]

    def na(v):
        # An em dash means not applicable for this grade kind: set benchmarks
        # have no token F1, single-answer benchmarks have no precision/recall.
        return "—" if v is None else v

    def wall(lat):
        # One cell, read as a shape: the tail matters more than any one value.
        if not lat or lat.get("p50_s") is None:
            return "—"
        return " / ".join(f"{lat[k]:.0f}"
                          for k in ("p50_s", "p90_s", "p99_s")) + "s"

    def usd(v, n_missing=0):
        # "≥" marks a lower bound: some episode costs were never reported by
        # the provider and count as $0 in the sum.
        if v is None:
            return "?"
        return f"{'≥' if n_missing else ''}{v:.4f}"

    def fmt_cell(m):
        # Table benchmarks only: how many episodes never reached scoring
        # because the answer would not parse into the required table, and what
        # the score is over the ones that did. A large count here means the
        # headline is measuring formatting, not research.
        if m.get("n_format_fail") is None:
            return "—"
        return f"{m['n_format_fail']} ({na(m.get('score_given_format'))})"

    for e in entries:
        m = e["metrics"]
        ci = m["ci95"]
        ci_s = f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "—"
        c = m["costs"]
        miss = c.get("missing") or {}
        any_miss = sum(miss.values()) if miss else 0
        lines.append(
            f"| {e['benchmark']} | {e['model']} | {m['accuracy']} | {ci_s} "
            f"| {m['em']} | {na(m.get('token_f1'))} | {na(m.get('cover_em'))} "
            f"| {na(m.get('mean_precision'))} | {na(m.get('mean_recall'))} "
            f"| {m['n']} | {m['mean_turns']} | {m['mean_tokens']} "
            f"| {wall(m.get('latency'))} "
            f"| {usd(c['model_usd'], miss.get('model', 0))} "
            f"| {usd(c['search_usd'], miss.get('search', 0))} "
            f"| {usd(c.get('subagent_usd', 0.0), miss.get('subagent', 0))} "
            f"| {usd(c['grader_usd'], miss.get('grader', 0))} "
            f"| {usd(c['total_usd'], any_miss)} | {fmt_cell(m)} "
            f"| {m['truncated']} | {m['errors']} |")
    return "\n".join(lines) + "\n"
