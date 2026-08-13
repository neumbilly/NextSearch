"""nextsearch-eval: prepare | rollout | grade | report | run.

`run` is the full pipeline for one evaluation matrix:

    nextsearch-eval run --benches main --models nextsearch-1-xs

`rollout --eval-id <id>` resumes an existing evaluation against its frozen
manifest; `grade` and `report` operate on an eval_id that already exists. Every
stage function stays importable without the CLI.
"""

import argparse
import asyncio
import hashlib
import time
from pathlib import Path

from . import benchmarks, harnesses, manifest
from .grading import grade_run, grader_config
from .harness import DEFAULT_CONCURRENCY, DEFAULT_MAX_CONTEXT, run_bench_model
from .models import get_model
from .models.registry import DEFAULT_JUDGE
from .paths import load_env
from .report import summarize
from .tools import SEARCH_MODES

# Named benchmark suites, accepted anywhere --benches is.
#
# `main` is the reported protocol: the audited SEAL-0 in full, fixed
# 100-question prefixes of FRAMES and DeepSearchQA, and the full 49-task
# WideSearch-sub. One alias, so every model's invocation is identical by
# construction.
SUITES = {
    "main": ("seal0", "frames:100", "deepsearchqa:100", "widesearch-sub"),
    "deep": ("seal0", "frames:100"),
    "wide": ("deepsearchqa:100", "widesearch-sub"),
    # The composite wide-research task, for the orchestrated harness. It is a
    # separate invocation because it needs `--harness orchestrated`.
    "composite": ("widesearch:20",),
}


def _bench_token(tok):
    """`name` or `name:n` -> (name, row cap or None)."""
    name, _, n = tok.partition(":")
    if n and not n.isdigit():
        raise SystemExit(f"--benches {tok!r}: row cap must be an integer")
    return name.strip(), (int(n) if n else None)


def _benches(arg):
    """Benchmark selection -> [(Benchmark, n_override or None)].

    Accepts registered names, suite aliases, and a per-benchmark row cap
    `name:n` that beats the run's global `--n`, so one suite can mix sizes — a
    table episode costs several times what a short-answer one does. A cap
    written on a suite alias applies to every benchmark in it.
    """
    tokens = arg.split(",") if arg else sorted(benchmarks.BENCHMARKS)
    out = []
    for tok in tokens:
        name, cap = _bench_token(tok.strip())
        for sub in SUITES.get(name, (name,)):
            sub_name, sub_cap = _bench_token(sub)
            out.append((benchmarks.get(sub_name),
                        cap if cap is not None else sub_cap))
    return out


def parse_tool_alias(arg) -> "dict | None":
    """--tool-alias "old=new[,old=new]" -> {old: new}.

    Malformed pairs, duplicate old names, and colliding new names are hard
    errors: a rename that half-applies would poison the whole comparison.
    """
    if not arg:
        return None
    alias = {}
    for pair in arg.split(","):
        old, sep, new = pair.partition("=")
        old, new = old.strip(), new.strip()
        if not sep or not old or not new:
            raise SystemExit(f"--tool-alias: malformed pair {pair!r} "
                             "(want old=new[,old=new])")
        if old in alias:
            raise SystemExit(f"--tool-alias: duplicate rename for {old!r}")
        alias[old] = new
    if len(set(alias.values())) != len(alias) \
            or set(alias.values()) & set(alias):
        raise SystemExit("--tool-alias: new names must be distinct and not "
                         "collide with renamed old names")
    return alias


def _progress(bench, model):
    def cb(done, total, r):
        print(f"[{bench.name}/{model.name}] {done}/{total} {r.sample_id} "
              f"stop={r.meta.get('stop_reason')} turns={r.n_turns} "
              f"tok={r.n_tokens} wall={r.timing.get('wall_s')}s"
              + (f" ERROR: {r.error}" if r.error else ""))
    return cb


async def _rollout(args, benches, models, judge):
    # A turn-cap override replaces every agentic benchmark's own cap for this
    # run. It is protocol, not execution: the manifest freezes the overridden
    # values per benchmark, so a resume validates it like everything else.
    max_turns = getattr(args, "max_turns", None)
    if max_turns:
        from dataclasses import replace
        benches = [(replace(b, max_turns=max_turns) if b.agentic else b, cap)
                   for b, cap in benches]
    search_mode = getattr(args, "search_mode", None)
    fetch_window = getattr(args, "fetch_window", None)
    tool_alias = parse_tool_alias(getattr(args, "tool_alias", None))
    harness = harnesses.get(getattr(args, "harness",
                                    harnesses.DEFAULT_HARNESS))
    # The task-date layer. Without it a model has no clock and resolves
    # "current" and "latest" from pretraining priors. The date is protocol —
    # it changes the prompt, hence the number — and resolves as: "off" for the
    # undated ablation, else an explicit --date, else the frozen date when
    # resuming (a run crossing midnight must not mutate its own prompt), else
    # today in UTC.
    today = getattr(args, "date", None)
    if today == "off":
        today = None
    elif not today:
        if args.eval_id:
            today = (manifest.load(args.eval_id).get("protocol") or {}) \
                .get("prompt_date")
        today = today or time.strftime("%Y-%m-%d", time.gmtime())
    system_suffix = harness.system_suffix(today, tool_alias=tool_alias)
    if tool_alias and harness.subagent:
        # Nested subagent episodes mint their own toolsets, so aliasing only
        # the orchestrator layer would record a half-true manifest.
        raise SystemExit("--tool-alias is not supported with the orchestrated "
                         "harness")
    # The orchestrated harness needs the subagent resolved once; the research
    # tool itself is minted per (benchmark, model) pair below, because its
    # rollout sink lives beside that pair's own rollouts.
    subagent_wiring, sub_cap = None, None
    if harness.subagent:
        from dataclasses import replace

        from .models.registry import DEFAULT_SUBAGENT
        from .tools.subagent import (DEFAULT_CONCURRENCY as SUB_CONCURRENCY,
                                     DEFAULT_MAX_CALLS_PER_TURN as SUB_CPT,
                                     DEFAULT_MAX_CONTEXT as SUB_MAX_CONTEXT,
                                     DEFAULT_MAX_WALL_S as SUB_MAX_WALL,
                                     SubagentRun)
        sub_model = get_model(getattr(args, "subagent", None)
                              or DEFAULT_SUBAGENT,
                              base_url=getattr(args, "base_url", None))
        sub_max_turns = getattr(args, "subagent_max_turns", None) \
            or harness.subagent.max_turns
        sub_max_context = getattr(args, "subagent_max_context", None) \
            or SUB_MAX_CONTEXT
        # 0 disables the per-turn cap; None only means "not passed", so a
        # plain `or` would be wrong here.
        sub_cpt = getattr(args, "subagent_max_calls_per_turn", None)
        sub_cpt = SUB_CPT if sub_cpt is None else (sub_cpt or None)
        sub_strict = bool(getattr(args, "subagent_strict_wall", False))
        sub_wall = getattr(args, "subagent_max_wall", None)
        sub_wall = SUB_MAX_WALL if sub_wall is None else (sub_wall or None)
        sub_cap = getattr(args, "subagent_concurrency", None) or SUB_CONCURRENCY
        if sub_model.concurrency:
            sub_cap = min(sub_cap, sub_model.concurrency)
        sub_sem = asyncio.Semaphore(sub_cap)

        def subagent_wiring(sink_path=None):
            return SubagentRun(model=sub_model, sink_path=sink_path,
                               sem=sub_sem, harness=harness.subagent.harness,
                               max_turns=sub_max_turns,
                               max_context=sub_max_context,
                               max_calls_per_turn=sub_cpt,
                               max_wall_s=sub_wall, strict_wall=sub_strict,
                               search_mode=search_mode,
                               doc=harness.subagent.doc, today=today)
        # The evaluated artifact is the PAIR, so the recorded model name is
        # composite: an orchestrated row can never be mistaken for a solo run
        # of either half. Orchestrator sampling also gets the higher
        # completion cap, because the final table IS the output.
        models = [replace(m, name=f"{m.name}+{sub_model.name}",
                          sampling={**m.sampling, "max_tokens": max(
                              m.sampling.get("max_tokens", 0),
                              harnesses.ORCHESTRATOR_MAX_TOKENS)})
                  for m in models]
    elif getattr(args, "subagent", None):
        raise SystemExit("--subagent requires --harness orchestrated")
    max_context = getattr(args, "max_context", None) or DEFAULT_MAX_CONTEXT
    # A wall-clock budget for the top-level episode is protocol, not
    # execution: it can end an episode early, so it changes the answer and a
    # resume must not silently re-scope it.
    max_wall_s = getattr(args, "max_wall", None) or None
    # An --ids-file subset is protocol too: the id-list hash is frozen, so a
    # resume against a different subset fails.
    ids, ids_sha = None, None
    ids_file = getattr(args, "ids_file", None)
    if ids_file:
        raw = Path(ids_file).read_text()
        ids = [line.strip() for line in raw.splitlines() if line.strip()]
        ids_sha = hashlib.sha256(raw.encode()).hexdigest()[:16]
    # Per-benchmark row caps decide which questions were asked, so they are
    # protocol as well.
    bench_caps = {b.name: c for b, c in benches if c is not None}
    proto = {"n": args.n,
             **({"bench_n": bench_caps} if bench_caps else {}),
             **({"search_mode": search_mode} if search_mode else {}),
             **({"max_context": max_context}
                if max_context != DEFAULT_MAX_CONTEXT else {}),
             **({"max_wall_s": max_wall_s} if max_wall_s else {}),
             **({"ids_sha": ids_sha, "n_ids": len(ids)} if ids else {}),
             **({"tool_alias": tool_alias} if tool_alias else {}),
             **harness.config(),
             **({"prompt_date": today} if today else {})}
    bench_graders = {b.name: grader_config(judge, b.grade_kind)
                     for b, _ in benches}
    # Execution conditions: provenance for the latency numbers, deliberately
    # outside the manifest hash, because concurrency moves p99 and not score.
    execution = {"concurrency": args.concurrency,
                 "model_concurrency": {m.name: m.concurrency
                                       for m in models if m.concurrency},
                 **({"subagent_concurrency": sub_cap} if sub_cap else {})}
    # The orchestrated research tool needs runtime wiring, and a fetch-window
    # or alias override needs the explicit toolset, so those manifests are
    # built from real tools rather than the harness defaults.
    manifest_tools = (
        harness.tools(search_mode, subagent_run=subagent_wiring())
        if subagent_wiring
        else (harness.tools(search_mode, fetch_window=fetch_window,
                            tool_alias=tool_alias)
              if (fetch_window or tool_alias) else None))
    current = manifest.build([b for b, _ in benches], models,
                             grader_config(judge), proto,
                             description=args.description,
                             bench_graders=bench_graders, harness=harness,
                             execution=execution, harness_tools=manifest_tools)
    if args.eval_id:
        manifest.check_resume(args.eval_id, current,
                              allow_code_drift=args.allow_code_drift)
        eval_id = args.eval_id
    else:
        eval_id = manifest.new_eval_id()
        manifest.freeze(eval_id, current)
    print(f"[rollout] eval_id={eval_id}")
    # All pairs run concurrently under one shared semaphore, so slow cells
    # overlap fast ones instead of blocking them.
    sem = asyncio.Semaphore(args.concurrency)
    model_sems = {m.name: asyncio.Semaphore(m.concurrency)
                  for m in models if m.concurrency}

    async def _pair(b, rows, tools, m):
        out = manifest.eval_dir(eval_id) / b.name / m.name / "rollouts.jsonl"
        res = await run_bench_model(b, m, rows, out, tools,
                                    max_turns=b.max_turns,
                                    max_context=max_context,
                                    max_wall_s=max_wall_s, sem=sem,
                                    model_sem=model_sems.get(m.name),
                                    progress=_progress(b, m),
                                    system_suffix=system_suffix)
        print(f"[rollout] {b.name}/{m.name}: ran {res['n_ran']} "
              f"(skipped {res['n_skipped']}, errors {res['n_errors']})")

    def pair_tools(b, m):
        """The toolset for one (benchmark, model) pair. The orchestrated
        harness mints a fresh research tool per pair so its nested-rollout
        sink lands beside that pair's rollouts."""
        if not b.agentic:
            return []
        if subagent_wiring:
            sink = (manifest.eval_dir(eval_id) / b.name / m.name
                    / "subagent_rollouts.jsonl")
            return harness.tools(search_mode,
                                 subagent_run=subagent_wiring(sink))
        return harness.tools(search_mode, fetch_window=fetch_window,
                             tool_alias=tool_alias)

    tasks = []
    for b, n_override in benches:
        n_rows = n_override if n_override is not None else args.n
        if ids:
            id_set = set(ids)
            rows = [r for r in b.load_rows(None) if r.id in id_set]
            missing = id_set - {r.id for r in rows}
            if missing:
                raise SystemExit(f"--ids-file: {len(missing)} ids not in "
                                 f"{b.name} (e.g. {sorted(missing)[:3]})")
            if n_rows:
                rows = rows[:n_rows]
        else:
            rows = b.load_rows(n_rows)
        tasks.extend(_pair(b, rows, pair_tools(b, m), m) for m in models)
    await asyncio.gather(*tasks)
    return eval_id


async def _grade(eval_id, judge, concurrency=16):
    from .grading import JudgeCache
    d = manifest.eval_dir(eval_id)
    # All pairs grade concurrently against one shared cache and one shared
    # semaphore capping total in-flight judge calls.
    cache = JudgeCache()
    sem = asyncio.Semaphore(concurrency)

    async def _pair(bench_name, model_dir, grade_kind):
        res = await grade_run(model_dir, judge, cache=cache, sem=sem,
                              grade_kind=grade_kind)
        print(f"[grade] {bench_name}/{model_dir.name}: {res['n']} graded, "
              f"{res['n_judge_calls']} judge calls -> {res['grader_id']}")

    tasks = []
    for bench_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        b = benchmarks.BENCHMARKS.get(bench_dir.name)
        grade_kind = b.grade_kind if b else "single"
        tasks.extend(
            _pair(bench_dir.name, model_dir, grade_kind)
            for model_dir in sorted(p for p in bench_dir.iterdir()
                                    if p.is_dir())
            if (model_dir / "rollouts.jsonl").exists())
    await asyncio.gather(*tasks)


def _report(eval_id):
    doc, md = summarize(eval_id)
    print(f"[report] {len(doc['results'])} entries -> "
          f"{manifest.eval_dir(eval_id) / 'summary.json'}")
    print(md)


def _add_run_args(p):
    p.add_argument("--benches", required=True,
                   help="comma-separated benchmark names, a suite alias "
                        f"({', '.join(sorted(SUITES))}), or `name:n` to cap "
                        "one benchmark's rows")
    p.add_argument("--models", required=True,
                   help="registry names (see `nextsearch-eval models`) or raw "
                        "'vendor/model' OpenRouter ids")
    p.add_argument("--base-url", default=None,
                   help="endpoint for self-hosted models, overriding "
                        "NEXTSEARCH_BASE_URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--n", type=int, default=None,
                   help="rows per benchmark (default: the whole prepared set)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="global cap on in-flight episodes across all pairs")
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument("--harness", choices=sorted(harnesses.HARNESSES),
                   default=harnesses.DEFAULT_HARNESS,
                   help="prompt layer plus toolset: 'solo' is one agent with "
                        "search and fetch (the reported configuration); "
                        "'orchestrated' is an orchestrator delegating to "
                        "research subagents")
    p.add_argument("--search-mode", choices=SEARCH_MODES, default=None,
                   help="search backend and tier (default: Parallel turbo). "
                        "Frozen in the manifest — this is the backend "
                        "comparison knob")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD|off",
                   help="the task date placed in every system prompt "
                        "(default: today, UTC). 'off' disables it — the "
                        "undated ablation")
    p.add_argument("--max-turns", type=int, default=None,
                   help="override every agentic benchmark's turn cap; frozen "
                        "per benchmark in the manifest")
    p.add_argument("--max-context", type=int, default=None,
                   help="context budget in tokens; episodes stop, truncated, "
                        "at this conversation size (default 512k, "
                        "effectively uncapped for evaluation)")
    p.add_argument("--max-wall", type=int, default=None,
                   help="wall-clock budget per top-level episode in seconds "
                        "(default unbounded): a wrap-up note at 70%%, a hard "
                        "stop into the forced-answer salvage at the budget")
    p.add_argument("--fetch-window", type=int, default=None,
                   help="fetch window size in characters (default 4000). "
                        "Changes the tool spec, the slicing, and the frozen "
                        "config — not the response cache, so every window "
                        "reads the same bytes")
    p.add_argument("--tool-alias", default=None, metavar="OLD=NEW[,OLD=NEW]",
                   help="rename tools for this run, e.g. "
                        "search=web_search,fetch=read_url. Tool specs and the "
                        "guidelines' backticked symbols get the new names; "
                        "behavior is unchanged. The tool-name memorization "
                        "probe")
    p.add_argument("--ids-file", default=None,
                   help="roll out only the sample ids listed in this file, "
                        "one per line; the file hash is frozen")
    p.add_argument("--eval-id", default=None,
                   help="resume an existing evaluation")
    p.add_argument("--allow-code-drift", action="store_true",
                   help="resume after code changes, recording a provenance "
                        "event")
    p.add_argument("--description", default=None,
                   help="one-line human label for this evaluation")
    # orchestrated-harness knobs
    p.add_argument("--subagent", default=None,
                   help="research subagent for --harness orchestrated "
                        "(default: the harness default). This is the model "
                        "under test in the orchestrated setting")
    p.add_argument("--subagent-max-turns", type=int, default=None,
                   help="subagent episode turn cap (default 10)")
    p.add_argument("--subagent-max-context", type=int, default=None,
                   help="subagent context policy cap (default 48000). Must "
                        "stay BELOW the subagent's real serving window, or "
                        "the graceful stop, the budget nudges, and the forced "
                        "final answer all go dead and overflows become "
                        "provider errors")
    p.add_argument("--subagent-max-calls-per-turn", type=int, default=None,
                   help="tool calls executed per subagent turn (default 5, "
                        "0 = unbounded)")
    p.add_argument("--subagent-max-wall", type=int, default=None,
                   help="wall-clock budget per nested episode in seconds "
                        "(default 420, 0 = unbounded)")
    p.add_argument("--subagent-strict-wall", action="store_true",
                   help="cut a nested call still running at its deadline "
                        "instead of letting tool execution and salvage grace "
                        "overshoot it. Makes that episode's cost a lower "
                        "bound, since a cut call was partly billed and never "
                        "reported")
    p.add_argument("--subagent-concurrency", type=int, default=None,
                   help="global cap on in-flight subagent episodes "
                        "(default 32). Lowering it queues research calls, "
                        "which inflates the parent episode's wall time")


def main(argv=None):
    load_env()
    ap = argparse.ArgumentParser(prog="nextsearch-eval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare",
                       help="pull source datasets -> canonical rows + manifest")
    p.add_argument("--benches", default=None)
    p.add_argument("--no-revisions", action="store_true",
                   help="skip our audited gold revisions and evaluate against "
                        "the unmodified upstream golds")

    for name in ("rollout", "run"):
        _add_run_args(sub.add_parser(
            name, help=("full pipeline: prepare, rollout, grade, report"
                        if name == "run" else "rollout stage only")))

    p = sub.add_parser("grade", help="grade an existing evaluation")
    p.add_argument("--eval-id", required=True)
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument("--concurrency", type=int, default=16,
                   help="global cap on in-flight judge calls")

    p = sub.add_parser("report", help="aggregate an evaluation into a table")
    p.add_argument("--eval-id", required=True)

    sub.add_parser("models", help="list registered models and benchmarks")

    args = ap.parse_args(argv)

    if args.cmd == "models":
        from .models.registry import MODELS
        print("models:")
        for name, m in sorted(MODELS.items()):
            where = m.base_url or m.client
            print(f"  {name:22s} {m.model_id:36s} {where}")
        print("\n(any 'vendor/model' OpenRouter id also works)\n")
        print("benchmarks:")
        for name, b in sorted(benchmarks.BENCHMARKS.items()):
            print(f"  {name:18s} grade={b.grade_kind:7s} "
                  f"max_turns={b.max_turns}")
        print("\nsuites:")
        for name, members in sorted(SUITES.items()):
            print(f"  {name:10s} {', '.join(members)}")
        return

    if args.cmd == "prepare":
        for b, _ in _benches(args.benches):
            b.prepare(apply_revisions=not args.no_revisions)
        return

    if args.cmd == "report":
        _report(args.eval_id)
        return

    judge = get_model(args.judge)
    if args.cmd == "grade":
        asyncio.run(_grade(args.eval_id, judge, args.concurrency))
        return

    benches = _benches(args.benches)
    models = [get_model(n.strip(), base_url=args.base_url)
              for n in args.models.split(",")]

    if args.cmd == "rollout":
        asyncio.run(_rollout(args, benches, models, judge))
        return

    async def full():
        for b, _ in benches:
            if not b.rows_path.exists() or not b.manifest_path.exists():
                b.prepare()
        eval_id = await _rollout(args, benches, models, judge)
        await _grade(eval_id, judge, args.concurrency)
        _report(eval_id)
    asyncio.run(full())


if __name__ == "__main__":
    main()
