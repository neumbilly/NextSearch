"""The `research` tool: one call is one complete nested research episode.

Under the orchestrated harness the top-level model gets only this tool. The
subagent is an independent model running the SAME episode loop
(`harness.run_episode`) with the `solo` toolset and guidelines — that is, the
exact configuration a research model is evaluated under standalone, so any
model slots into the subagent seat with no special-casing.

The orchestrator sees only the subagent's final message text, or a failure
note it can act on (split the task, retry). This tool never raises past the
harness contract.

Identity versus plumbing: `SubagentRun` carries both, but only the identity
fields — model, harness, limits, search tier — land in `Tool.config` and from
there into the run manifest. The sink, semaphore, client, and pre-built tools
are execution wiring. Nested rollouts append to `subagent_rollouts.jsonl`
beside the pair's own rollouts, linked to the parent episode.
"""

import contextlib
import hashlib
from dataclasses import dataclass

from ..types import tool_spec

DEFAULT_MAX_TURNS = 10
# The subagent's context POLICY cap — deliberately well below the serving
# window, not near it. It has to be, twice over: the harness's graceful stop
# and its 80/90% nudges only mean anything if they trip before the provider's
# own hard error, and the forced final answer needs headroom to make one more
# call. Set this too high for the model actually being served and all three
# mechanisms go dead at once — overflows then arrive as provider errors that
# return nothing at all. 48k keeps roughly 17k of headroom under a 64k window;
# raise it per-run only against a measured window.
DEFAULT_MAX_CONTEXT = 48_000
# Tool calls per TURN inside a nested episode. The runaway this bounds is a
# decoding failure rather than a strategy: the policy emits several turns'
# worth of tool-call blocks in one completion — hundreds of calls in a single
# message at the extreme — and cannot observe any result until the turn ends,
# so the entire tail is issued blind. A small cap prevents nearly all of the
# resulting context deaths.
DEFAULT_MAX_CALLS_PER_TURN = 5
# Ceiling on ONE report entering the parent's context. A rail, not a
# compression policy: typical reports are a few thousand characters, but a
# single runaway report can consume a fifth of the orchestrator's whole
# window, and ten of those in one turn ends the parent episode. Generous
# enough that real wide tables pass through untouched.
DEFAULT_MAX_RESULT_CHARS = 40_000
# Wall-clock budget for ONE nested episode. Sized off the measured
# distribution so it truncates a small fraction of episodes while reclaiming
# the pathological tail — where the parent's turn is blocked on the slowest
# member of its batch the entire time. Soft note at 70% of the budget, hard
# stop into the forced-answer salvage at the budget.
DEFAULT_MAX_WALL_S = 420
DEFAULT_BUDGET = 40        # research calls per parent episode
DEFAULT_CONCURRENCY = 32   # run-global in-flight subagent episodes

SUBAGENT_DOC = "subagent.md"

SPEC = tool_spec(
    "research",
    "Delegates one research task to an independent web-research agent that "
    "searches the web and returns its findings as text. The agent has NO "
    "memory of previous calls and cannot see this conversation: every task "
    "must be fully self-contained — name the entities, constraints, "
    "timeframe, and the exact output format you want back. Prefer several "
    "small, focused tasks issued together in one turn over one broad task. "
    "If a call returns a failure note, split the task into smaller pieces "
    "and retry.",
    {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "A complete, self-contained research task: "
                    "what to find, for which entities, under which "
                    "constraints, and the exact output format wanted "
                    "(e.g. 'a markdown table with columns X, Y, Z')."},
        },
        "required": ["task"],
    },
)

# Orchestrator-visible failure notes, keyed by the nested stop reason. These
# are actionable — what to do next — never a stack trace.
_FAIL_NOTES = {
    "max_turns": ("research agent ran out of turns before finishing. Treat "
                  "anything below as incomplete, and re-delegate what is "
                  "still missing as a smaller, more specific task."),
    "context_limit": ("research agent ran out of context before finishing. "
                      "Treat anything below as incomplete, and re-delegate "
                      "what is still missing as a smaller, more specific "
                      "task."),
    "wall_clock": ("research agent ran out of time before finishing. Treat "
                   "anything below as incomplete; re-delegate only the "
                   "specific gap that remains, as a narrower task."),
}


def _partial(note, rollout):
    """A failed nested episode's note PLUS whatever it had already written.

    Replacing a whole episode with a one-line failure note throws away
    finished work: most failed nested episodes still hold non-empty final
    text, sometimes a fully formed table. An orchestrator told only "it
    failed" re-delegates the same slice several times over. The note stays
    FIRST so the incompleteness is read before the content.
    """
    for m in reversed(rollout.messages):
        if m.get("role") == "assistant":
            text = (m.get("content") or "").strip()
            return f"{note}\n\nPARTIAL RESULT:\n{text}" if text else note
    return note


@dataclass
class SubagentRun:
    """Runtime wiring for one (benchmark, model) pair's research tool."""
    model: object                    # the subagent Model
    sink_path: object = None         # where nested rollouts land (None = don't record)
    sem: object = None               # run-global asyncio.Semaphore (None = uncapped)
    client: object = None            # override; default get_client(model)
    web_tools: list = None           # override; default: the harness's toolset
    harness: str = "solo"            # the subagent's harness (prompt + toolset)
    max_turns: int = DEFAULT_MAX_TURNS
    max_context: int = DEFAULT_MAX_CONTEXT
    max_calls_per_turn: "int | None" = DEFAULT_MAX_CALLS_PER_TURN
    max_result_chars: "int | None" = DEFAULT_MAX_RESULT_CHARS
    max_wall_s: "int | None" = DEFAULT_MAX_WALL_S
    strict_wall: bool = False        # cut a straggler at the budget rather than
    # letting tool execution plus salvage grace overshoot it
    search_mode: "str | None" = None
    budget: int = DEFAULT_BUDGET
    doc: str = SUBAGENT_DOC          # the subagent's role prompt
    today: "str | None" = None       # task date for the nested episodes: the
    # subagent, not the orchestrator, is the one reading the web, so the date
    # layer has to reach this level too


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def tool(run: SubagentRun):
    from .. import harnesses
    from ..prompts import load_doc
    from . import Tool

    sub_harness = harnesses.get(run.harness)
    doc = load_doc(run.doc)
    guidelines = sub_harness.system_suffix(run.today)
    web_tools = (run.web_tools if run.web_tools is not None
                 else sub_harness.tools(run.search_mode))

    async def execute(args, context=None):
        from ..harness import STOP_FINAL, run_episode
        from ..models import get_client
        from ..types import Row, system, user

        task = (args.get("task") or "").strip()
        if not task:
            raise ValueError("research requires a non-empty 'task' — a "
                             "self-contained description of what to find and "
                             "the output format wanted")
        context = context or {}
        parent = context.get("episode_id") or "?"
        call_id = context.get("call_id") or "?"
        row = Row(messages=[system(doc), user(task)],
                  id=f"{parent}#{call_id}",
                  meta={"parent_episode_id": parent, "call_id": call_id,
                        "task": task})
        client = run.client or get_client(run.model)
        async with contextlib.AsyncExitStack() as stack:
            if run.sem is not None:
                await stack.enter_async_context(run.sem)
            rollout = await run_episode(
                client, run.model, row, web_tools,
                max_turns=run.max_turns, max_context=run.max_context,
                max_calls_per_turn=run.max_calls_per_turn,
                max_wall_s=run.max_wall_s, strict_wall=run.strict_wall,
                bench=context.get("bench"), system_suffix=guidelines)
        if run.sink_path is not None:
            from .. import io
            io.write_rollouts(run.sink_path, [rollout], append=True)
        sr = rollout.meta.get("stop_reason")
        if sr == STOP_FINAL:
            content = rollout.messages[-1].get("content") or ""
        else:
            content = _partial(
                _FAIL_NOTES.get(sr, f"research agent failed ({sr}). Retry, "
                                    "possibly rephrased."), rollout)
        if run.max_result_chars and len(content) > run.max_result_chars:
            content = (content[:run.max_result_chars]
                       + "\n\n[report truncated by the harness at "
                       f"{run.max_result_chars} characters — it was too "
                       "large to merge. Re-delegate this as narrower slices "
                       "if you need the rest.]")
        t = rollout.timing or {}
        info = {"tool": "research", "stop_reason": sr,
                "sub_episode_id": rollout.meta.get("episode_id"),
                "sub_turns": rollout.n_turns,
                "sub_tokens": rollout.n_tokens,
                # The nested episode's OWN wall time, excluding what it spent
                # waiting for a subagent slot. The call's `latency_s` is
                # queue plus service; this is service alone. Their difference
                # is the queue, which is what makes an orchestrated episode's
                # wall time capacity-dependent.
                "sub_wall_s": (rollout.timing or {}).get("wall_s"),
                "n_search_calls": len(t.get("tool_calls") or []),
                # search_usd/cost_usd may be None when a nested call had
                # unreported cost; the report counts those as missing, never
                # as $0.
                "search_usd": t.get("tool_cost_usd"),
                "cost_usd": t.get("cost_usd")}
        return content, info

    config = {"kind": "subagent",
              "model": run.model.to_config(),
              "harness": run.harness,
              "max_turns": run.max_turns,
              "max_context": run.max_context,
              "max_calls_per_turn": run.max_calls_per_turn,
              "max_result_chars": run.max_result_chars,
              "max_wall_s": run.max_wall_s,
              "strict_wall": run.strict_wall,
              "budget": run.budget,
              **({"search_mode": run.search_mode} if run.search_mode else {}),
              "prompt_sha": _sha(doc),
              # Hash the UNDATED guidelines, so it stays stable across task
              # dates — same rule as Harness.config; the date is explicit.
              **({"guidelines_sha": _sha(sub_harness.system_suffix() or "")}
                 if guidelines else {}),
              **({"prompt_date": run.today} if run.today else {})}
    return Tool(spec=SPEC, execute=execute, config=config,
                max_calls=run.budget, wants_context=True)
