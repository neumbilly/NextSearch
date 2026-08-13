"""The agent harness: (model, task row) -> one complete episode.

This is the loop the NextSearch-1 models were trained and evaluated in. It is
model-agnostic — everything provider-specific sits behind `Client.generate` —
and prompt/tool-agnostic: which instructions and which tools an episode gets
is configuration (see `harnesses.py`), not behavior encoded here.

Per-episode failures are captured into the resulting `Rollout` with a
structured `stop_reason`, never raised. One bad episode must not kill a run
of several hundred.

Rollout files are append-only and immutable once written. The accepted record
for a sample is the LATEST line for it (episode ids are
`"<sample_id>/<attempt_idx>"`); resuming re-runs only samples whose accepted
record carries an error, appending a fresh attempt.
"""

import asyncio
import contextlib
import json
import random
import time

from . import io
from .types import Rollout, tool as tool_msg, tool_call_args, user as user_msg

STOP_FINAL = "final"
STOP_EMPTY = "empty_response"
STOP_MAX_TURNS = "max_turns"
STOP_CONTEXT = "context_limit"
STOP_WALL_CLOCK = "wall_clock"
STOP_MODEL_ERROR = "model_error"
STOP_INTERNAL_ERROR = "internal_error"

# Turn budget. Benchmarks override this with their own cap; the reported
# NextSearch-1 protocol uses 10 turns for the deep benchmarks and the
# subagent table benchmark, 20 for the set-answer benchmark.
DEFAULT_MAX_TURNS = 10
# Context budget in tokens. An episode stops (truncated, with
# stop_reason=context_limit) once the conversation reaches this size, measured
# by the provider's own usage counts — prompt+completion of the latest call is
# approximately the next call's prompt. 512k is effectively uncapped for
# evaluation. Distinct from max_turns: turns bound actions, this bounds tokens.
DEFAULT_MAX_CONTEXT = 512_000
# Episodes are latency-bound (tens of seconds each), so throughput is set by
# how many run at once rather than by any per-call speed.
DEFAULT_CONCURRENCY = 32
# Some providers intermittently return a completion with no content AND no
# tool calls, sometimes without usage. Accepting that as the terminal answer
# would silently zero the episode, so the call is retried; a persistent empty
# is recorded as stop_reason=empty_response.
EMPTY_RETRIES = 2
# The same argument one level up: a RAISED call (transport flake, an
# unparseable response body) would end the episode outright, so an agent whose
# research was already finished would score zero for want of one more call.
# A context overflow is NOT retried — re-sending the same oversized prompt
# fails identically, and that path has its own handling below.
MODEL_ERROR_RETRIES = 2
MODEL_ERROR_BACKOFF_S = 1.0
# Per-turn tool-call cap (None = unbounded). Calls beyond the cap are answered
# like an exhausted tool budget: never executed, $0, and the model is told.
# This bounds ONE turn's context growth, which max_context structurally cannot
# see coming — it is checked against the PREVIOUS call's usage, before this
# turn's results exist. Without it, a model that emits a very large batch of
# tool calls in a single completion can bury itself in results before the
# context check ever runs again.
DEFAULT_MAX_CALLS_PER_TURN = None
# Wall-clock budget in seconds (None = unbounded). Turns, context, and
# calls-per-turn all bound WORK; none of them bounds TIME. That matters most
# under the orchestrated harness, where a turn blocks on the slowest subagent
# in its batch and one pathological episode can stall its whole parent. Two
# tiers, matching the turn and context nudges: a SOFT note at WALL_SOFT_FRAC
# of the budget telling the model to wrap up, and a HARD stop at the budget
# that ends the episode and routes into the forced-answer salvage below —
# never a bare cancel, which would discard exactly the work the deadline
# exists to protect.
DEFAULT_MAX_WALL_S = None
WALL_SOFT_FRAC = 0.7
# The salvage call has to run AFTER the budget is spent, so it gets its own
# allowance rather than inheriting a deadline that has already passed.
FORCED_ANSWER_GRACE_S = 180
# Budget-nudge thresholds. The loop's limits are otherwise invisible to the
# model, and hitting one with no answer written was the single largest source
# of lost episodes we measured. A soft note when the budget gets close, a firm
# one on the last turn.
NEAR_BUDGET_CTX_FRAC = 0.8
LAST_BUDGET_CTX_FRAC = 0.9


def budget_near_note(turns_left, ctx_pct=None):
    """The soft nudge. `ctx_pct` is omitted where the caller cannot know it."""
    ctx = f", context {ctx_pct}% used" if ctx_pct is not None else ""
    return (f"[harness note: you are near the episode limit ({turns_left} "
            f"tool turns left{ctx}). Get anything still missing in your "
            "next step, then answer; a best-supported partial answer beats "
            "no answer.]")


def budget_last_note(what):
    """The firm nudge; `what` names the tripped limit."""
    return (f"[harness note: {what} — answer in your next message using "
            "what you already have. Give your best-supported answer; if "
            "something could not be confirmed, say so plainly instead of "
            "continuing to search.]")


def wall_note(seconds_left):
    """The soft time nudge, injected like the turn and context ones. Phrased
    in work terms rather than seconds: a model cannot perceive elapsed time,
    so a raw countdown is noise — what it can act on is 'stop opening new
    threads'."""
    return ("[harness note: this research is taking too long and will be cut "
            f"off in about {int(seconds_left)}s. Stop opening new lines of "
            "inquiry, finish the one you are on, and answer with what you "
            "have — a well-sourced partial answer beats being cut off.]")


def forced_answer_note(what):
    """The last-resort prompt when a hard limit is reached with no answer
    written. The nudges above WARN before the wall; this is the wall — the
    loop otherwise ends on a tool result, leaving the episode with no final
    message at all, which scores zero even when the research is sitting
    complete in the transcript."""
    return (f"[harness note: {what}. No further tools are available. Write "
            "your final answer now from the results already in this "
            "conversation, in exactly the format the task asked for. Mark "
            "anything you could not confirm the way the task specifies "
            "rather than omitting it.]")


def _cost(usage_log, pricing):
    """Dollar cost of the episode's LLM calls. A provider-reported cost wins
    over the pricing table; the table is the fallback; None when neither is
    known (never a fabricated zero)."""
    if usage_log and all(u.get("cost") is not None for u in usage_log):
        return round(sum(u["cost"] for u in usage_log), 6)
    if not pricing or pricing.get("in") is None or pricing.get("out") is None:
        return None
    p_in = sum(u.get("prompt_tokens", 0) for u in usage_log)
    p_out = sum(u.get("completion_tokens", 0) for u in usage_log)
    return round((p_in * pricing.get("in", 0)
                  + p_out * pricing.get("out", 0)) / 1e6, 6)


def _is_context_error(exc) -> bool:
    s = str(exc).lower()
    return "context" in s and ("length" in s or "window" in s or "exceed" in s)


def _call_key(tc):
    """Canonical identity of a tool call for within-turn dedupe: (name,
    normalized args). Falls back to the raw argument string when the payload
    will not parse — a malformed call is still identical to its own copy."""
    fn = tc.get("function") if isinstance(tc, dict) else None
    name = fn.get("name") if isinstance(fn, dict) else None
    if not name:
        return None
    try:
        return name, json.dumps(tool_call_args(tc), sort_keys=True)
    except Exception:  # noqa: BLE001 — identity only; execution reports errors
        return name, str(fn.get("arguments"))


async def run_episode(client, model, row, tools, max_turns=DEFAULT_MAX_TURNS,
                      max_context=DEFAULT_MAX_CONTEXT,
                      max_calls_per_turn=DEFAULT_MAX_CALLS_PER_TURN,
                      max_wall_s=DEFAULT_MAX_WALL_S, strict_wall=False,
                      attempt_idx=0, bench=None, system_suffix=None) -> Rollout:
    """One complete episode: the tool loop, with full bookkeeping.

    `system_suffix` is the harness prompt layer (the task date plus the
    research guidelines). It is appended to the row's system message — or
    becomes one if the row has none — so the recorded messages show exactly
    what the model saw.

    `max_calls_per_turn` bounds ONE assistant turn's tool calls. Identical
    calls within a turn are collapsed to a single execution regardless of the
    cap: a model that asks the same question twice in one breath has asked it
    once. Both limits are recorded in meta, so an episode's budget is readable
    straight off the rollout.
    """
    messages = [dict(m) for m in row.messages]
    if system_suffix:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                f"{messages[0].get('content', '')}\n\n{system_suffix}")
        else:
            messages.insert(0, {"role": "system", "content": system_suffix})
    tool_map = {t.spec["name"]: t for t in tools}
    specs = [t.spec for t in tools]
    sampling = dict(model.sampling)
    usage_log, tool_log, llm_latencies = [], [], []
    calls_by_tool = {}   # per-episode execution counts (Tool.max_calls budgets)
    budget_notes = {}    # nudges injected: {"near"|"last"|"wall": turn number}
    stop_reason, error = STOP_MAX_TURNS, None
    n_empty_retries = n_model_retries = 0
    n_calls_capped = n_calls_deduped = n_calls_cut = 0
    forced_answer = None
    started_at = time.time()
    t0 = time.monotonic()
    episode_id = f"{row.id}/{attempt_idx}"

    def wall_left():
        """Seconds of budget remaining, or None when unbounded."""
        return None if not max_wall_s else max_wall_s - (time.monotonic() - t0)

    async def generate():
        """One model call, timed. Latency is recorded for every attempt —
        including discarded empty-response retries and a final failing call —
        so `llm_s` is the episode's real model time. That makes the list
        one-per-attempt, NOT index-aligned with `usage`, which holds successes
        only. Provider-side queueing and the SDK's own retries are inside
        these numbers, exactly as they are inside `wall_s`."""
        t_call = time.monotonic()
        try:
            call = client.generate(model.model_id, messages, specs, sampling)
            left = wall_left()
            # The hard bound has to sit on the CALL, not just between turns: a
            # single long generation is exactly the shape that blows past a
            # budget, and a turn-boundary check can never interrupt one.
            return await (asyncio.wait_for(call, timeout=max(left, 1.0))
                          if left is not None else call)
        finally:
            llm_latencies.append(round(time.monotonic() - t_call, 3))

    for turn_idx in range(max_turns):
        if (left := wall_left()) is not None and left <= 0:
            stop_reason = STOP_WALL_CLOCK
            break
        msg = usage = None
        for retry_idx in range(MODEL_ERROR_RETRIES + 1):
            try:
                msg, usage = await generate()
                while (n_empty_retries < EMPTY_RETRIES
                       and not msg.get("tool_calls")
                       and not (msg.get("content") or "").strip()):
                    if usage:  # a discarded empty attempt still cost something
                        usage_log.append(usage)
                    n_empty_retries += 1
                    msg, usage = await generate()
                error = None
                break
            except (TimeoutError, asyncio.TimeoutError):
                # The wall-clock bound, not a provider failure: no error is
                # recorded (nothing went wrong) and no retry is attempted
                # (there is no time left to retry into). Salvage handles it.
                stop_reason, error, msg = STOP_WALL_CLOCK, None, None
                break
            except Exception as e:  # noqa: BLE001 — captured, never raised
                is_ctx = _is_context_error(e)
                stop_reason = STOP_CONTEXT if is_ctx else STOP_MODEL_ERROR
                error = f"{type(e).__name__}: {e}"
                msg = None
                if is_ctx or retry_idx == MODEL_ERROR_RETRIES:
                    break
                n_model_retries += 1
                await asyncio.sleep(MODEL_ERROR_BACKOFF_S * 2 ** retry_idx
                                    + random.uniform(0, MODEL_ERROR_BACKOFF_S))
        if msg is None:
            break
        messages.append(msg)
        usage_log.append(usage or {})
        calls = msg.get("tool_calls") or []
        if not calls:
            stop_reason = STOP_FINAL if (msg.get("content") or "").strip() \
                else STOP_EMPTY
            break
        # Context budget: stop before executing tools if the conversation has
        # reached max_context. Usage the provider never reported cannot trip
        # the cap — the API's own overflow error remains the backstop.
        ctx_tokens = (usage or {}).get("prompt_tokens", 0) \
            + (usage or {}).get("completion_tokens", 0)
        if max_context and ctx_tokens >= max_context:
            stop_reason = STOP_CONTEXT
            break

        async def run_call(call_idx, tc, over_budget, over_turn):
            call_id = (tc.get("id") if isinstance(tc, dict) else None) \
                or f"call-{turn_idx + 1}-{call_idx + 1}"
            if isinstance(tc, dict):
                tc["id"] = call_id
            name = "<malformed>"
            t0_tool = time.monotonic()
            try:
                if not isinstance(tc, dict):
                    raise TypeError("tool call must be an object")
                fn = tc.get("function")
                if not isinstance(fn, dict) or not fn.get("name"):
                    raise ValueError("tool call requires function.name")
                name = fn["name"]
                t_tool = tool_map.get(name)
                if t_tool is None:
                    raise KeyError(f"unknown tool {name!r}")
                if over_budget:
                    # A skipped call genuinely spent $0 — the cost is KNOWN, so
                    # it must not push the episode into missing-cost territory.
                    content = (f"tool budget exhausted: {name} is limited to "
                               f"{t_tool.max_calls} calls per episode and this "
                               "call was not executed — finish the task with "
                               "what you already have.")
                    info = {"tool": name, "budget_exhausted": True,
                            "cost_usd": 0.0}
                elif over_turn:
                    content = (f"turn call cap reached: at most "
                               f"{max_calls_per_turn} tool calls run per turn "
                               "and this one was not executed — read the "
                               "results you did get, then ask again next turn "
                               "for whatever is still missing.")
                    info = {"tool": name, "turn_cap_skipped": True,
                            "cost_usd": 0.0}
                elif t_tool.wants_context:
                    content, info = await t_tool.execute(
                        tool_call_args(tc),
                        context={"episode_id": episode_id, "call_id": call_id,
                                 "bench": bench})
                else:
                    content, info = await t_tool.execute(tool_call_args(tc))
            except Exception as e:  # noqa: BLE001 — tool errors are feedback
                # A raised call never completed, so it was never billed: the
                # cost is KNOWN to be $0. Omitting it instead would poison the
                # episode's tool-cost rollup to None and drop the episode from
                # the run's cost table entirely, understating the run.
                content = f"tool error: {type(e).__name__}: {e}"
                info = {"tool": name, "error": str(e), "cost_usd": 0.0}
            # Every tool call carries a latency, failing ones included: a
            # failed search still burned wall time, and an episode missing ANY
            # per-call latency drops out of the model/tool split entirely. The
            # tool's own measurement wins; this is the outer bound that fills
            # the gap.
            info = {"latency_s": round(time.monotonic() - t0_tool, 3),
                    **(info or {})}
            return call_id, content, info

        # A turn's tool calls run CONCURRENTLY — they are independent by
        # contract, since the model batched them into one turn — and transcript
        # order is preserved by appending in call order after the gather. This
        # is immaterial for sub-second searches and essential for the
        # orchestrated harness, where one call is a whole subagent episode.
        # Per-episode Tool.max_calls budgets are decided up front so a parallel
        # batch cannot overshoot the cap.
        #
        # Two admission rules run first, both decided in call order so the
        # batch is deterministic:
        #   dedupe — an identical (name, args) call already in THIS turn is not
        #     executed again; every copy gets the original's result. The model
        #     saw no result between the two, so the repeat carries no new
        #     information: collapsing it is free.
        #   turn cap — beyond max_calls_per_turn DISTINCT calls, the rest are
        #     skipped with a note. Duplicates never count against the cap,
        #     since they cost nothing.
        budgeted, primary_of, first_seen = [], {}, {}
        n_turn_calls = 0
        for call_idx, tc in enumerate(calls):
            name = tc.get("function", {}).get("name") \
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict) \
                else None
            t_tool = tool_map.get(name) if name else None
            key = _call_key(tc) if isinstance(tc, dict) else None
            if key is not None and key in first_seen:
                primary_of[call_idx] = first_seen[key]
                n_calls_deduped += 1
                budgeted.append(None)
                continue
            over = (t_tool is not None and t_tool.max_calls is not None
                    and calls_by_tool.get(name, 0) >= t_tool.max_calls)
            over_turn = (not over and max_calls_per_turn is not None
                         and n_turn_calls >= max_calls_per_turn)
            if t_tool is not None and not over and not over_turn:
                calls_by_tool[name] = calls_by_tool.get(name, 0) + 1
            if over_turn:
                n_calls_capped += 1
            else:
                n_turn_calls += 1
                if key is not None:
                    first_seen[key] = call_idx
            budgeted.append(run_call(call_idx, tc, over, over_turn))
        idxs = [i for i, c in enumerate(budgeted) if c is not None]
        left = wall_left()
        if strict_wall and left is not None:
            # Strict mode: the batch may not outlive the episode's budget. A
            # turn otherwise blocks on its SLOWEST member, so one straggler
            # sets the whole turn's latency. Unfinished calls are cancelled and
            # reported; the finished ones are kept.
            tasks = [asyncio.ensure_future(c) for c in budgeted if c is not None]
            done, pending = await asyncio.wait(tasks, timeout=max(left, 1.0))
            for t in pending:
                t.cancel()
            executed = []
            for t, i in zip(tasks, idxs):
                if t in done and not t.cancelled():
                    executed.append(t.result())
                else:
                    n_calls_cut += 1
                    executed.append((
                        f"call-{turn_idx + 1}-{i + 1}",
                        "not completed: this call was still running when the "
                        "episode's time budget ran out, and was stopped. Use "
                        "the results you did get.",
                        # The cost here is genuinely UNKNOWN — the call was
                        # billed in part but never reported back — so the
                        # episode's cost becomes a lower bound, not a fiction.
                        {"tool": None, "wall_cut": True, "cost_usd": None,
                         "latency_s": round(max(left, 1.0), 3)}))
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)
        else:
            executed = await asyncio.gather(
                *(c for c in budgeted if c is not None))
        results = {}
        for (call_id, content, info), call_idx in zip(executed, idxs):
            results[call_idx] = (call_id, content, info)
        for call_idx, tc in enumerate(calls):
            if call_idx in results:
                call_id, content, info = results[call_idx]
            else:  # a duplicate: same content, its own id, $0 and no latency
                _, content, src = results[primary_of[call_idx]]
                call_id = (tc.get("id") if isinstance(tc, dict) else None) \
                    or f"call-{turn_idx + 1}-{call_idx + 1}"
                if isinstance(tc, dict):
                    tc["id"] = call_id  # keep assistant/tool ids paired
                info = {"tool": src.get("tool"), "deduped": True,
                        "latency_s": 0.0, "cost_usd": 0.0}
            tool_log.append(info)
            messages.append(tool_msg(content, call_id))
        # Budget nudge: appended to the turn's last tool result — environment
        # feedback at decision time, with no mid-thread system message that
        # might trip a provider. At most one of each tier per episode.
        turns_left = max_turns - turn_idx - 1
        ctx_frac = (ctx_tokens / max_context) if max_context else 0.0
        note = None
        # Time tier first: it is the only one that can fire on a turn where
        # turns and context both look healthy.
        left = wall_left()
        if left is not None and "wall" not in budget_notes \
                and left <= max_wall_s * (1 - WALL_SOFT_FRAC):
            note = wall_note(left)
            budget_notes["wall"] = turn_idx + 1
        elif turns_left >= 1 and "last" not in budget_notes:
            if turns_left == 1 or ctx_frac >= LAST_BUDGET_CTX_FRAC:
                what = ("this is your last tool turn" if turns_left == 1
                        else f"context is {int(ctx_frac * 100)}% used")
                note = budget_last_note(what)
                budget_notes["last"] = turn_idx + 1
            elif not budget_notes and (turns_left == 2
                                       or ctx_frac >= NEAR_BUDGET_CTX_FRAC):
                note = budget_near_note(turns_left,
                                        ctx_pct=int(ctx_frac * 100))
                budget_notes["near"] = turn_idx + 1
        if note:
            messages[-1]["content"] = \
                f"{messages[-1].get('content', '')}\n\n{note}"

    # The wall: every hard limit leaves the transcript with no final message,
    # so the episode scores zero even when the work is done. Spend one
    # no-tools call to turn it into an answer; if that call fails the episode
    # keeps its original outcome, so this can only add an answer, never take
    # one away.
    #
    # `error is None` is the discriminator that keeps it cheap. A context stop
    # we CHOSE (the max_context policy cap, checked before tools ran) leaves a
    # servable prompt, because max_context is set below the model's real
    # window. One the PROVIDER raised means the window is already exceeded and
    # a longer prompt would fail identically.
    if stop_reason in (STOP_MAX_TURNS, STOP_CONTEXT, STOP_WALL_CLOCK) \
            and error is None:
        what = {STOP_MAX_TURNS: "the turn budget is spent",
                STOP_CONTEXT: "the context budget is spent",
                STOP_WALL_CLOCK: "this research ran out of time"}[stop_reason]
        # The context path breaks BEFORE running the turn's tools, so the
        # transcript ends on an assistant message whose tool_calls have no
        # results — a shape the chat APIs reject outright. Answer them as
        # not-executed: the conversation stays well-formed, the model is told
        # why it has no results, and the $0 is known rather than missing.
        n_added = 0
        last = messages[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            for call_idx, tc in enumerate(last["tool_calls"]):
                n_added += 1
                call_id = (tc.get("id") if isinstance(tc, dict) else None) \
                    or f"call-limit-{call_idx + 1}"
                if isinstance(tc, dict):
                    tc["id"] = call_id
                messages.append(tool_msg(
                    f"not executed: {what}, so no tools ran this turn.",
                    call_id))
                tool_log.append({"tool": None, "not_executed": True,
                                 "latency_s": 0.0, "cost_usd": 0.0})
        messages.append(user_msg(forced_answer_note(what)))
        n_added += 1

        def _rewind():
            """Nothing usable came back: leave the transcript exactly as the
            loop left it, including the synthesized tool results."""
            del messages[len(messages) - n_added:]
            del tool_log[len(tool_log) - (n_added - 1):]

        try:
            # Its own allowance: on the wall-clock path the budget is already
            # spent, so inheriting the deadline would time this out instantly.
            msg, usage = await asyncio.wait_for(
                client.generate(model.model_id, messages, [], sampling),
                timeout=(min(FORCED_ANSWER_GRACE_S, max_wall_s * 0.25)
                         if strict_wall and max_wall_s
                         else FORCED_ANSWER_GRACE_S))
        except Exception as e:  # noqa: BLE001 — best effort, never raised
            _rewind()
            forced_answer = {"limit": stop_reason, "ok": False,
                             "error": f"{type(e).__name__}: {e}"}
        else:
            usage_log.append(usage or {})  # billed either way
            ok = bool((msg.get("content") or "").strip())
            if ok:
                messages.append(msg)
            else:
                _rewind()
            forced_answer = {"limit": stop_reason, "ok": ok}
    # stop_reason deliberately KEEPS the limit that was hit, and with it
    # truncated=True: the episode really did run out, and the run report must
    # keep showing that. What changed is that grading now finds an answer.

    llm_cost = _cost(usage_log, model.pricing)
    tool_cost_known = all(t.get("cost_usd") is not None for t in tool_log)
    tool_cost = round(sum(t["cost_usd"] for t in tool_log), 6) \
        if tool_cost_known else None
    total_cost = (round(llm_cost + tool_cost, 6)
                  if llm_cost is not None and tool_cost is not None else None)
    prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage_log)
    completion_tokens = sum(u.get("completion_tokens", 0) for u in usage_log)
    return Rollout(
        messages=messages,
        tools=specs,
        sample_id=row.id,
        gold=row.gold,
        model=model.name,
        sampling=sampling,
        n_turns=sum(1 for m in messages if m.get("role") == "assistant"),
        n_tokens=(prompt_tokens + completion_tokens) or None,
        truncated=stop_reason in (STOP_MAX_TURNS, STOP_CONTEXT,
                                  STOP_WALL_CLOCK),
        error=error,
        timing={
            # wall_s is measured INSIDE the concurrency semaphores (t0 is set
            # here, not at submission): in-flight episode latency, not queue
            # time. started_at/ended_at are epoch seconds — wall_s is monotonic
            # and unanchored, so these are what make run-level makespan and
            # throughput computable.
            "wall_s": round(time.monotonic() - t0, 3),
            "started_at": round(started_at, 3),
            "ended_at": round(time.time(), 3),
            "usage": usage_log,             # raw per-call usage
            "tool_calls": tool_log,         # per-call latency/cost/errors
            "llm_latency_s": llm_latencies,  # per model-call attempt, in order
            "llm_s": round(sum(llm_latencies), 3) if llm_latencies else None,
            "prompt_tokens": prompt_tokens or None,
            "completion_tokens": completion_tokens or None,
            "llm_cost_usd": llm_cost,
            "tool_cost_usd": tool_cost,
            "cost_usd": total_cost,
            "n_empty_retries": n_empty_retries,
            "n_model_retries": n_model_retries,
            # snapshot, so historical cost survives price changes
            "pricing": dict(model.pricing),
        },
        meta={**row.meta, "bench": bench,
              "episode_id": episode_id, "attempt": attempt_idx,
              "stop_reason": stop_reason,
              **({"budget_notes": budget_notes} if budget_notes else {}),
              **({"max_calls_per_turn": max_calls_per_turn}
                 if max_calls_per_turn is not None else {}),
              **({"max_wall_s": max_wall_s} if max_wall_s else {}),
              **({"n_calls_capped": n_calls_capped} if n_calls_capped else {}),
              **({"n_calls_deduped": n_calls_deduped}
                 if n_calls_deduped else {}),
              **({"n_calls_cut": n_calls_cut} if n_calls_cut else {}),
              **({"strict_wall": True} if strict_wall else {}),
              **({"forced_answer": forced_answer} if forced_answer else {})},
    )


def accepted_attempts(path):
    """rollouts.jsonl -> {sample_id: accepted Rollout} (latest line wins)."""
    accepted = {}
    try:
        for r in io.read_rollouts(path):
            accepted[r.sample_id] = r
    except FileNotFoundError:
        pass
    return accepted


async def run_bench_model(bench, model, rows, out_path, tools,
                          max_turns=DEFAULT_MAX_TURNS,
                          max_context=DEFAULT_MAX_CONTEXT,
                          max_calls_per_turn=DEFAULT_MAX_CALLS_PER_TURN,
                          max_wall_s=DEFAULT_MAX_WALL_S, strict_wall=False,
                          concurrency=DEFAULT_CONCURRENCY,
                          client=None, progress=None, sem=None,
                          model_sem=None, system_suffix=None):
    """The rollout stage for one (benchmark, model) pair: resume-aware,
    concurrent, each finished episode appended and flushed immediately.

    Pass a shared `sem` to cap in-flight episodes GLOBALLY across pairs
    running at the same time; without it each pair gets its own cap and the
    providers see the product.
    """
    from .models import get_client
    client = client or get_client(model)
    accepted = accepted_attempts(out_path)
    todo = [r for r in rows
            if r.id not in accepted or accepted[r.id].error is not None]
    attempt_base = {r.id: accepted[r.id].meta.get("attempt", 0) + 1
                    for r in rows if r.id in accepted}
    sem = sem or asyncio.Semaphore(concurrency)
    # Model.concurrency caps this model's in-flight episodes under the global
    # cap, for providers with their own request-rate ceiling. Callers running
    # several benchmarks for one model must pass a SHARED model_sem, else each
    # pair gets its own cap and the provider still sees n_pairs x cap.
    if model_sem is None and model.concurrency:
        model_sem = asyncio.Semaphore(model.concurrency)
    done = 0

    async def one(row):
        nonlocal done
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(sem)
            if model_sem:
                await stack.enter_async_context(model_sem)
            attempt_idx = attempt_base.get(row.id, 0)
            try:
                rollout = await run_episode(client, model, row, tools,
                                            max_turns=max_turns,
                                            max_context=max_context,
                                            max_calls_per_turn=max_calls_per_turn,
                                            max_wall_s=max_wall_s,
                                            strict_wall=strict_wall,
                                            attempt_idx=attempt_idx,
                                            bench=bench.name,
                                            system_suffix=system_suffix)
            except Exception as e:  # noqa: BLE001 — preserve the rest of the batch
                rollout = Rollout(
                    messages=[dict(m) for m in row.messages],
                    sample_id=row.id, gold=row.gold, model=model.name,
                    sampling=dict(model.sampling), truncated=False,
                    error=f"{type(e).__name__}: {e}",
                    meta={**row.meta, "bench": bench.name,
                          "episode_id": f"{row.id}/{attempt_idx}",
                          "attempt": attempt_idx,
                          "stop_reason": STOP_INTERNAL_ERROR},
                )
        io.write_rollouts(out_path, [rollout], append=True)
        done += 1
        if progress:
            progress(done, len(todo), rollout)
        return rollout

    results = await asyncio.gather(*(one(r) for r in todo))
    return {"path": str(out_path), "n_rows": len(rows), "n_ran": len(todo),
            "n_skipped": len(rows) - len(todo),
            "n_errors": sum(1 for r in results if r.error)}
