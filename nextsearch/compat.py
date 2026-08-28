"""`nextsearch-compat`: a model/server tool-calling compatibility probe.

Before spending search credits or GPU hours on a rollout, one question decides
whether a served model can run in the harness at all: does its server emit
OpenAI-format tool calls that this harness can parse, dispatch, and feed back?
A model that free-texts its tool intentions, or a server whose tool-call parser
is misconfigured (the single most common vLLM setup mistake, see
docs/serving.md), fails silently as a wall of zero-scoring episodes.

This probe answers that question deterministically and cheaply. It uses a
SYNTHETIC `lookup` tool and a fixed prompt, so it needs no PARALLEL_API_KEY and
spends no search credits — only two calls to the model server itself. It:

  1. sends a prompt that can only be answered via the `lookup` tool;
  2. confirms a parsed OpenAI-format tool call came back;
  3. verifies the function name and that a tool-call id is present;
  4. parses and validates the JSON arguments;
  5. returns a synthetic tool result carrying a sentinel value;
  6. makes a second call;
  7. confirms the model used the sentinel in its final answer;
  8. reports whether `reasoning_content` was captured, plus per-call latency
     and token usage.

It prints one machine-readable JSON object and exits non-zero when any required
check fails, so it drops straight into a notebook cell or CI gate.
"""

import argparse
import asyncio
import json
import sys
import time

from .types import assistant, tool as tool_msg, tool_call_args, tool_spec

# The synthetic tool. Its only job is to be un-guessable: the answer lives
# behind the call, so a model that free-texts instead of calling it cannot
# pass. `key` is a required string argument, which is what lets the probe
# validate JSON-argument parsing rather than merely "a call happened".
LOOKUP_TOOL = tool_spec(
    "lookup",
    "Look up the authoritative value for a named key in the private "
    "knowledge base. The value is NOT available any other way — you must call "
    "this tool to answer.",
    {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to look up, exactly as given by the "
                               "user (for example 'zubrowka_capital')."},
        },
        "required": ["key"],
    },
)

LOOKUP_KEY = "zubrowka_capital"
# The sentinel the synthetic tool returns. It is deliberately not a real place
# and not memorizable, so finding it in the final answer proves the model
# consumed the tool result rather than answering from priors.
SENTINEL = "Zubrownistadt-7Q"

SYSTEM_PROMPT = (
    "You are a precise assistant with access to tools. When a fact is only "
    "available through a tool, you MUST call the tool rather than guessing. "
    "Call one tool at a time and wait for its result.")

USER_PROMPT = (
    f"I need the capital city of the country Zubrowka. That fact is only in "
    f"the private knowledge base under the key '{LOOKUP_KEY}'. Use the lookup "
    f"tool to retrieve it, then tell me the capital city by name.")


def _usage_brief(usage):
    """The few usage fields worth surfacing, tolerant of servers that omit
    some of them. `reasoning_tokens` lives inside completion_tokens_details on
    servers that report it."""
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


def _check(checks, name, ok, detail=None):
    checks.append({"name": name, "ok": bool(ok),
                   **({"detail": detail} if detail is not None else {})})
    return bool(ok)


async def probe(client, model_id, sampling=None) -> dict:
    """Run the two-call compatibility probe against a Client.

    Returns a machine-readable result dict: `passed` (all required checks
    green), the per-check list, captured latency/usage for both calls, and
    whether thinking was captured. Never raises for a model/server failure —
    an exception during a call is recorded as a failed check so the report is
    always well formed.
    """
    sampling = dict(sampling or {})
    checks = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}]
    reasoning_seen = False
    timings, usages = [], []

    # ---- Call 1: expect a tool call --------------------------------------
    t0 = time.monotonic()
    try:
        msg, usage = await client.generate(model_id, messages,
                                            [LOOKUP_TOOL], sampling)
        call1_error = None
    except Exception as e:  # noqa: BLE001 — recorded as a failed check
        msg, usage, call1_error = None, {}, f"{type(e).__name__}: {e}"
    timings.append(round(time.monotonic() - t0, 3))
    usages.append(_usage_brief(usage))
    _check(checks, "first_call_succeeded", call1_error is None, call1_error)

    tc = None
    if msg is not None:
        reasoning_seen = reasoning_seen or bool(msg.get("reasoning_content"))
        calls = msg.get("tool_calls") or []
        if _check(checks, "returned_tool_call", bool(calls),
                  None if calls else "model returned no tool call — it likely "
                  "answered from priors, or the server's tool-call parser is "
                  "not configured (see docs/serving.md)"):
            tc = calls[0]

    parsed_args = None
    if tc is not None:
        fn = tc.get("function") if isinstance(tc, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        _check(checks, "tool_name_matches", name == "lookup",
               None if name == "lookup" else f"expected 'lookup', got {name!r}")
        _check(checks, "tool_call_id_present", bool(tc.get("id")),
               None if tc.get("id") else "tool call has no id; the harness "
               "keys assistant/tool message pairs on it")
        try:
            parsed_args = tool_call_args(tc)
            ok_args = isinstance(parsed_args, dict)
            _check(checks, "arguments_parse_as_json", ok_args,
                   None if ok_args else f"arguments are not a JSON object: "
                   f"{parsed_args!r}")
        except Exception as e:  # noqa: BLE001
            _check(checks, "arguments_parse_as_json", False,
                   f"{type(e).__name__}: {e}")

    # ---- Call 2: feed a synthetic result, expect it to be used -----------
    passed_final = False
    if tc is not None:
        call_id = tc.get("id") or "compat-lookup-1"
        if isinstance(tc, dict):
            tc["id"] = call_id
        messages.append(assistant(msg.get("content") or "", tool_calls=[tc]))
        messages.append(tool_msg(
            json.dumps({"key": LOOKUP_KEY,
                        "value": f"The capital of Zubrowka is {SENTINEL}."}),
            call_id))
        t1 = time.monotonic()
        try:
            msg2, usage2 = await client.generate(model_id, messages,
                                                 [LOOKUP_TOOL], sampling)
            call2_error = None
        except Exception as e:  # noqa: BLE001
            msg2, usage2, call2_error = None, {}, f"{type(e).__name__}: {e}"
        timings.append(round(time.monotonic() - t1, 3))
        usages.append(_usage_brief(usage2))
        _check(checks, "second_call_succeeded", call2_error is None,
               call2_error)
        if msg2 is not None:
            reasoning_seen = reasoning_seen or bool(
                msg2.get("reasoning_content"))
            final = (msg2.get("content") or "")
            passed_final = SENTINEL.lower() in final.lower()
            _check(checks, "final_answer_uses_tool_result", passed_final,
                   None if passed_final else "final answer did not contain the "
                   "sentinel from the tool result; the model may not be "
                   "consuming tool messages")

    required = [c for c in checks if c["name"] != "reasoning_content_captured"]
    passed = all(c["ok"] for c in required)
    return {
        "passed": passed,
        "checks": checks,
        "reasoning_content_captured": reasoning_seen,
        "latency_s": {"first_call": timings[0] if timings else None,
                      "second_call": timings[1] if len(timings) > 1 else None},
        "usage": {"first_call": usages[0] if usages else None,
                  "second_call": usages[1] if len(usages) > 1 else None},
        "sentinel": SENTINEL,
    }


def run_probe(model_name=None, base_url=None, model_id=None, client=None):
    """Resolve a model/client and run the probe synchronously. Either pass a
    registry `model_name` (resolved via the registry, honoring `base_url`) or
    an explicit `model_id` plus a `client`."""
    from .models import get_client, get_model
    if client is None:
        model = get_model(model_name, base_url=base_url)
        client = get_client(model)
        model_id = model.model_id
        sampling = model.sampling
        resolved = {"model": model.name, "model_id": model.model_id,
                    "client": model.client, "base_url": model.base_url}
    else:
        sampling = {}
        resolved = {"model": model_name or model_id, "model_id": model_id}
    result = asyncio.run(probe(client, model_id, sampling))
    result["model"] = resolved
    return result


def main(argv=None):
    from .paths import load_env
    load_env()
    ap = argparse.ArgumentParser(
        prog="nextsearch-compat",
        description="Probe whether a served model speaks OpenAI-format tool "
                    "calls the NextSearch harness can use. Uses a synthetic "
                    "tool — no PARALLEL_API_KEY, no search credits, two model "
                    "calls. Prints JSON; exits non-zero on failure.")
    ap.add_argument("--model", default="lfm2.5-2.6b",
                    help="registry model name (default: lfm2.5-2.6b) or a raw "
                         "'vendor/model' OpenRouter id")
    ap.add_argument("--base-url", default=None,
                    help="endpoint for self-hosted models, overriding "
                         "NEXTSEARCH_BASE_URL (e.g. http://localhost:8000/v1)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the JSON result, no human summary line")
    args = ap.parse_args(argv)

    result = run_probe(model_name=args.model, base_url=args.base_url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.quiet:
        n_ok = sum(c["ok"] for c in result["checks"])
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{verdict}] {n_ok}/{len(result['checks'])} checks green; "
              f"reasoning_content "
              f"{'captured' if result['reasoning_content_captured'] else 'not seen'}",
              file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
