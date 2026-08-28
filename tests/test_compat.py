"""Offline tests for the `nextsearch-compat` probe — no server, no keys.

Fake clients stand in for a model server: a healthy one that emits a proper
tool call and then uses the tool result, and a text-only one that never calls
the tool (the exact failure the probe exists to catch)."""

import asyncio

from nextsearch.compat import (LOOKUP_KEY, SENTINEL, probe, run_probe)
from nextsearch.types import assistant, tool_call


class FakeHealthyClient:
    """Two well-formed turns: a `lookup` tool call, then a final answer that
    quotes the sentinel from the tool result. Optionally emits thinking."""

    def __init__(self, with_reasoning=False, arguments=None):
        self.with_reasoning = with_reasoning
        self.arguments = {"key": LOOKUP_KEY} if arguments is None else arguments
        self.calls = 0

    async def generate(self, model_id, messages, tools, sampling):
        self.calls += 1
        if self.calls == 1:
            msg = assistant(
                "", reasoning_content=("I should look that up."
                                       if self.with_reasoning else None),
                tool_calls=[tool_call("lookup", self.arguments, id="call-1")])
            return msg, {"prompt_tokens": 40, "completion_tokens": 12,
                         "total_tokens": 52,
                         "completion_tokens_details": {"reasoning_tokens": 5}}
        msg = assistant(f"The capital of Zubrowka is {SENTINEL}.")
        return msg, {"prompt_tokens": 70, "completion_tokens": 9,
                     "total_tokens": 79}


class FakeTextOnlyClient:
    """Never calls a tool — answers straight from priors. This is the server
    (or model) the probe must reject."""

    def __init__(self):
        self.calls = 0

    async def generate(self, model_id, messages, tools, sampling):
        self.calls += 1
        return assistant("The capital is Paris."), {"prompt_tokens": 40,
                                                    "completion_tokens": 6}


class FakeWrongNameClient:
    """Emits a tool call for a different function than the one offered."""

    async def generate(self, model_id, messages, tools, sampling):
        return (assistant("", tool_calls=[tool_call("search", {"q": "x"},
                                                     id="c1")]),
                {"prompt_tokens": 10, "completion_tokens": 3})


class FakeRaisingClient:
    """A server that errors on the first call — a probe must report, not
    raise."""

    async def generate(self, model_id, messages, tools, sampling):
        raise RuntimeError("connection refused")


def test_healthy_client_passes_all_required_checks():
    client = FakeHealthyClient(with_reasoning=True)
    r = asyncio.run(probe(client, "fake"))
    assert r["passed"] is True
    assert client.calls == 2, "the probe must make exactly two model calls"
    names = {c["name"]: c["ok"] for c in r["checks"]}
    for required in ("first_call_succeeded", "returned_tool_call",
                     "tool_name_matches", "tool_call_id_present",
                     "arguments_parse_as_json", "second_call_succeeded",
                     "final_answer_uses_tool_result"):
        assert names[required] is True, required
    assert r["reasoning_content_captured"] is True
    assert r["latency_s"]["first_call"] is not None
    assert r["usage"]["first_call"]["reasoning_tokens"] == 5


def test_healthy_without_reasoning_still_passes():
    r = asyncio.run(probe(FakeHealthyClient(with_reasoning=False), "fake"))
    assert r["passed"] is True
    # Reasoning capture is reported, not required.
    assert r["reasoning_content_captured"] is False


def test_text_only_client_fails_on_missing_tool_call():
    client = FakeTextOnlyClient()
    r = asyncio.run(probe(client, "fake"))
    assert r["passed"] is False
    names = {c["name"]: c["ok"] for c in r["checks"]}
    assert names["returned_tool_call"] is False
    # No tool call means the probe never makes the second call.
    assert client.calls == 1


def test_wrong_tool_name_fails():
    r = asyncio.run(probe(FakeWrongNameClient(), "fake"))
    assert r["passed"] is False
    names = {c["name"]: c["ok"] for c in r["checks"]}
    assert names["tool_name_matches"] is False


def test_client_exception_is_reported_not_raised():
    r = asyncio.run(probe(FakeRaisingClient(), "fake"))
    assert r["passed"] is False
    first = next(c for c in r["checks"] if c["name"] == "first_call_succeeded")
    assert first["ok"] is False
    assert "connection refused" in (first.get("detail") or "")


def test_run_probe_works_inside_a_running_event_loop():
    """Colab and Jupyter run cells inside a live loop, where asyncio.run
    raises. run_probe must still work by falling back to a worker thread."""
    async def call_from_within_a_loop():
        return run_probe(client=FakeHealthyClient(with_reasoning=True),
                         model_id="fake")
    r = asyncio.run(call_from_within_a_loop())
    assert r["passed"] is True
    assert r["model"]["model_id"] == "fake"


def test_final_answer_without_sentinel_fails():
    class NoSentinel(FakeHealthyClient):
        async def generate(self, model_id, messages, tools, sampling):
            self.calls += 1
            if self.calls == 1:
                return (assistant("", tool_calls=[tool_call(
                    "lookup", {"key": LOOKUP_KEY}, id="c1")]),
                    {"prompt_tokens": 1, "completion_tokens": 1})
            return assistant("I could not determine it."), {}
    r = asyncio.run(probe(NoSentinel(), "fake"))
    assert r["passed"] is False
    names = {c["name"]: c["ok"] for c in r["checks"]}
    assert names["final_answer_uses_tool_result"] is False
