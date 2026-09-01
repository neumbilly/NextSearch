"""Tests for the episode trace utilities: the rendered transcript surfaces the
tool calls, and the raw trace round-trips through disk."""

import json

from nextsearch.experiment import trace
from nextsearch.types import assistant, system, tool as tool_msg, tool_call, user


def _rollout():
    return {
        "sample_id": "t-1", "model": "lfm2.5-2.6b",
        "messages": [
            system("guidelines"), user("what is X since 2023?"),
            assistant("", tool_calls=[
                tool_call("search", {"objective": "X since 2023",
                                     "search_queries": ["a", "b", "c"]}, id="s1")]),
            tool_msg("[1] Some Source — https://example.com\nexcerpt", "s1"),
            assistant("The answer is 11 unique players."),
        ],
        "n_turns": 2, "n_tokens": 300, "truncated": False, "error": None,
        "timing": {"wall_s": 12.0, "llm_s": 4.0, "llm_cost_usd": None,
                   "tool_cost_usd": 0.001, "tool_calls": [
                       {"tool": "search", "latency_s": 1.0, "cost_usd": 0.001}]},
        "meta": {"bench": "seal0", "episode_id": "t-1/0", "stop_reason": "final"},
    }


def test_tool_call_summary_counts_calls():
    s = trace.tool_call_summary(_rollout())
    assert s["by_tool"] == {"search": 1}
    assert s["total_calls"] == 1 and s["parallel_turns"] == 0


def test_format_trace_shows_the_tool_call_and_result_and_answer():
    text = trace.format_trace(_rollout())
    # The call, its parsed arguments, the result, and the final answer are all
    # visible — this is what makes "did it call search?" answerable by eye.
    assert "CALL search(" in text
    assert '"objective": "X since 2023"' in text
    assert "id=s1" in text
    assert "result id=s1" in text
    assert "11 unique players" in text
    assert "stop=final" in text
    assert "search" in text.split("tool calls:")[1].splitlines()[0]


def test_format_trace_truncates_long_results_but_can_show_full():
    r = _rollout()
    r["messages"][3] = {"role": "tool", "tool_call_id": "s1",
                        "content": "y" * 5000}
    clipped = trace.format_trace(r, result_chars=100)
    assert "+4900 chars" in clipped
    full = trace.format_trace(r, result_chars=None)
    assert "y" * 5000 in full


def test_save_and_load_raw_trace_round_trips(tmp_path):
    p = trace.save_trace(_rollout(), tmp_path / "smoke" / "t-1.trace.json")
    assert p.exists()
    back = trace.load_trace(p)
    assert back["sample_id"] == "t-1"
    assert back["messages"][-1]["content"] == "The answer is 11 unique players."


def test_save_as_rollouts_is_readable_by_telemetry(tmp_path):
    from nextsearch.experiment import rollout_telemetry
    p = tmp_path / "smoke" / "lfm2.5-2.6b" / "rollouts.jsonl"
    trace.save_as_rollouts(_rollout(), p)
    assert p.exists()
    eps = rollout_telemetry(p)
    assert len(eps) == 1
    assert eps[0]["n_search_calls"] == 1 and eps[0]["final_answer"] is True


def test_malformed_tool_call_is_shown_not_hidden():
    r = _rollout()
    r["messages"][2] = {"role": "assistant", "content": "",
                        "tool_calls": [{"id": "x", "type": "function",
                                        "function": {"name": "search",
                                                     "arguments": "{not json"}}]}
    text = trace.format_trace(r)
    assert "CALL search(" in text  # raw arg string is shown rather than crashing
