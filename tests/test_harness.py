"""Offline tests for the episode loop and the graders — no network, no keys.

These cover the behaviors that are easy to break and expensive to notice: the
budget nudges, within-turn deduplication, the forced-answer salvage, and the
table grader's parsing and scoring.
"""

import asyncio
import json

import pytest

from nextsearch import harnesses
from nextsearch.grading import (extract_markdown_table, parse_verdict,
                                score_ws_tables, token_f1, ws_judge_scores)
from nextsearch.harness import STOP_FINAL, STOP_MAX_TURNS, run_episode
from nextsearch.models import Model
from nextsearch.tools import Tool
from nextsearch.types import Row, assistant, system, tool_call, user


class ScriptedClient:
    """Replays a fixed list of assistant messages, recording what it saw."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.calls = []

    async def generate(self, model_id, messages, tools, sampling):
        self.calls.append([dict(m) for m in messages])
        msg = self._messages.pop(0) if self._messages else assistant("done")
        return msg, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}


def _tool(name="search", record=None):
    async def execute(args):
        if record is not None:
            record.append(args)
        return f"result for {args}", {"tool": name, "cost_usd": 0.001}
    return Tool(spec={"name": name, "description": "", "parameters": {}},
                execute=execute)


MODEL = Model(name="test", client="openai", model_id="test",
              sampling={"temperature": 0.0})


def _row():
    return Row(messages=[system("be helpful"), user("what is X?")],
               gold={"answer": "42"}, id="t-000", meta={"question": "X?"})


def test_final_answer_ends_the_episode():
    client = ScriptedClient([assistant("the answer is 42")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool()]))
    assert r.meta["stop_reason"] == STOP_FINAL
    assert r.truncated is False
    assert r.messages[-1]["content"] == "the answer is 42"


def test_system_suffix_is_appended_to_the_row_prompt():
    client = ScriptedClient([assistant("ok")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool()],
                                system_suffix="Today's date is 2026-07-31."))
    assert r.messages[0]["role"] == "system"
    assert "be helpful" in r.messages[0]["content"]
    assert "Today's date is 2026-07-31." in r.messages[0]["content"]


def test_identical_calls_in_one_turn_execute_once():
    seen = []
    call = tool_call("search", {"objective": "same"}, id="a")
    dup = tool_call("search", {"objective": "same"}, id="b")
    client = ScriptedClient([assistant("", tool_calls=[call, dup]),
                             assistant("done")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool(record=seen)]))
    assert len(seen) == 1, "the duplicate should not have been executed"
    # Both calls still get a tool message, so the transcript stays well formed.
    tool_msgs = [m for m in r.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert r.meta["n_calls_deduped"] == 1


def test_per_turn_call_cap_skips_without_executing():
    seen = []
    calls = [tool_call("search", {"objective": f"q{i}"}, id=str(i))
             for i in range(5)]
    client = ScriptedClient([assistant("", tool_calls=calls),
                             assistant("done")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool(record=seen)],
                                max_calls_per_turn=2))
    assert len(seen) == 2
    assert r.meta["n_calls_capped"] == 3
    skipped = [m for m in r.messages
               if m["role"] == "tool" and "turn call cap" in m["content"]]
    assert len(skipped) == 3


def test_last_turn_nudge_is_injected():
    calls = [tool_call("search", {"objective": "q"}, id="1")]
    client = ScriptedClient([assistant("", tool_calls=calls),
                             assistant("", tool_calls=calls),
                             assistant("answer")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool()], max_turns=3))
    text = "\n".join(m.get("content") or "" for m in r.messages)
    assert "harness note" in text
    assert r.meta["budget_notes"]


def test_turn_budget_exhaustion_salvages_an_answer():
    """Running out of turns mid-tool-call must still produce a final answer:
    the research is already in the transcript, and without the salvage the
    episode would score zero."""
    calls = [tool_call("search", {"objective": "q"}, id="1")]
    client = ScriptedClient([assistant("", tool_calls=calls),
                             assistant("salvaged answer")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool()], max_turns=1))
    assert r.meta["stop_reason"] == STOP_MAX_TURNS
    assert r.truncated is True, "the limit was still hit and must be reported"
    assert r.meta["forced_answer"]["ok"] is True
    assert r.messages[-1]["content"] == "salvaged answer"


def test_tool_error_is_fed_back_not_raised():
    async def boom(args):
        raise RuntimeError("provider exploded")
    bad = Tool(spec={"name": "search", "description": "", "parameters": {}},
               execute=boom)
    calls = [tool_call("search", {"objective": "q"}, id="1")]
    client = ScriptedClient([assistant("", tool_calls=calls),
                             assistant("recovered")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [bad]))
    tool_msg = [m for m in r.messages if m["role"] == "tool"][0]
    assert "tool error" in tool_msg["content"]
    assert r.meta["stop_reason"] == STOP_FINAL
    # A raised call was never billed, so its cost is known to be zero rather
    # than missing — otherwise the episode drops out of the cost table.
    assert r.timing["tool_cost_usd"] == 0.0


def test_unknown_tool_is_reported_to_the_model():
    calls = [tool_call("nope", {}, id="1")]
    client = ScriptedClient([assistant("", tool_calls=calls),
                             assistant("ok")])
    r = asyncio.run(run_episode(client, MODEL, _row(), [_tool()]))
    tool_msg = [m for m in r.messages if m["role"] == "tool"][0]
    assert "unknown tool" in tool_msg["content"]


# --------------------------------------------------------------------------
# harness configuration


def test_both_harnesses_carry_a_date_and_guidelines():
    for name in ("solo", "orchestrated"):
        suffix = harnesses.get(name).system_suffix("2026-07-31")
        assert suffix.startswith("Today's date is 2026-07-31.")
        assert len(suffix) > 100


def test_solo_harness_exposes_search_and_fetch():
    tools = harnesses.get("solo").tools()
    assert sorted(t.spec["name"] for t in tools) == ["fetch", "search"]


def test_tool_alias_renames_specs_and_guidelines():
    h = harnesses.get("solo")
    alias = {"search": "web_search", "fetch": "read_url"}
    tools = h.tools(tool_alias=alias)
    assert sorted(t.spec["name"] for t in tools) == ["read_url", "web_search"]
    doc = h.system_suffix(tool_alias=alias)
    assert "`web_search`" in doc and "`search`" not in doc
    # The unaliased toolset must be unaffected: the spec dicts are shared
    # module constants and renaming must never mutate them.
    assert sorted(t.spec["name"] for t in h.tools()) == ["fetch", "search"]


def test_unknown_tool_alias_is_an_error():
    with pytest.raises(ValueError):
        harnesses.get("solo").tools(tool_alias={"nope": "x"})


def test_orchestrated_harness_requires_subagent_wiring():
    with pytest.raises(ValueError):
        harnesses.get("orchestrated").tools()


# --------------------------------------------------------------------------
# graders


def test_verdict_parsing_handles_bare_letters_and_words():
    assert parse_verdict("A") == "CORRECT"
    assert parse_verdict("  b ") == "INCORRECT"
    assert parse_verdict("NOT ATTEMPTED") == "NOT_ATTEMPTED"
    assert parse_verdict("The grade is: CORRECT.") == "CORRECT"
    assert parse_verdict("who knows") == "UNPARSEABLE"


def test_token_f1_guards_yes_no():
    assert token_f1("yes", "yes") == 1.0
    assert token_f1("yes", "no") == 0.0
    assert 0 < token_f1("Paris, France", "Paris") <= 1.0


def test_table_extraction_from_a_fenced_block():
    text = ("here you go\n```markdown\n| Name | Year |\n|---|---|\n"
            "| Ada | 1815 |\n| Alan | 1912 |\n```\n")
    header, rows = extract_markdown_table(text)
    assert header == ["name", "year"]
    assert rows == [["Ada", "1815"], ["Alan", "1912"]]


def test_table_extraction_returns_none_without_a_table():
    assert extract_markdown_table("just prose, no table here") is None


def test_judge_silence_is_reported_not_scored_zero():
    """A judge that lists only failures plus a catch-all must not zero every
    unlisted row, and rows nobody covered must be reported."""
    scores, unjudged = ws_judge_scores({"idx_0": 0, "idx_others": 1}, 4)
    assert scores == [0.0, 1.0, 1.0, 1.0]
    assert unjudged == []
    scores, unjudged = ws_judge_scores({"idx_0": 1}, 3)
    assert scores == [1.0, 0.0, 0.0]
    assert unjudged == [1, 2]


def test_row_f1_takes_the_minimum_over_a_rows_cells():
    spec = {"required": ["name", "year", "city"], "unique_columns": ["name"]}
    gold = [{"name": "a", "year": "1", "city": "x"},
            {"name": "b", "year": "2", "city": "y"}]
    # Row "a" has one wrong cell, so it scores 0 even though its other cell is
    # right; row "b" is perfect.
    resp = [{"name": "a", "year": "1", "city": "WRONG"},
            {"name": "b", "year": "2", "city": "y"}]
    scores = score_ws_tables(gold, resp, spec,
                             {"year": [1.0, 1.0], "city": [0.0, 1.0]})
    assert scores["row_f1"] == 0.5
    assert scores["n_matched_rows"] == 2
    # The strict success rate compares the tables cell by cell, so a wrong
    # cell also loses it.
    assert scores["sr"] == 0


def test_perfect_table_scores_one():
    spec = {"required": ["name", "year"], "unique_columns": ["name"]}
    gold = [{"name": "a", "year": "1"}]
    scores = score_ws_tables(gold, list(gold), spec, {"year": [1.0]})
    assert scores["row_f1"] == 1.0
    assert scores["sr"] == 1


def test_missing_rows_count_against_recall():
    spec = {"required": ["name", "year"], "unique_columns": ["name"]}
    gold = [{"name": "a", "year": "1"}, {"name": "b", "year": "2"}]
    resp = [{"name": "a", "year": "1"}]
    scores = score_ws_tables(gold, resp, spec, {"year": [1.0]})
    assert scores["row_precision"] == 1.0
    assert scores["row_recall"] == 0.5


# --------------------------------------------------------------------------
# shipped data assets


def test_widesearch_sub_golds_are_well_formed():
    from nextsearch.benchmarks import widesearch_sub as ws
    records = [json.loads(line) for line in
               ws.DATA_FILE.read_text().split("\n") if line.strip()]
    assert len(records) == 49
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    for r in records:
        assert r["gold_table"], f"{r['id']}: empty gold table"
        assert r["unique_columns"], f"{r['id']}: no key column"
        assert set(r["unique_columns"]) <= set(r["columns"])
        assert r.get("as_of"), f"{r['id']}: gold has no as-of date"
        assert len(r.get("evidence") or []) >= 2, \
            f"{r['id']}: fewer than two evidence URLs"


def test_gold_revisions_are_valid():
    from nextsearch.benchmarks import revisions
    for bench in ("seal0", "frames", "deepsearchqa", "widesearch"):
        records = revisions.load(revisions.path_for(bench))
        assert records, f"{bench}: no revisions"
        for rid, rec in records.items():
            assert rec["verdict"] in revisions.ALL_VERDICTS
            if rec["verdict"] in revisions.REVISING_VERDICTS:
                assert rec.get("new_answer"), f"{rid}: revision without answer"
