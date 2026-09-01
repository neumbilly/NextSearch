"""Offline tests for the experiment layer: telemetry extraction from synthetic
rollouts, run logging, and secret redaction. No network, no keys."""

import json

from nextsearch.experiment import telemetry
from nextsearch.experiment.runlog import RunLogger, read_metrics
from nextsearch.types import assistant, system, tool as tool_msg, tool_call, user


def _search_call(cid):
    return tool_call("search", {"objective": "x",
                                "search_queries": ["a", "b", "c"]}, id=cid)


def _fetch_call(cid, urls):
    return tool_call("fetch", {"urls": urls, "objective": "y"}, id=cid)


def _rollout(**over):
    """A realistic single-episode rollout dict: one parallel turn (search +
    fetch), then a final answer."""
    base = {
        "sample_id": "t-1",
        "model": "lfm2.5-2.6b",
        "sampling": {"temperature": 0.1},
        "messages": [
            system("s"), user("q"),
            assistant("", reasoning_content="think",
                      tool_calls=[_search_call("a"),
                                  _fetch_call("b", ["https://www.example.com/p",
                                                    "https://foo.org/q"])]),
            tool_msg("res-a", "a"), tool_msg("res-b", "b"),
            assistant("final answer"),
        ],
        "n_turns": 2,
        "n_tokens": 300,
        "truncated": False,
        "error": None,
        "timing": {
            "wall_s": 10.0, "started_at": 1000.0, "ended_at": 1010.0,
            "usage": [
                {"prompt_tokens": 100, "completion_tokens": 50,
                 "completion_tokens_details": {"reasoning_tokens": 20}},
                {"prompt_tokens": 100, "completion_tokens": 50}],
            "tool_calls": [
                {"tool": "search", "latency_s": 1.0, "cost_usd": 0.001},
                {"tool": "fetch", "latency_s": 2.0, "cost_usd": 0.002}],
            "llm_s": 4.0, "prompt_tokens": 200, "completion_tokens": 100,
            "llm_cost_usd": 0.0, "tool_cost_usd": 0.003, "cost_usd": 0.003,
        },
        "meta": {"bench": "seal0", "episode_id": "t-1/0",
                 "stop_reason": "final"},
    }
    base.update(over)
    return base


def test_episode_metrics_from_a_realistic_rollout():
    m = telemetry.episode_metrics(_rollout(), gpu="L4", vllm_version="0.9.0",
                                  gpu_hourly_usd=0.72)
    assert m["benchmark"] == "seal0"
    assert m["task_id"] == "t-1"
    assert m["gpu"] == "L4" and m["vllm_version"] == "0.9.0"
    assert m["n_search_calls"] == 1 and m["n_fetch_calls"] == 1
    assert m["n_tool_calls"] == 2
    assert m["max_calls_per_turn"] == 2 and m["mean_calls_per_turn"] == 2
    assert m["n_parallel_turns"] == 1 and m["has_parallel_calls"] is True
    assert m["n_unique_fetch_urls"] == 2 and m["n_unique_domains"] == 2
    assert m["reasoning_chars"] == len("think")
    assert m["reasoning_tokens"] == 20      # only the first usage reported it
    assert m["prompt_tokens"] == 200 and m["completion_tokens"] == 100
    assert m["total_tokens"] == 300
    assert m["model_latency_s"] == 4.0 and m["tool_latency_s"] == 3.0
    assert m["output_tps"] == 25.0          # 100 completion / 4.0 model-seconds
    assert m["search_usd"] == 0.001 and m["fetch_usd"] == 0.002
    assert m["final_answer"] is True and m["error"] is False
    # GPU dollars for this episode: rate * wall / 3600.
    assert m["gpu"] == "L4"


def test_reasoning_tokens_none_when_server_never_reports_them():
    r = _rollout()
    for u in r["timing"]["usage"]:
        u.pop("completion_tokens_details", None)
    m = telemetry.episode_metrics(r)
    assert m["reasoning_tokens"] is None    # distinct from 0
    assert m["reasoning_chars"] == len("think")


def test_deduped_and_budget_calls_do_not_count_as_executed():
    r = _rollout()
    r["timing"]["tool_calls"] = [
        {"tool": "search", "latency_s": 1.0, "cost_usd": 0.001},
        {"tool": "search", "deduped": True, "latency_s": 0.0, "cost_usd": 0.0},
        {"tool": "search", "budget_exhausted": True, "cost_usd": 0.0,
         "latency_s": 0.0},
    ]
    m = telemetry.episode_metrics(r)
    assert m["n_search_calls"] == 1, "only the executed call counts"


def test_judge_cost_joins_from_a_grade_sidecar():
    grade = {"episode_id": "t-1/0", "judge_usage": {"cost": 0.004}}
    m = telemetry.episode_metrics(_rollout(), grade=grade)
    assert m["judge_usd"] == 0.004
    assert telemetry.episode_metrics(_rollout())["judge_usd"] is None


def test_aggregate_rates_and_throughput():
    good = telemetry.episode_metrics(_rollout())
    # A truncated-but-salvaged episode: still a final answer, and truncated.
    trunc = telemetry.episode_metrics(_rollout(
        sample_id="t-2", truncated=True,
        meta={"bench": "seal0", "episode_id": "t-2/0",
              "stop_reason": "max_turns", "forced_answer": {"ok": True}}))
    # A hard error episode.
    err = telemetry.episode_metrics(_rollout(
        sample_id="t-3", error="RuntimeError: boom",
        meta={"bench": "seal0", "episode_id": "t-3/0",
              "stop_reason": "model_error"}))
    agg = telemetry.aggregate([good, trunc, err], gpu_hourly_usd=0.72)
    assert agg["n_episodes"] == 3
    assert agg["error_rate"] == round(1 / 3, 4)
    assert agg["truncation_rate"] == round(1 / 3, 4)
    # good + trunc produced a final answer; err did not.
    assert agg["final_answer_rate"] == round(2 / 3, 4)
    # success = final answer AND no error -> good + trunc.
    assert agg["success_rate"] == round(2 / 3, 4)
    assert agg["parallel_call_rate"] == 1.0
    assert agg["latency_p50_s"] is not None
    assert agg["aggregate_tps"] is not None
    assert agg["total_search_usd"] == round(3 * (0.001 + 0.002), 6)
    assert agg["est_gpu_usd_per_episode"] is not None


def test_aggregate_empty_is_safe():
    assert telemetry.aggregate([]) == {"n_episodes": 0}


def test_telemetry_cli_writes_json_and_csv(tmp_path):
    rollouts = tmp_path / "seal0" / "lfm2.5-2.6b" / "rollouts.jsonl"
    rollouts.parent.mkdir(parents=True)
    with open(rollouts, "w") as f:
        f.write(json.dumps(_rollout()) + "\n")
        f.write(json.dumps(_rollout(sample_id="t-2")) + "\n")
    out_json = tmp_path / "telemetry.json"
    out_csv = tmp_path / "telemetry.csv"
    rc = telemetry.main([str(tmp_path), "--gpu", "L4", "--vllm-version",
                         "0.9.0", "--gpu-hourly-usd", "0.72",
                         "--json", str(out_json), "--csv", str(out_csv)])
    assert rc == 0
    doc = json.loads(out_json.read_text())
    assert doc["aggregate"]["n_episodes"] == 2
    assert len(doc["episodes"]) == 2
    assert doc["config"]["gpu"] == "L4"
    csv_text = out_csv.read_text().strip().splitlines()
    assert len(csv_text) == 3          # header + two rows
    assert "n_search_calls" in csv_text[0]


def test_latest_line_wins_on_resume(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    with open(rollouts, "w") as f:
        f.write(json.dumps(_rollout(error="boom")) + "\n")   # first attempt
        f.write(json.dumps(_rollout()) + "\n")               # accepted retry
    eps = telemetry.rollout_telemetry(rollouts)
    assert len(eps) == 1 and eps[0]["error"] is False


# --------------------------------------------------------------------------
# run logging


def test_runlogger_streams_metrics_and_reads_back(tmp_path):
    with RunLogger(tmp_path, experiment="exp", run_id="r1") as log:
        log.log(step=1, phase="train", loss=1.5, lr=1e-4)
        log.log(step=2, phase="train", loss=1.2, lr=1e-4)
        log.log(phase="eval", accuracy=0.5)
    recs = read_metrics(tmp_path / "exp" / "r1")
    assert len(recs) == 3
    train = read_metrics(tmp_path / "exp" / "r1", phase="train")
    assert [r["loss"] for r in train] == [1.5, 1.2]
    assert all("wall" in r and "t_rel" in r for r in recs)


def test_runlogger_redacts_secrets_in_config(tmp_path):
    log = RunLogger(tmp_path, experiment="exp", run_id="r2",
                    config={"model": "lfm2.5-2.6b",
                            "PARALLEL_API_KEY": "sk-secret",
                            "nested": {"WANDB_TOKEN": "t", "gpu": "L4"}})
    doc = json.loads((log.dir / "run.json").read_text())
    cfg = doc["config"]
    assert cfg["model"] == "lfm2.5-2.6b"
    assert cfg["PARALLEL_API_KEY"] == "<redacted>"
    assert cfg["nested"]["WANDB_TOKEN"] == "<redacted>"
    assert cfg["nested"]["gpu"] == "L4"
    log.close()


def test_runlogger_mints_unique_dirs(tmp_path):
    a = RunLogger(tmp_path, experiment="exp")
    b = RunLogger(tmp_path, experiment="exp")
    assert a.dir != b.dir, "auto run ids must not collide or overwrite"
    a.close()
    b.close()


def test_runlogger_drops_nonfinite(tmp_path):
    with RunLogger(tmp_path, experiment="exp", run_id="r3") as log:
        log.log(step=1, phase="train", loss=float("nan"), reward=0.3)
    rec = read_metrics(tmp_path / "exp" / "r3")[0]
    assert "loss" not in rec and rec["reward"] == 0.3
