"""Evaluation ids and the frozen manifest.

One invocation is one evaluation, identified by one `eval_id`, over a
(benchmarks x models) matrix. The manifest freezes everything a number depends
on — dataset content hashes, selected sample ids, model configs, grader
identity, harness and prompt hashes, tool configuration, protocol caps — and
comparisons are only valid within a single eval_id.

Resuming validates the current configuration against the frozen manifest and
hard-fails on any drift. That is the point: an agentic evaluation costs real
money and takes hours, so the failure mode worth engineering against is not a
crash but a run that silently continues under changed settings and produces a
table nobody can interpret afterwards.

Code drift has an explicit override, which records a provenance event in the
evaluation directory rather than passing silently.
"""

import hashlib
import json
import secrets
import subprocess
import time
from pathlib import Path

from .paths import RUNS_DIR

# Excluded from the hard comparison: provenance and labels, not configuration
# a number depends on. `execution` (the concurrency caps) belongs here too —
# latency depends on it, which is why it is recorded at all, but a score does
# not, so resuming at a different concurrency stays legal.
_VOLATILE = ("eval_id", "created", "code", "description", "execution")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def new_eval_id():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) \
        + "-" + secrets.token_hex(2)


def git_sha():
    """Short commit of the checkout this package lives in, marked dirty with a
    hash of the uncommitted diff. Provenance only — never correctness."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=_PACKAGE_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"],
                                cwd=_PACKAGE_ROOT, capture_output=True,
                                check=True).stdout
        if not status:
            return sha
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"],
                              cwd=_PACKAGE_ROOT, capture_output=True,
                              check=True).stdout
        dirty_hash = hashlib.sha256(status + diff).hexdigest()[:10]
        return f"{sha}-dirty-{dirty_hash}"
    except Exception:  # noqa: BLE001 — not a git checkout, or no git installed
        return "unknown"


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, default=str).encode()).hexdigest()


def build(benches, models, grader, protocol, description=None,
          bench_graders=None, harness=None, execution=None,
          harness_tools=None) -> dict:
    """Freeze the configuration for one evaluation.

    `benches` and `models` are resolved objects; `grader` comes from
    `grading.grader_config()`; `protocol` holds the caps and the task date.
    `bench_graders` freezes the per-benchmark grader configs, since the
    benchmarks do not all use the same grade kind. `harness_tools` overrides
    `harness.tools()` as the toolset to freeze — needed for the orchestrated
    harness, whose research tool depends on runtime wiring the CLI supplies.
    """
    dataset_manifests = {}
    selections = {}
    for b in benches:
        dataset_manifests[b.name] = b.dataset_manifest()
        selected = b.load_rows(protocol.get("n"))
        selections[b.name] = {
            "n": len(selected),
            "sample_ids_hash": canonical_hash([r.id for r in selected]),
        }
    m = {
        "v": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": git_sha(),
        "description": description,
        "benches": {b.name: {"agentic": b.agentic, "max_turns": b.max_turns,
                             "dataset": dataset_manifests[b.name],
                             "selection": selections[b.name],
                             **({"grader": bench_graders[b.name]}
                                if bench_graders and b.name in bench_graders
                                else {})}
                    for b in benches},
        "models": {m_.name: m_.to_config() for m_ in models},
        "grader": grader,
        "protocol": protocol,
        "execution": execution or {},
        "tools": ({t.spec["name"]: t.config
                   for t in (harness_tools if harness_tools is not None
                             else harness.tools())}
                  if harness and any(b.agentic for b in benches) else {}),
    }
    m["manifest_hash"] = canonical_hash(
        {k: v for k, v in m.items() if k not in _VOLATILE})
    return m


def eval_dir(eval_id) -> Path:
    return RUNS_DIR / eval_id


def freeze(eval_id, manifest):
    d = eval_dir(eval_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")


def load(eval_id) -> dict:
    return json.loads((eval_dir(eval_id) / "manifest.json").read_text())


def check_resume(eval_id, current, allow_code_drift=False) -> dict:
    """Validate `current` (a build() result) against the frozen manifest.
    Returns the frozen manifest; raises on hard drift."""
    frozen = load(eval_id)
    if frozen["manifest_hash"] != current["manifest_hash"]:
        drifted = [k for k in current
                   if k not in _VOLATILE + ("manifest_hash",)
                   and current.get(k) != frozen.get(k)]
        raise ValueError(
            f"resume config drift in {drifted} for eval {eval_id}; "
            "start a new eval_id (comparisons only hold within one manifest)")
    # Concurrency is not hard drift — it moves latency, not scores — but the
    # frozen block would otherwise describe only the FIRST run's conditions,
    # so a change is recorded rather than dropped.
    if (current.get("execution") or {}) != (frozen.get("execution") or {}):
        _record_event(eval_id, {"from_execution": frozen.get("execution"),
                                "to_execution": current.get("execution"),
                                "manifest_hash": frozen.get("manifest_hash")})
    if frozen.get("code") != current.get("code"):
        if not allow_code_drift:
            raise ValueError(
                f"resume code drift for eval {eval_id}: {frozen.get('code')} "
                f"-> {current.get('code')}; start a new eval or pass "
                "--allow-code-drift")
        _record_event(eval_id, {"from_code": frozen.get("code"),
                                "to_code": current.get("code"),
                                "manifest_hash": frozen.get("manifest_hash")})
        print(f"[manifest] warning: code drift allowed and recorded "
              f"for eval {eval_id}")
    return frozen


def list_eval_ids():
    """Every local eval_id with a frozen manifest, sorted — which is also
    chronological, since ids are timestamp-prefixed."""
    if not RUNS_DIR.exists():
        return []
    return sorted(d.name for d in RUNS_DIR.iterdir()
                  if (d / "manifest.json").exists())


def _record_event(eval_id, event):
    """Append one provenance event to the evaluation's resume_events.jsonl."""
    event = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **event}
    with open(eval_dir(eval_id) / "resume_events.jsonl", "a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
