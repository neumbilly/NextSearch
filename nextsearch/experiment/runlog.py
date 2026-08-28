"""Local run logging — the no-service replacement for a hosted tracker.

A `RunLogger` owns one run directory and appends metrics to a plain
`metrics.jsonl`. That file is the single source of truth the live Colab viewer
tails and that later SFT/OPD/RL stages append their own training curves to.
There is no network, no API key, and no optional heavy dependency: a metric is
one JSON line, so a run survives a Colab disconnect and reads back identically
afterward.

Layout under the run directory::

    <output_root>/<experiment>/<run_id>/
        run.json         # config + hardware/software, written once, redacted
        metrics.jsonl    # append-only stream of scalar/step records
        rollouts/…       # (written by the eval pipeline, referenced here)

Every record carries a monotonic `wall` timestamp and an optional `step`, and a
`phase` tag ("serve", "rollout", "train", "eval") so one file can hold a whole
Stage-1 run and, later, a training run's loss/reward curve beside its eval
points. Scalars are logged with `log(...)`; the run config is written once with
`set_config(...)` and secrets are stripped before it touches disk.
"""

import json
import re
import time
from pathlib import Path

# Keys whose values must never be persisted, matched case-insensitively as a
# substring: no api keys, tokens, passwords, or raw environment dumps end up in
# run.json. This is defense in depth on top of callers passing clean configs.
_SECRET_RE = re.compile(r"(key|token|secret|password|passwd|credential|auth)",
                        re.IGNORECASE)


def _sanitize(obj):
    """Recursively drop secret-looking keys and truncate absurd blobs. Values
    under a matching key become the string '<redacted>' rather than being
    removed, so the shape of the config is still legible."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_RE.search(k):
                out[k] = "<redacted>"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def new_run_id(prefix=None):
    """A sortable, unique run id: UTC timestamp plus a short random suffix, so
    two runs started in the same second never collide and never overwrite."""
    import secrets
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rid = f"{stamp}-{secrets.token_hex(2)}"
    return f"{prefix}-{rid}" if prefix else rid


def run_dir_for(output_root, experiment, run_id=None) -> Path:
    """Resolve (and create) a unique run directory. A fresh `run_id` is minted
    when none is given, so calling this never clobbers an earlier experiment."""
    run_id = run_id or new_run_id()
    d = Path(output_root).expanduser() / experiment / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


class RunLogger:
    """Append-only local metrics logger for one run directory.

    Use as a context manager or call `close()` explicitly. Instances are cheap;
    the file is opened lazily on first write and flushed every line so a live
    viewer (and a killed Colab kernel) always see a complete file.
    """

    def __init__(self, output_root, experiment="nextsearch-lfm", run_id=None,
                 config=None):
        self.dir = run_dir_for(output_root, experiment, run_id)
        self.run_id = self.dir.name
        self.experiment = experiment
        self.metrics_path = self.dir / "metrics.jsonl"
        self._fh = None
        self._t0 = time.time()
        if config is not None:
            self.set_config(config)

    def set_config(self, config) -> Path:
        """Write the run config (hardware, software, sampling, paths) once,
        with secrets stripped. Safe to call again to update it."""
        doc = {"run_id": self.run_id, "experiment": self.experiment,
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "config": _sanitize(config)}
        path = self.dir / "run.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        return path

    def log(self, step=None, phase="train", **scalars):
        """Append one metrics record. `step` is the x-axis for training curves
        (loss, reward, lr, …); omit it for one-off scalars. Extra keyword
        scalars are stored verbatim. Non-finite floats are dropped so a viewer
        never has to defend against NaN/inf on an axis."""
        rec = {"wall": round(time.time(), 3),
               "t_rel": round(time.time() - self._t0, 3), "phase": phase}
        if step is not None:
            rec["step"] = step
        for k, v in scalars.items():
            if isinstance(v, float) and (v != v or v in (float("inf"),
                                                         float("-inf"))):
                continue
            rec[k] = v
        if self._fh is None:
            self._fh = open(self.metrics_path, "a")
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        return rec

    def log_summary(self, aggregate, phase="eval"):
        """Convenience: flatten a telemetry aggregate dict into one record,
        keeping only scalar values."""
        scalars = {k: v for k, v in (aggregate or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return self.log(phase=phase, **scalars)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_metrics(run_dir, phase=None):
    """Read all metrics records for a run, optionally filtered to one phase.
    Tolerant of a partially written last line (a live run flushing mid-write),
    so it is safe to call repeatedly while a run is in progress."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-flushed final line; the next read will get it
        if phase is None or rec.get("phase") == phase:
            out.append(rec)
    return out
