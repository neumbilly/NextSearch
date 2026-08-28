"""The experiment layer: telemetry, run logging, and a live Colab viewer.

This layer sits BESIDE the harness and evaluation, never inside them. The
harness produces rollout JSONL; this package turns those artifacts into
per-episode and aggregate telemetry, records scalar/step metrics for training
curves in a plain `metrics.jsonl`, and renders both live inside a Colab
notebook. There is no external experiment service — every artifact is a local
file under the run directory, so the same data drives the live view, the CLI,
and whatever SFT/OPD/RL stages read next.

Nothing here is imported by the core NextSearch package, and its extra
dependencies (matplotlib, IPython) live in the optional `experiment` extra, so
ordinary harness and evaluation use never needs them.
"""

from .runlog import RunLogger, read_metrics, run_dir_for
from .telemetry import (aggregate, episode_metrics, load_grades,
                        rollout_telemetry)

__all__ = [
    "RunLogger", "read_metrics", "run_dir_for",
    "episode_metrics", "aggregate", "rollout_telemetry", "load_grades",
]
