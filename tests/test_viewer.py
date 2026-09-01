"""Tests for the live viewer's training-curve extraction and rendering.
Headless (Agg backend), no display needed."""

import matplotlib
matplotlib.use("Agg")

from nextsearch.experiment.runlog import RunLogger
from nextsearch.experiment.viewer import render_curves, training_series


def _run_with_curves(tmp_path):
    with RunLogger(tmp_path, experiment="exp", run_id="r") as log:
        for step in range(0, 5):
            log.log(step=step, phase="train", loss=1.0 / (step + 1), reward=0.1 * step)
        log.log(step=4, phase="eval", accuracy=0.5)
    return tmp_path / "exp" / "r"


def test_training_series_groups_by_phase_and_field(tmp_path):
    series = training_series(_run_with_curves(tmp_path))
    assert set(series) == {"train/loss", "train/reward", "eval/accuracy"}
    # Sorted by step, values preserved.
    steps = [s for s, _ in series["train/loss"]]
    assert steps == sorted(steps) == [0, 1, 2, 3, 4]
    assert series["train/reward"][-1] == (4, 0.4)


def test_render_curves_returns_a_figure_with_one_axis_per_series(tmp_path):
    fig = render_curves(_run_with_curves(tmp_path))
    # 3 series -> 2 cols x 2 rows grid = 4 axes (one blanked).
    assert len(fig.axes) >= 3
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert "train/loss" in titles and "eval/accuracy" in titles


def test_render_curves_placeholder_when_no_curves(tmp_path):
    with RunLogger(tmp_path, experiment="exp", run_id="empty") as log:
        log.log(phase="eval", success_rate=0.9)   # no `step` -> not a curve
    fig = render_curves(tmp_path / "exp" / "empty")
    txt = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "no training curves logged yet" in txt
