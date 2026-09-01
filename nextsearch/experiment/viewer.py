"""A live, in-Colab experiment viewer — no external tracker.

Everything renders inline in a notebook cell from local run artifacts: the
`metrics.jsonl` a `RunLogger` writes and the `rollouts.jsonl` the eval pipeline
appends. `live_view(...)` polls those files on a timer and redraws in place, so
you watch rollouts stream in and (once SFT/OPD/RL stages log them) training
curves grow, all without leaving Colab. `render(...)` draws the same dashboard
once, for a static snapshot or a headless environment.

The dashboard has two halves, each shown only when it has data:

  * Experiment telemetry — derived live from rollouts via `telemetry`:
    cumulative episodes, a wall-latency histogram, output-TPS, and a scalar
    summary (success / tool-call / truncation rates, latency percentiles,
    throughput, cost).
  * Training curves — drawn straight from `metrics.jsonl` step records: any
    numeric series logged under a `step` (loss, reward, lr, eval accuracy, …),
    one line per series.

matplotlib and IPython are imported lazily and only here, so they stay in the
optional `experiment` extra and never burden ordinary harness use. In Colab
both are preinstalled.
"""

import glob
import time
from pathlib import Path

from .runlog import read_metrics
from .telemetry import aggregate, rollout_telemetry

# Metric fields that are bookkeeping, not training curves, so the auto-plotter
# never turns them into a line series.
_NON_SERIES = {"wall", "t_rel", "step", "phase"}


def _find_rollout_files(run_dir):
    """rollouts.jsonl files associated with a run: both a top-level rollouts/
    tree and the eval-style <bench>/<model>/rollouts.jsonl layout."""
    run_dir = Path(run_dir)
    files = set()
    for pat in ("rollouts.jsonl", "**/rollouts.jsonl"):
        files.update(Path(p) for p in glob.glob(str(run_dir / pat),
                                                 recursive=True))
    return sorted(files)


def training_series(run_dir):
    """Training curves from `metrics.jsonl`: every numeric field logged with a
    `step`, grouped into one sorted series per `phase/field` (e.g.
    `train/loss`, `train/reward`, `eval/accuracy`). Empty until an SFT/OPD/RL
    stage logs step records — which is exactly Stage 1 before any training."""
    series = {}
    for rec in read_metrics(run_dir):
        if "step" not in rec:
            continue
        phase = rec.get("phase", "train")
        for k, v in rec.items():
            if k in _NON_SERIES or not isinstance(v, (int, float)) \
                    or isinstance(v, bool):
                continue
            series.setdefault(f"{phase}/{k}", []).append((rec["step"], v))
    for key in series:
        series[key].sort(key=lambda sv: sv[0])
    return series


def collect(run_dir, *, gpu=None, vllm_version=None, gpu_hourly_usd=None):
    """Gather everything the dashboard needs from a run directory, without
    drawing: per-episode telemetry, the aggregate, and training-curve series.
    Safe to call repeatedly on a live run."""
    episodes = []
    for f in _find_rollout_files(run_dir):
        try:
            episodes.extend(rollout_telemetry(
                f, gpu=gpu, vllm_version=vllm_version,
                gpu_hourly_usd=gpu_hourly_usd))
        except (FileNotFoundError, ValueError):
            continue
    summary = aggregate(episodes, gpu_hourly_usd=gpu_hourly_usd) if episodes \
        else {"n_episodes": 0}
    return {"episodes": episodes, "aggregate": summary,
            "series": training_series(run_dir)}


def _summary_lines(agg):
    """The scalar block, formatted for a text panel. Only keys that are
    present and non-None are shown, so an early live snapshot stays tidy."""
    def fmt(v, pct=False):
        if v is None:
            return "—"
        if pct and isinstance(v, (int, float)):
            return f"{v * 100:.1f}%"
        return f"{v:g}" if isinstance(v, float) else str(v)

    rows = [
        ("episodes", fmt(agg.get("n_episodes"))),
        ("success rate", fmt(agg.get("success_rate"), pct=True)),
        ("tool-call rate", fmt(agg.get("parsed_tool_call_rate"), pct=True)),
        ("final-answer rate", fmt(agg.get("final_answer_rate"), pct=True)),
        ("truncation rate", fmt(agg.get("truncation_rate"), pct=True)),
        ("error rate", fmt(agg.get("error_rate"), pct=True)),
        ("mean turns", fmt(agg.get("mean_turns"))),
        ("mean search / fetch", f"{fmt(agg.get('mean_search_calls'))} / "
                                f"{fmt(agg.get('mean_fetch_calls'))}"),
        ("latency p50/p90/p99", " / ".join(
            fmt(agg.get(k)) for k in ("latency_p50_s", "latency_p90_s",
                                      "latency_p99_s")) + " s"),
        ("aggregate TPS", fmt(agg.get("aggregate_tps"))),
        ("episodes/hour", fmt(agg.get("episodes_per_hour"))),
        ("search $", fmt(agg.get("total_search_usd"))),
        ("est GPU $/episode", fmt(agg.get("est_gpu_usd_per_episode"))),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def render(run_dir, *, gpu=None, vllm_version=None, gpu_hourly_usd=None,
           title=None, data=None):
    """Draw the dashboard once and return the matplotlib Figure.

    Pass pre-`collect`ed `data` to avoid re-reading (the live loop does this).
    Panels with no data are annotated rather than left blank, so a snapshot
    taken seconds into a run is still readable.
    """
    import matplotlib.pyplot as plt

    data = data or collect(run_dir, gpu=gpu, vllm_version=vllm_version,
                           gpu_hourly_usd=gpu_hourly_usd)
    episodes, agg, series = data["episodes"], data["aggregate"], data["series"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title or f"NextSearch experiment — {Path(run_dir).name}",
                 fontsize=13, fontweight="bold")

    # (0,0) scalar summary as a text panel.
    ax = axes[0][0]
    ax.axis("off")
    ax.set_title("summary", loc="left", fontsize=11, fontweight="bold")
    ax.text(0.0, 1.0, _summary_lines(agg), va="top", ha="left",
            family="monospace", fontsize=9, transform=ax.transAxes)

    # (0,1) cumulative episodes over the run's wall clock — the live pulse.
    ax = axes[0][1]
    ax.set_title("episodes completed", loc="left", fontsize=11,
                 fontweight="bold")
    ended = sorted(e["ended_at"] for e in episodes if e["ended_at"])
    if ended:
        t0 = ended[0]
        ax.plot([t - t0 for t in ended], range(1, len(ended) + 1),
                marker=".", linewidth=1.5)
        ax.set_xlabel("seconds since first completion")
        ax.set_ylabel("episodes")
    else:
        ax.text(0.5, 0.5, "waiting for episodes…", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    # (1,0) wall-latency distribution.
    ax = axes[1][0]
    ax.set_title("episode wall latency (s)", loc="left", fontsize=11,
                 fontweight="bold")
    walls = [e["wall_s"] for e in episodes if e["wall_s"] is not None]
    if walls:
        ax.hist(walls, bins=min(20, max(5, len(walls))), color="#4C78A8",
                edgecolor="white")
        ax.set_xlabel("seconds")
        ax.set_ylabel("episodes")
    else:
        ax.text(0.5, 0.5, "no latency yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    # (1,1) training curves when present, else output-TPS distribution so the
    # panel is useful in Stage 1 before any training exists.
    ax = axes[1][1]
    if series:
        ax.set_title("training curves", loc="left", fontsize=11,
                     fontweight="bold")
        for name, pts in sorted(series.items()):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=".", linewidth=1.3, label=name)
        ax.set_xlabel("step")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.set_title("output tokens/sec (per episode)", loc="left",
                     fontsize=11, fontweight="bold")
        tps = [e["output_tps"] for e in episodes if e["output_tps"]]
        if tps:
            ax.hist(tps, bins=min(20, max(5, len(tps))), color="#59A14F",
                    edgecolor="white")
            ax.set_xlabel("tokens/sec")
            ax.set_ylabel("episodes")
        else:
            ax.text(0.5, 0.5, "no training curves logged yet\n(SFT/OPD/RL "
                    "stages will populate this)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def live_view(run_dir, *, refresh_s=5, max_seconds=None, gpu=None,
              vllm_version=None, gpu_hourly_usd=None, title=None,
              stop_when_idle_s=None):
    """Redraw the dashboard in place every `refresh_s` seconds.

    Runs until interrupted (the Colab stop button / KeyboardInterrupt),
    `max_seconds` elapses, or — when `stop_when_idle_s` is set — no new episode
    has completed for that many seconds (a natural end-of-run signal). Draws a
    final frame on exit so the cell is left showing the finished state.
    """
    from IPython.display import clear_output, display

    start = time.monotonic()
    last_n, last_change = -1, time.monotonic()
    try:
        while True:
            data = collect(run_dir, gpu=gpu, vllm_version=vllm_version,
                           gpu_hourly_usd=gpu_hourly_usd)
            n = data["aggregate"].get("n_episodes", 0)
            if n != last_n:
                last_n, last_change = n, time.monotonic()
            fig = render(run_dir, title=title, data=data)
            clear_output(wait=True)
            display(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)

            now = time.monotonic()
            if max_seconds is not None and now - start >= max_seconds:
                break
            # Stop once the run is steady for stop_when_idle_s — including when
            # nothing has appeared at all, so a "Run all" can't hang forever on
            # an empty run. Set stop_when_idle_s=None to wait indefinitely.
            if stop_when_idle_s is not None \
                    and now - last_change >= stop_when_idle_s:
                break
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        pass
    # One last frame so the finished dashboard remains on screen.
    fig = render(run_dir, title=title, gpu=gpu, vllm_version=vllm_version,
                 gpu_hourly_usd=gpu_hourly_usd)
    from IPython.display import clear_output, display
    clear_output(wait=True)
    display(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return run_dir


def render_curves(run_dir, *, title=None, series=None, ncols=2):
    """Draw the training curves for a run: one subplot per logged series
    (`train/loss`, `train/reward`, `eval/accuracy`, …), each value vs step.

    This is the focused view behind the training-curves notebook. It reads the
    same `metrics.jsonl` any stage writes via `RunLogger`, so it works for real
    SFT/OPD/RL runs and shows a clear placeholder when nothing is logged yet.
    """
    import matplotlib.pyplot as plt

    series = training_series(run_dir) if series is None else series
    if not series:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, "no training curves logged yet\n\n"
                "log them with nextsearch.experiment.RunLogger:\n"
                "  log.log(step=i, phase='train', loss=…, reward=…)",
                ha="center", va="center", transform=ax.transAxes, color="gray",
                family="monospace", fontsize=10)
        fig.suptitle(title or f"training curves — {Path(run_dir).name}",
                     fontweight="bold")
        return fig

    names = sorted(series)
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.4 * nrows),
                             squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        xs, ys = zip(*series[name])
        ax.plot(xs, ys, marker=".", linewidth=1.5)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        # Latest value in the corner — the number you actually watch live.
        ax.annotate(f"{ys[-1]:g}", xy=(1.0, 1.0), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, color="#333")
    for j in range(len(names), nrows * ncols):   # blank any unused cells
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title or f"training curves — {Path(run_dir).name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def live_curves(run_dir, *, refresh_s=5, max_seconds=None, title=None,
                stop_when_idle_s=None):
    """Redraw the training curves in place every `refresh_s` seconds — the live
    view for the training-curves notebook. Ends on interrupt, after
    `max_seconds`, or when no new step has been logged for `stop_when_idle_s`.
    """
    from IPython.display import clear_output, display
    import matplotlib.pyplot as plt

    start = time.monotonic()
    last_steps, last_change = -1, time.monotonic()
    try:
        while True:
            series = training_series(run_dir)
            n_steps = sum(len(v) for v in series.values())
            if n_steps != last_steps:
                last_steps, last_change = n_steps, time.monotonic()
            fig = render_curves(run_dir, title=title, series=series)
            clear_output(wait=True)
            display(fig)
            plt.close(fig)
            now = time.monotonic()
            if max_seconds is not None and now - start >= max_seconds:
                break
            # Stop when steady for stop_when_idle_s, empty included, so a
            # "Run all" cannot hang forever before any curve is logged.
            if stop_when_idle_s is not None \
                    and now - last_change >= stop_when_idle_s:
                break
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        pass
    fig = render_curves(run_dir, title=title)
    clear_output(wait=True)
    display(fig)
    plt.close(fig)
    return run_dir
