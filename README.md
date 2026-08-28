<h1 align="center">NextSearch-1</h1>

<p align="center">
  Open models for wide and deep web research — weights, harness, evaluations, and data.
</p>

<p align="center">
  <a href="https://huggingface.co/NextTokenAI">Models</a> ·
  <a href="https://huggingface.co/datasets/NextTokenAI/NextSearch-1-Tasks">Data</a> ·
  <a href="#the-harness">Harness</a> ·
  <a href="docs/evals.md">Reproducing the numbers</a>
</p>

---

NextSearch-1 is a family of post-trained web research agents. They decompose a
question, search and fetch from the live web, reconcile conflicting evidence,
and return either a concise answer or a structured research artifact.

They are built for a specific role: the **research worker inside a larger
system**, called repeatedly by an orchestrator that plans the work and
assembles the final report. That role has different economics from a chat
model. One user request can fan out into dozens of research calls, so per-call
accuracy, tail latency, and cost all compound.

| Model | Base | Params | |
|---|---|---|---|
| **NextSearch-1-M** | Inkling-Small | 276B total, 12B active | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-M) |
| **NextSearch-1-S** | Qwen3.6-35B-A3B | 35B total, 3B active | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-S) |
| **NextSearch-1-XS** | Qwen3.5-9B | 9B dense | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-XS) |

All three are Apache 2.0, share one tool interface, and occupy different
points on the quality, latency, and serving-cost curve. Headline results, with
comparators and cost, are in the
[technical report](https://nexttoken.co/research/nextsearch-1) and on each
model card.

## Quick start

```bash
pip install -e .
```

Set two keys, in your environment or a `.env` file:

```
PARALLEL_API_KEY=...     # search and fetch
OPENROUTER_API_KEY=...   # the judge, and any hosted model you evaluate
```

Serve a model and evaluate it:

```bash
vllm serve NextTokenAI/NextSearch-1-XS \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 65536
```

```bash
nextsearch-eval run --benches seal0 --models nextsearch-1-xs --n 10
```

Nothing here is specific to our weights — any OpenAI-compatible endpoint works
with `--base-url`, and any OpenRouter model works by passing its id:

```bash
nextsearch-eval run --benches seal0 --models deepseek/deepseek-v4-flash --n 10
```

## Using the agent in your own code

```python
import asyncio
from nextsearch import harnesses
from nextsearch.harness import run_episode
from nextsearch.models import get_client, get_model
from nextsearch.types import Row, system, user

harness = harnesses.get("solo")
model = get_model("nextsearch-1-xs")
row = Row(messages=[system("Answer concisely."),
                    user("Which US states raised their minimum wage in 2026?")],
          id="q-1")

rollout = asyncio.run(run_episode(
    get_client(model), model, row, harness.tools(),
    max_turns=10, system_suffix=harness.system_suffix("2026-07-31")))

print(rollout.messages[-1]["content"])
print(rollout.n_turns, "turns,", rollout.timing["cost_usd"], "USD")
```

## The harness

The harness is the system prompt, the tool schemas, the interaction loop, the
resource limits, and the recovery behavior — everything around the weights.

It is worth treating as a first-class part of the system. Developing this
harness against frozen models, *before any training*, moved a nine-cell
benchmark mean from 0.648 to 0.747 — comparable to a round of post-training,
and invisible in any table that reports only model names.

Two tools: `search` returns ranked excerpts, `fetch` reads specific pages
against an objective. Adding `fetch` improved comprehensive set-answer
accuracy and *lowered* cost at the same time, because reading a selected page
replaces several rounds of re-searching. Every system prompt carries the task
date, without which "current" and "latest" resolve from pretraining priors.
Budgets are told to the model before they bind, and an episode that hits a
limit still gets one no-tools call to write an answer from research already in
its transcript.

For wide table tasks, the `orchestrated` harness gives the top-level model
only a `research` tool, each call running a complete nested episode on a
subagent.

**[docs/harness.md](docs/harness.md)** covers the design and what each choice
was worth.

## Evaluations

Four benchmarks, each stressing one capability of the research role, all run
against the live web rather than a fixed corpus:

| Benchmark | Rows | Measures |
|---|---:|---|
| SEAL-0 | 97 | resolving fresh or conflicting evidence |
| FRAMES | 100 | satisfying several retrieval constraints |
| DeepSearchQA | 100 | recovering a comprehensive answer set |
| WideSearch-sub | 49 | filling a structured table as a subagent |

```bash
nextsearch-eval run --benches main --models nextsearch-1-xs --date 2026-07-31
```

Two things about these numbers. We **audited the reference answers** and
corrected them: models kept returning well-sourced current answers against
stale or mis-extracted golds, which penalizes the right behavior exactly where
model comparisons are decided. Those corrections ship here and are applied by
default. And **all four benchmarks are graded by one shared judge**, which
keeps model comparisons consistent but means absolute numbers are not
comparable to those published under each benchmark's own auto-rater.

**[docs/evals.md](docs/evals.md)** has the full protocol, costs, and the other
axes the harness can measure — search backend, turn budget, tool renaming, and
the task date.

## Serving

**[docs/serving.md](docs/serving.md)** covers per-model sampling, why the
context cap must sit below your real serving window, and the two tool-calling
misconfigurations that fail silently.

## LFM experiment

An experiment track for serving and studying **LiquidAI/LFM2.5-2.6B** as a
web-research agent inside this harness. Stage 1 serves the model with vLLM,
proves its native `lfm2` tool calling works end to end, runs reproducible
rollouts against Parallel Turbo, and views telemetry and training curves **live
in Colab — no W&B, no external tracker**. It is built so later SFT/OPD/RL stages
reuse the same configs, rollout artifacts, and dashboards.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neumbilly/NextSearch/blob/cursor/lfm2.5-2.6b-stage1-3be6/notebooks/01_lfm_serving_and_harness.ipynb)

```bash
pip install -e ".[experiment]"          # adds the live-viewer deps (matplotlib, IPython)
nextsearch-compat --model lfm2.5-2.6b   # tool-calling compatibility gate (no search credits)
nextsearch-telemetry <run-dir>          # per-episode + aggregate telemetry (JSON/CSV)
```

Full protocol, serving command, L4 vs A100 guidance, and Stage-1 acceptance
criteria: **[docs/lfm2-step1.md](docs/lfm2-step1.md)**. The runnable notebook is
[notebooks/01_lfm_serving_and_harness.ipynb](notebooks/01_lfm_serving_and_harness.ipynb).

## Data

Task pools and full trajectories on the Hub:
**[data/README.md](data/README.md)**.

## Layout

```
nextsearch/
  harness.py        the episode loop
  harnesses.py      prompt layer + toolset: `solo` and `orchestrated`
  prompts/          the research, orchestration, and subagent instructions
  tools/            search (Parallel · Exa · Tavily) · fetch · research subagent
  benchmarks/       the four benchmarks, plus our audited gold revisions
  grading.py        single-answer, set-answer, and table graders
  manifest.py       what gets frozen so a run is reproducible
  report.py         aggregation: scores, intervals, cost, latency
docs/               harness design · reproducing evals · serving
```

## Citation

```bibtex
@techreport{nextsearch1,
  title  = {NextSearch-1: Open models for wide and deep web research},
  author = {Nitish Kulkarni and Alankar Jain},
  institution = {NextToken},
  year   = {2026},
  url    = {https://nexttoken.co/research/nextsearch-1}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Upstream benchmarks, base models, and
data sources are credited in [THIRD_PARTY.md](THIRD_PARTY.md); no upstream
benchmark data is redistributed here.
