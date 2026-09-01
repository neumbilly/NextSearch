# LFM2.5-2.6B — Stage 1: serving and harness compatibility

Stage 1 of an LFM2.5-2.6B web-research experiment. The goal is narrow and
deliberate: **serve the model, prove it works inside the existing NextSearch
harness, and establish the artifact and telemetry formats that later stages
(SFT, OPD, RL) will reuse** — without implementing any training yet.

Everything Stage 1 produces is a plain file under one run directory, and the
experiment view is rendered **live inside Colab with no external tracker**.

## Architecture boundary

Responsibilities are kept separate on purpose, so a later training stage swaps
in one layer without touching the others:

| Layer | Owns | Code |
|---|---|---|
| vLLM | rendering, generation, tool-call parsing (`lfm2`), inference metrics | external server |
| NextSearch **harness** | prompts, `search`/`fetch` tools, the episode loop, budgets | `nextsearch/harness.py`, `nextsearch/harnesses.py` |
| NextSearch **evaluation** | prepare → rollout → grade → report | `nextsearch/cli.py`, `nextsearch/benchmarks`, `nextsearch/grading.py` |
| **Experiment** layer | configuration, telemetry, run logging, the live viewer | `nextsearch/experiment/` |
| Colab notebook | thin orchestration only | `notebooks/01_lfm_serving_and_harness.ipynb` |

The model is one registry line (`nextsearch/models/registry.py`), served through
the existing OpenAI-compatible vLLM client. Nothing model-specific leaks into
the harness or the evaluation code.

## The model entry

`lfm2.5-2.6b` in the registry:

- `model_id`: `LiquidAI/LFM2.5-2.6B`, `client`: `vllm`, endpoint from
  `NEXTSEARCH_BASE_URL` (or `--base-url`).
- sampling: `temperature 0.1`, `max_tokens 8192`, and vLLM-only
  `extra_body={top_k: 50, repetition_penalty: 1.1}`.

LFM2.5 is a different model family, so it **does not** inherit NextSearch-1's
`temperature 0.7`; its sampling is written out explicitly.

## Serving with vLLM

LFM2.5's architecture and the `lfm2` tool-call parser ship in **vLLM ≥ 0.23.0**.
The exact command Stage 1 uses (see the notebook's config cell for the knobs):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model LiquidAI/LFM2.5-2.6B \
  --served-model-name LiquidAI/LFM2.5-2.6B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 \
  --port 8000
```

- `--enable-auto-tool-choice --tool-call-parser lfm2` is mandatory: without it,
  the model's Pythonic `<|tool_call_start|>…<|tool_call_end|>` calls arrive as
  plain assistant text and the harness sees **no** tool call — a silent wall of
  zero-scoring episodes. This is the failure the compatibility gate exists to
  catch.
- LFM2.5 *reasoning* variants also need `--reasoning-parser qwen3` so thinking
  is surfaced as `reasoning_content`; the 2.6B dense model does not.
- Keep `--max-model-len` (32768) **above** the harness policy cap (28000): the
  harness stops an episode at its cap while the server still has headroom, which
  is what makes the graceful stop, the budget nudges, and the forced-answer
  salvage work instead of turning overflows into provider errors. See
  [serving.md](serving.md).

### L4 vs A100

- **L4 (24 GB, the default Colab GPU):** ample for a 2.6B model at BF16 with a
  32k window. Start at `--gpu-memory-utilization 0.90`, `--max-num-seqs 32`.
  This is the recommended Stage-1 development GPU — cheap and always available.
- **A100 (40/80 GB):** use it for throughput sweeps and larger
  `--max-num-seqs`, or when moving to a longer context or a bigger checkpoint in
  a later stage. It is not required for Stage 1.

Process management: the notebook launches vLLM with `subprocess.Popen`, keeps
the handle, registers an `atexit` cleanup, streams logs to a file, and polls
`/v1/models` in a bounded readiness loop that prints the server log on failure.
`nohup` alone is deliberately avoided — a detached server that outlives the
kernel and cannot be stopped is worse than none.

## Serving on Modal (pay-as-you-go), driving locally

Because the harness talks to the model only over an OpenAI-compatible endpoint
(`NEXTSEARCH_BASE_URL`), serving and driving are decoupled: the GPU runs only
`vllm serve`, and everything else (compat gate, rollouts, telemetry, viewer) can
run on a laptop or a CPU-only notebook. `deploy/modal_lfm_server.py` serves
LFM2.5-2.6B on a Modal GPU that **scales to zero when idle**, so you pay only
while it serves — and you never reinstall an environment per session.

```bash
pip install "nextsearch[modal]"        # or: pip install modal
modal setup                            # authenticate once
# Create a secret (Modal's HF template defaults to the name `huggingface-secret`);
# one secret can hold HF_TOKEN plus any other keys.
modal deploy deploy/modal_lfm_server.py
# point at a different secret, or none: MODAL_SECRETS=my-secret modal deploy …
# override model/GPU: MODEL=LiquidAI/LFM2.5-8B-A1B GPU=H100 modal deploy …
```

Modal prints a URL; point NextSearch at it (note the trailing `/v1`) and run
everything remotely:

```bash
export NEXTSEARCH_BASE_URL="https://<workspace>--nextsearch-lfm-vllm-serve.modal.run/v1"
nextsearch-compat --model lfm2.5-2.6b
nextsearch-eval rollout --benches seal0:20 --models lfm2.5-2.6b
```

The endpoint sets `--served-model-name LiquidAI/LFM2.5-2.6B`, so the
`lfm2.5-2.6b` registry entry works unchanged. For auth, attach a `vllm-api-key`
secret (the server passes it to vLLM as `--api-key`) and set the same
`VLLM_API_KEY` locally — the vllm client already reads it. In the Colab
notebook, paste the Modal URL into `CFG["remote_base_url"]` and the local vLLM
cells become no-ops, so `01_lfm_serving_and_harness.ipynb` runs on a CPU
runtime.

## Run it in a Modal Notebook (no Colab)

To drop Colab entirely, run the **driver** in a cheap CPU Modal Notebook and let
the **scale-to-zero GPU app** serve the model. The notebook needs no GPU and no
vLLM.

One time:

1. Deploy the GPU endpoint (scale-to-zero): `modal deploy deploy/modal_lfm_server.py`.
2. (Optional, for instant kernel starts) deploy the driver image:
   `modal deploy deploy/modal_notebook_image.py`.
3. Put `PARALLEL_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN` in a Modal Secret
   (one secret can hold all of them — e.g. Modal's `huggingface-secret`). Add
   `VLLM_API_KEY` too if you enabled endpoint auth. Attached secrets are
   injected as **environment variables** — the secret's name doesn't matter and
   there's no `userdata.get`.
4. Create a Modal Volume (e.g. `nextsearch-runs`); it mounts at
   `/mnt/nextsearch-runs` and persists across kernel restarts.

In the Notebook: keep the kernel on **CPU** (`GPU None`), attach the secrets and
the volume, and optionally select the `nextsearch-notebook` image in the
sidebar. Then:

```python
# Cell 1 — clone + install (instant if you selected the prebuilt image; use
# `pip install -e ".[experiment]"` without --no-deps on the Default image)
import os
if not os.path.isdir("NextSearch"):
    !git clone https://github.com/neumbilly/NextSearch.git
%cd NextSearch
!git fetch --all -q && git checkout cursor/lfm2.5-2.6b-stage1-3be6 -q
!pip install -q -e ".[experiment]"    # add --no-deps with the prebuilt image

# Cell 2 — wire the endpoint + persistent run root (secrets are already env vars)
import os
os.environ["NEXTSEARCH_BASE_URL"] = "https://<workspace>--nextsearch-lfm-vllm-serve.modal.run/v1"
os.environ["NEXTSEARCH_HOME"] = "/mnt/nextsearch-runs"   # datasets/ + runs/ persist here

# Cell 3 — compatibility gate against the served model (async: use await)
import json
from nextsearch.compat import probe
from nextsearch.models import get_client, get_model
m = get_model("lfm2.5-2.6b")
print(json.dumps(await probe(get_client(m), m.model_id, m.sampling), indent=2))

# Cell 4 — prepare + a 20-task dev rollout, persisted to the volume
from nextsearch import benchmarks
benchmarks.get("seal0").prepare()
!nextsearch-eval rollout --benches seal0:20 --models lfm2.5-2.6b --date 2026-07-31

# Cell 5 — live telemetry + training curves (inline, no tracker)
%matplotlib inline
from nextsearch.experiment.viewer import render, render_curves
import matplotlib.pyplot as plt
render(os.environ["NEXTSEARCH_HOME"]); plt.show()
render_curves(os.environ["NEXTSEARCH_HOME"]); plt.show()
```

Everything under `/mnt/nextsearch-runs` (manifests, `rollouts.jsonl`,
`metrics.jsonl`, telemetry, curves) survives kernel restarts, and the GPU app
spins down when idle so you only pay while it serves.

### Or: serve vLLM inside the notebook's own GPU kernel

If you'd rather not run a separate app, give the **notebook kernel a GPU**
(L4, RAM ≳ 16 GB, CPU ≳ 4) and serve vLLM on `localhost` in the kernel. That is
`notebooks/03_modal_notebook_gpu.ipynb`: install → start vLLM via `Popen` with a
readiness loop → compat → prepare → smoke + dev rollout → live telemetry/curves
→ stop. It is simpler (one place, no cold-start URL) but the **GPU bills for the
whole time the kernel runs** (the idle timeout stops it), so it is not
scale-to-zero. Attach `huggingface-secret` (keys become env vars) and a Volume
for persistence exactly as above. Choose the deployed app for pay-as-you-go, or
the in-kernel GPU for a single, self-contained notebook.

## The compatibility gate

`nextsearch-compat` (module `nextsearch/compat.py`) is a generic model/server
probe. It uses a **synthetic `lookup` tool** and a fixed prompt, so it needs
**no `PARALLEL_API_KEY` and spends no search credits** — just two calls to the
model server. It:

1. sends a prompt only answerable via the tool;
2. confirms a parsed OpenAI-format tool call came back;
3. verifies the function name and a tool-call id;
4. parses and validates the JSON arguments;
5. returns a synthetic result carrying a sentinel;
6. makes a second call;
7. confirms the sentinel appears in the final answer;
8. reports whether `reasoning_content` was captured, plus per-call latency and
   token usage.

It prints one JSON object and exits non-zero on any required-check failure:

```bash
nextsearch-compat --model lfm2.5-2.6b --base-url http://localhost:8000/v1
```

Run it before spending anything on rollouts. Fake healthy / text-only clients
cover it in `tests/test_compat.py`. The result includes `observed_tool_call`
(name + parsed arguments) so a pass *shows* the tool being invoked, not just a
green check.

## Reasoning: LFM2.5-2.6B is a dense, non-reasoning model

Per Liquid AI's model page, LFM2.5-2.6B is a **2.6B dense** model for agentic
workloads with native tool calling — it has **no separate thinking/scratchpad
channel**. So:

- Do **not** pass `--reasoning-parser` for the 2.6B model. That flag is for the
  reasoning variants (`LFM2.5-1.2B-Thinking`, `LFM2.5-8B-A1B`), which emit a
  `<think>…</think>` block that `qwen3` splits into `reasoning_content`.
- When the 2.6B model "thinks", that text is part of its **answer** — there is
  nothing to parse out, and `reasoning_content` is legitimately empty. This is
  expected behavior, not a tool-parsing bug.
- If you want an explicit scratchpad, either prompt for one (ask for a delimited
  "Reasoning:" section followed by a final answer) or switch to a reasoning
  variant served with `--reasoning-parser qwen3`, in which case the harness
  captures the thinking in `reasoning_content` automatically and the telemetry
  `reasoning_tokens`/`reasoning_chars` fields fill in.

## Inspecting a run: full traces

A stop reason does not prove tools were called correctly — you want the whole
conversation. `nextsearch.experiment.trace`:

- `save_trace(rollout, path)` writes the raw rollout JSON (a complete,
  re-loadable trace); `save_as_rollouts(rollout, path)` appends it to a
  `rollouts.jsonl` so the live viewer and `nextsearch-telemetry` treat a
  one-off smoke episode like any run.
- `print_trace(rollout)` renders the transcript: a header (stop reason, turns,
  wall/model time, cost) and a **tool-call summary**, then every turn with each
  tool call's parsed arguments, each tool result, any reasoning, and the final
  answer.

The Stage-1 smoke cell saves the raw trace and prints it in full, which is how
you confirm — by eye — that `search`/`fetch` were called with sensible
arguments and the model used the results. Note that a self-hosted vLLM model has
no per-token price, so `model_$` is `None` in a trace (not an error); search/
fetch cost is tracked separately and GPU dollars are estimated from wall time in
the telemetry aggregate.

## Ungraded rollout vs paid grading

- A **rollout** runs the agent live against the web (needs `PARALLEL_API_KEY`)
  and writes `rollouts.jsonl`. It costs search credits and GPU time but **no**
  judge spend, so Stage-1 development runs ungraded by default.
- **Grading** is a separate paid step that scores answers with an LLM judge.
  The reported NextSearch-1 numbers use a fixed OpenRouter judge
  (`OPENROUTER_API_KEY`). If you only have a Google AI Studio key, a Gemini
  judge is registered — pass it explicitly:

  ```bash
  nextsearch-eval grade  --eval-id <id> --judge gemini-3.6-flash
  nextsearch-eval report --eval-id <id>
  ```

  The Gemini judge is **not** the judge the headline table used, so scores
  graded with it are self-consistent but not comparable to the reported
  numbers. Reproducing the reported scores requires `OPENROUTER_API_KEY` and the
  default judge.

## Frozen task dates and id files

Every system prompt carries the task date; it is protocol, not decoration, and
is frozen in the run manifest. Stage 1 pins `--date 2026-07-31` so a run that
crosses midnight cannot mutate its own prompt. To fix an exact task subset
across runs, pass `--ids-file <file>` (one sample id per line); its hash is
frozen too. Both make a rollout reproducible and a resume validate-able.

## Persistence and Colab session interruption

Colab runtimes are ephemeral and disconnect. Point the run at a Drive-backed
root and everything survives:

```
/content/drive/MyDrive/nextsearch-lfm/runs/<experiment>/<run_id>/
    run.json         # config + hardware/software, secrets stripped
    metrics.jsonl    # scalar/step stream (training curves + eval summary)
    datasets/        # prepared rows + manifests (NEXTSEARCH_HOME)
    runs/<eval_id>/  # manifests, rollouts.jsonl, grade sidecars, summaries
    telemetry.json / telemetry.csv
```

`RunLogger` mints a **unique** `run_id` (timestamp + random suffix) so no run
ever overwrites an earlier one, and `NEXTSEARCH_HOME` is pointed at the run
directory so the evaluation pipeline writes there too. Rollout files are
append-only and resumable: after a disconnect, re-run the rollout cell and it
continues from the accepted episodes already on disk. Later stages write adapter
checkpoints under the same root.

## Live experiment view and training curves

The experiment layer renders inline in Colab — **no W&B, no external service**.
All data is a live view over local files:

- `nextsearch.experiment.telemetry` derives per-episode and aggregate metrics
  from `rollouts.jsonl` (see the matrix below), as a Python API and the
  `nextsearch-telemetry` CLI (JSON + CSV).
- `nextsearch.experiment.runlog.RunLogger` appends scalars and `step` records to
  `metrics.jsonl` — this is where SFT/OPD/RL stages log loss / reward / lr /
  eval curves.
- `nextsearch.experiment.viewer` reads both and draws a four-panel dashboard;
  `render(run_dir)` for a snapshot, `live_view(run_dir)` to redraw in place
  every few seconds while a rollout streams. The training-curve panel populates
  automatically once any stage logs `step` records; in Stage 1 it shows
  output-TPS instead.
- For a **focused, live training-curves view**, `render_curves(run_dir)` /
  `live_curves(run_dir)` draw one subplot per logged series (`train/loss`,
  `train/reward`, `eval/accuracy`, …) straight from `metrics.jsonl`. The
  `notebooks/02_training_curves.ipynb` notebook is a runnable, auto-refreshing
  wrapper around them (with a synthetic-curve demo cell so the live view works
  before any real training exists). Because it just tails a local file, it can
  watch a run training on Modal, a rented box, or your laptop.

### Telemetry matrix

Per episode: benchmark, task id, checkpoint/model, GPU, vLLM version, sampling,
stop reason, error/truncation flags, turns, assistant messages, reasoning
chars/tokens (when the server reports them), prompt/completion/cumulative
tokens, search and fetch call counts, max/mean calls per tool-turn, parallel-call
turns, unique fetch URLs/domains, model/tool/wall latency, effective output
tokens/sec, and model/search/fetch/judge cost where known.

Aggregate: success rate, parsed-tool-call rate, final-answer rate, mean & p90
turns, mean reasoning/output tokens, mean search/fetch calls, parallel-call
rate, truncation/error rates, p50/p90/p99 wall latency, aggregate and
single-episode TPS, episodes/hour, estimated GPU $/episode, and total search
dollars.

## Performance matrix (fill in per GPU)

Record one row per serving configuration; the numbers come straight from
`nextsearch-telemetry` / the viewer aggregate.

| GPU | max-num-seqs | context | aggregate TPS | p50 / p90 / p99 wall (s) | episodes/hour | $/episode (GPU) | search $/episode |
|---|---|---|---|---|---|---|---|
| L4 | 32 | 32768 | — | — | — | — | — |
| A100-40GB | 64 | 32768 | — | — | — | — | — |

## Stage-1 acceptance criteria

- Existing upstream tests pass, and the new compatibility, telemetry, and model
  tests pass.
- `nextsearch-eval models` lists `lfm2.5-2.6b`.
- `nextsearch-compat` has a useful `--help` and passes against a served LFM2.5.
- Telemetry parsing is covered against synthetic rollouts.
- The notebook is valid and runs top-to-bottom on a fresh Colab GPU runtime.
- No secrets, benchmark datasets, or generated rollouts are committed.

## How later stages reuse Stage 1

SFT, OPD, and RL stages consume the **same** artifacts unchanged:

- the model entry and sampling recipe (`nextsearch.models.registry`);
- the harness prompts, tools, and episode loop (rollouts are the training
  signal);
- `rollouts.jsonl` and the telemetry schema (evaluation of a trained checkpoint
  is the same rollout + telemetry path, with `--checkpoint`/`gpu` stamped in);
- `RunLogger` + the live viewer — a training stage logs `loss`/`reward`/`lr` per
  `step`, and the same Colab dashboard shows the curves live beside the eval
  telemetry.

No Stage-1 format needs to change for a training notebook to plug in.
