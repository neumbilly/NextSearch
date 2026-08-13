# Training data

The NextSearch-1 training data is released on the Hugging Face Hub rather than
in this repository, so it comes with a dataset viewer and pinned revisions.

## Release 1 — tasks

**[`NextTokenAI/NextSearch-1-Tasks`](https://huggingface.co/datasets/NextTokenAI/NextSearch-1-Tasks)**

The task pools behind both training stages: prompt, gold answer, grading
spec, difficulty band, and source attribution. Configs are split by
license class so every row carries clear terms:

| Config | Contents | License |
|---|---|---|
| `sft-tasks`, `sft-tasks-sharealike`, `sft-tasks-nc` | the tasks behind the supervised corpora | Apache-2.0 / CC BY-SA 4.0 / CC BY-NC 4.0 |
| `rl-tasks`, `rl-tasks-sharealike`, `rl-tasks-nc` | the verified prompt and gold pools used for reinforcement learning | Apache-2.0 / CC BY-SA 4.0 / CC BY-NC 4.0 |

```python
from datasets import load_dataset

tasks = load_dataset("NextTokenAI/NextSearch-1-Tasks", "rl-tasks")
```

Every row is byte-compatible with the `Row` shape this package evaluates
(`nextsearch/types.py`), so a pool can be run through the harness directly —
which is how difficulty was assigned in the first place.

## Release 2 — trajectories

**[`NextTokenAI/NextSearch-1-Trajectories`](https://huggingface.co/datasets/NextTokenAI/NextSearch-1-Trajectories)**

The full supervised trajectories: complete episodes with reasoning, tool
calls, tool results, and final answers — directly trainable, in the same
license-split configs as the tasks, joined by `id`.

## How difficulty was assigned

A task is only useful for training *relative to the policy being trained*. One
that is hard for a small student may be routine for a stronger model, and one
that defeats every available model often has a broken gold rather than a hard
question.

So difficulty here is not an intrinsic label. Each prompt was run through the
full search-and-fetch harness by the intended student, then by progressively
stronger models, and the band records which tier first solved it. Tasks that
no model solved were kept only where the gold was independently trusted.

Two consequences worth knowing before using these bands:

- They are **student-relative**. A band computed against our students is a
  reasonable prior for a similar-capability model and close to meaningless for
  a much stronger one. Re-deriving them against your own model is cheap: run
  the pool through the harness as a benchmark.
- Agreement against the recorded answer was treated as a **data-quality
  signal**, not just a difficulty signal. Where several capable models
  converged on an answer that differed from the gold, the task was re-verified
  or excluded rather than labeled hard.

## Provenance and licensing

These pools are derived from public research datasets — filtered,
deduplicated, decontaminated against the evaluation sets, and in some cases
re-verified. They are not new questions.

Each dataset card carries the license split and a source table naming every
upstream pool it derives from. Attribution
for the benchmark data used by this repository is in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

If you use this data, cite the upstream sources as well as this release. They
did the harder half of the work.
