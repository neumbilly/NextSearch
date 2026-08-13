# Third-party data and code

This repository is licensed under Apache 2.0. It builds on work by others,
credited here.

## Benchmarks

No upstream benchmark data is redistributed in this repository. Each benchmark
is pulled from its own home at a pinned revision by `nextsearch-eval prepare`,
and is governed by its own license and terms of use. What we ship is our
audited gold *revisions* — id-keyed verdicts describing corrections we made —
under Apache 2.0.

| Benchmark | Source | Used for |
|---|---|---|
| SEAL-0 | [`vtllms/sealqa`](https://huggingface.co/datasets/vtllms/sealqa) ([paper](https://arxiv.org/abs/2506.01062)) | conflicting and fresh evidence |
| FRAMES | [`google/frames-benchmark`](https://huggingface.co/datasets/google/frames-benchmark) ([paper](https://arxiv.org/abs/2409.12941)) | multi-constraint retrieval |
| DeepSearchQA | [`google/deepsearchqa`](https://huggingface.co/datasets/google/deepsearchqa) ([paper](https://arxiv.org/abs/2601.20975)) | comprehensive set answers |
| WideSearch | [`ByteDance-Seed/WideSearch`](https://huggingface.co/datasets/ByteDance-Seed/WideSearch) ([paper](https://arxiv.org/abs/2508.07999)) | wide table enrichment |

SEAL-0 ships a canary string to detect training contamination. Prepared row
files are gitignored here for that reason, and should not be committed or
published.

`widesearch-sub` is our own derivative: research subtasks harvested from
orchestrated WideSearch runs, with gold tables we authored against primary
sources. The tasks derive from WideSearch prompts; the golds are ours.

## Graders

Our graders reimplement published evaluation protocols so that scores mean
what the original authors intended:

- The single-answer judge prompt adapts the SEAL-0 auto-rater, itself derived
  from the SimpleQA grader template (Wei et al., 2024).
- The set-answer grader uses the DeepSearchQA official whole-response
  auto-rater prompt.
- The table grader reimplements the WideSearch official evaluator — its
  alignment and column-scoring prompts are used verbatim, and the scoring
  pipeline reproduces its semantics. Two deliberate, verdict-preserving
  deviations and one deliberate fix are documented in
  [`nextsearch/grading.py`](nextsearch/grading.py).

## Search and extraction providers

The tools call commercial APIs, which are subject to their own terms and
pricing and require your own API keys:
[Parallel](https://docs.parallel.ai/) (search and extract, the default),
[Exa](https://exa.ai/docs), and [Tavily](https://docs.tavily.com/).

## Base models

The released weights are fine-tunes of Apache 2.0 base models, redistributable
with attribution:

| Release | Base model |
|---|---|
| NextSearch-1-M | [`thinkingmachines/Inkling-Small`](https://huggingface.co/thinkingmachines/Inkling-Small) |
| NextSearch-1-S | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| NextSearch-1-XS | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) |

## Training data sources

The released task pools derive from public research datasets. Full per-source
derivation tables, with licenses, are on the dataset cards; see
[`data/README.md`](data/README.md).

## Related work

The harness and evaluation design draw on the open research-agent literature,
in particular [Search-R1](https://arxiv.org/abs/2503.09516),
[s3](https://arxiv.org/abs/2505.14146),
[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701), and
[QUEST](https://arxiv.org/abs/2605.24218).
