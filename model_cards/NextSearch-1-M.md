---
license: apache-2.0
base_model: thinkingmachines/Inkling-Small
pipeline_tag: text-generation
library_name: transformers
tags:
- agent
- web-research
- agentic-search
- tool-use
---

# NextSearch-1-M

NextSearch-1-M is the largest of the three **NextSearch-1** web research
agents: post-trained models that decompose a question, search and fetch from
the live web, reconcile conflicting evidence, and return a concise answer or
a structured research artifact. They are built to work as the research
component inside a larger system — called repeatedly by an orchestrator —
where per-call accuracy, tail latency, and cost compound.

| Model | Base | Params | |
|---|---|---|---|
| **NextSearch-1-M** (this repo) | Inkling-Small | 276B-A12B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-M) |
| NextSearch-1-S | Qwen3.6-35B-A3B | 35B-A3B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-S) |
| NextSearch-1-XS | Qwen3.5-9B | 9B dense | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-XS) |

Technical report: **[nexttoken.co/research/nextsearch-1](https://nexttoken.co/research/nextsearch-1)**.
Harness, evaluation suite, and audited benchmark golds:
**[github.com/NextTokenAI/nextsearch](https://github.com/NextTokenAI/nextsearch)**.

## Results

Live-web evaluation (August 2026), 12B-active M against frontier API
anchors. Benchmarks: SEAL-0 (fresh/conflicting evidence, n=97), FRAMES
(multi-constraint retrieval, n=100), DeepSearchQA (comprehensive answer
sets, n=100), WideSearch-sub (structured table sub-tasks, n=49), and
WideSearch (full tasks under the orchestrated harness, n=20). Best per
column in **bold**.

| | SEAL-0 | FRAMES | DeepSearchQA | WideSearch-sub | WideSearch | mean $/ep | mean turns |
|---|---|---|---|---|---|---|---|
| **NextSearch-1-M** | **0.515** | 0.850 | **0.803** | 0.805 | 0.708 | $0.074 | 5.9 |
| glm-5.2 (355B-A32B) | 0.505 | **0.920** | 0.790 | 0.856 | **0.806** | $0.015 | 7.5 |
| gemini-3.6-flash | 0.495 | 0.880 | 0.773 | **0.885** | 0.712 | $0.127 | 8.8 |
| gpt-5.6-luna-med | 0.484 | 0.820 | 0.788 | 0.763 | 0.742 | $0.010 | 6.9 |
| deepseek-v4-flash | 0.474 | 0.820 | 0.781 | 0.682 | 0.758 | $0.029 | 10.1 |
| nemotron-3-ultra (550B-A55B) | 0.423 | 0.850 | 0.693 | 0.760 | — | $0.083 | 9.3 |

All rows run under **our harness** (same tools, prompts, turn budgets,
pinned task date) against **audited golds** with one shared judge —
consistent within this table, not comparable to other papers'
leaderboards. Protocol, costs, and reproduction:
[docs/evals.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/evals.md);
full analysis in the
[technical report](https://nexttoken.co/research/nextsearch-1).

## Quick start

The weights are ~530 GB bf16 — plan for a multi-GPU node (e.g. 8×H200).
See the [vLLM recipe for Inkling](https://recipes.vllm.ai/thinkingmachines/Inkling)
for current serving flags; our serving notes (sampling, context caps,
tool-call parsing pitfalls) are in
[docs/serving.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/serving.md).

```bash
vllm serve NextTokenAI/NextSearch-1-M --tensor-parallel-size 8 \
  --enable-auto-tool-choice --max-model-len 65536
```

Recommended sampling: temperature 0.7, max 16k tokens per turn, reasoning
effort 0.7. The model expects a task date in its system prompt and two
tools (`search`, `fetch`); the exact prompts and tool schemas it was
tuned for ship in the [harness](https://github.com/NextTokenAI/nextsearch):

```bash
pip install nextsearch && nextsearch-eval run --benches seal0 --models nextsearch-1-m --n 10
```

## License

Released under the **Apache License 2.0**, as is the base model
[`thinkingmachines/Inkling-Small`](https://huggingface.co/thinkingmachines/Inkling-Small).

## Citation

```bibtex
@techreport{nextsearch1,
  title       = {NextSearch-1: Open models for wide and deep web research},
  author      = {Nitish Kulkarni and Alankar Jain},
  institution = {NextToken},
  year        = {2026},
  url         = {https://nexttoken.co/research/nextsearch-1}
}
```
