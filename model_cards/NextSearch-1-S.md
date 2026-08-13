---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
library_name: transformers
tags:
- agent
- web-research
- agentic-search
- tool-use
---

# NextSearch-1-S

NextSearch-1-S is the mid-size **NextSearch-1** web research agent: a
post-trained model that decomposes a question, searches and fetches from the
live web, reconciles conflicting evidence, and returns a concise answer or a
structured research artifact. The family is built to work as the research
component inside a larger system — called repeatedly by an orchestrator —
where per-call accuracy, tail latency, and cost compound. S is the family's
serving sweet spot: MoE inference economics with near-M accuracy on breadth
benchmarks.

| Model | Base | Params | |
|---|---|---|---|
| NextSearch-1-M | Inkling-Small | 276B-A12B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-M) |
| **NextSearch-1-S** (this repo) | Qwen3.6-35B-A3B | 35B-A3B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-S) |
| NextSearch-1-XS | Qwen3.5-9B | 9B dense | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-XS) |

Technical report: **[nexttoken.co/research/nextsearch-1](https://nexttoken.co/research/nextsearch-1)**.
Harness, evaluation suite, and audited benchmark golds:
**[github.com/NextTokenAI/nextsearch](https://github.com/NextTokenAI/nextsearch)**.

## Results

Live-web evaluation (August 2026), against open models of its class.
Benchmarks: SEAL-0 (fresh/conflicting evidence, n=97), FRAMES
(multi-constraint retrieval, n=100), DeepSearchQA (comprehensive answer
sets, n=100), WideSearch-sub (structured table sub-tasks, n=49), and
WideSearch (full tasks under the orchestrated harness, n=20). Best per
column in **bold**.

| | SEAL-0 | FRAMES | DeepSearchQA | WideSearch-sub | WideSearch | mean $/ep | mean turns |
|---|---|---|---|---|---|---|---|
| **NextSearch-1-S** (avg@2) | 0.381 | **0.830** | 0.620 | **0.737** | **0.730** | $0.072 | 7.0 |
| inkling-med (API) | **0.433** | 0.810 | **0.689** | 0.683 | 0.709 | $0.052 | 8.2 |
| qwen3.6-35b-a3b (base) | 0.402 | 0.790 | 0.638 | 0.732 | 0.619 | $0.025 | 9.2 |
| nemotron-3-super (120B-A12B) | 0.320 | 0.740 | 0.538 | 0.462 | — | $0.028 | 10.9 |
| gemma-4-31b | 0.155 | 0.670 | 0.565 | 0.667 | — | $0.013 | 5.0 |
| gpt-oss-120b | 0.227 | 0.670 | 0.450 | 0.406 | — | $0.018 | 9.3 |
| tongyi-dr-30b (30B-A3B) † | 0.351 | 0.780 | 0.407 | 0.322 | 0.388 | $0.011\* | 20.4 |
| quest-35b-rl (35B-A3B) † | 0.371 | 0.740 | 0.467 | 0.157 | 0.382 | $0.023\* | 24.0 |

The table above is the conservative arm (parallel search backend). **With
the recommended exa-auto backend, S's four-bench mean rises ~+10pp to
0.738** (all four benches up) at ~1.3× episode cost; see the serving
notes.

† published deep-research baselines, self-hosted under our harness; their
rows ran under a *more generous* turn budget than the rest of the table
(upper bounds). \* self-hosted: $/ep excludes GPU time. All rows run under
**our harness** (same tools, prompts, turn budgets, pinned task date)
against **audited golds** with one shared judge — consistent within this
table, not comparable to other papers' leaderboards. Protocol and
reproduction:
[docs/evals.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/evals.md);
full analysis in the
[technical report](https://nexttoken.co/research/nextsearch-1).

## Quick start

```bash
vllm serve NextTokenAI/NextSearch-1-S \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 65536
```

Recommended sampling: temperature 0.7, max 16k tokens per turn, thinking
on. The model expects a task date in its system prompt and two tools
(`search`, `fetch`); the exact prompts and tool schemas it was tuned for
ship in the [harness](https://github.com/NextTokenAI/nextsearch):

```bash
pip install nextsearch && nextsearch-eval run --benches seal0 --models nextsearch-1-s --n 10
```

Or plain `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "NextTokenAI/NextSearch-1-S", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("NextTokenAI/NextSearch-1-S")
```

Serving pitfalls that fail silently (tool-call parsing, context caps,
thinking retention):
[docs/serving.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/serving.md).

## License

Released under the **Apache License 2.0**, as is the base model
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

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
