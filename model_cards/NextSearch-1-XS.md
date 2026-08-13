---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
pipeline_tag: text-generation
library_name: transformers
tags:
- agent
- web-research
- agentic-search
- tool-use
---

# NextSearch-1-XS

NextSearch-1-XS is the smallest **NextSearch-1** web research agent: a
post-trained model that decomposes a question, searches and fetches from the
live web, reconciles conflicting evidence, and returns a concise answer or a
structured research artifact. The family is built to work as the research
component inside a larger system — called repeatedly by an orchestrator —
where per-call accuracy, tail latency, and cost compound. XS is the
price/latency point: a 9B dense model you can serve on one GPU.

| Model | Base | Params | |
|---|---|---|---|
| NextSearch-1-M | Inkling-Small | 276B-A12B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-M) |
| NextSearch-1-S | Qwen3.6-35B-A3B | 35B-A3B MoE | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-S) |
| **NextSearch-1-XS** (this repo) | Qwen3.5-9B | 9B dense | [weights](https://huggingface.co/NextTokenAI/NextSearch-1-XS) |

Technical report: **[nexttoken.co/research/nextsearch-1](https://nexttoken.co/research/nextsearch-1)**.
Harness, evaluation suite, and audited benchmark golds:
**[github.com/NextTokenAI/nextsearch](https://github.com/NextTokenAI/nextsearch)**.

## Results

Live-web evaluation (August 2026), against small and cheap models.
Benchmarks: SEAL-0 (fresh/conflicting evidence, n=97), FRAMES
(multi-constraint retrieval, n=100), DeepSearchQA (comprehensive answer
sets, n=100), WideSearch-sub (structured table sub-tasks, n=49). Best per
column in **bold**.

| | SEAL-0 | FRAMES | DeepSearchQA | WideSearch-sub | mean $/ep | mean turns |
|---|---|---|---|---|---|---|
| **NextSearch-1-XS** | **0.289** | **0.790** | 0.588 | 0.703 | $0.093 | 7.0 |
| claude-haiku-4.5 | 0.258 | 0.720 | **0.670** | **0.864** | $0.154 | 8.7 |
| qwen3.5-9b (base) | 0.206 | 0.700 | 0.599 | 0.610 | $0.030 | 9.7 |
| gemma-4-31b | 0.155 | 0.670 | 0.565 | 0.667 | $0.013 | 5.0 |
| gemini-3.5-flash-lite | 0.216 | 0.570 | 0.535 | 0.640 | $0.020 | 7.3 |
| gpt-oss-20b | 0.206 | 0.710 | 0.406 | 0.470 | $0.012 | 10.5 |

The table above is the conservative arm (parallel search backend). **With
the recommended exa-auto backend XS's four-bench mean rises from 0.592 to
0.693** (0.412 / 0.860 / 0.771 / 0.728) — above every genuinely small
model on every bench except haiku's WideSearch-sub, and edging the 550B
nemotron-3-ultra anchor (0.682) at a fraction of the size.

All rows run under **our harness** (same tools, prompts, turn budgets,
pinned task date) against **audited golds** with one shared judge —
consistent within this table, not comparable to other papers'
leaderboards. Protocol and reproduction:
[docs/evals.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/evals.md);
full analysis in the
[technical report](https://nexttoken.co/research/nextsearch-1).

## Quick start

```bash
vllm serve NextTokenAI/NextSearch-1-XS \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 65536
```

Recommended sampling: temperature 0.7, max 16k tokens per turn, thinking
on. The model expects a task date in its system prompt and two tools
(`search`, `fetch`); the exact prompts and tool schemas it was tuned for
ship in the [harness](https://github.com/NextTokenAI/nextsearch):

```bash
pip install nextsearch && nextsearch-eval run --benches seal0 --models nextsearch-1-xs --n 10
```

Or plain `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "NextTokenAI/NextSearch-1-XS", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("NextTokenAI/NextSearch-1-XS")
```

Serving pitfalls that fail silently (tool-call parsing, context caps,
thinking retention):
[docs/serving.md](https://github.com/NextTokenAI/nextsearch/blob/main/docs/serving.md).

## License

Released under the **Apache License 2.0**, as is the base model
[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B).

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
