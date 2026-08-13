# Reproducing the evaluations

The suite is staged: **prepare** pulls the source datasets and writes
canonical rows, **rollout** runs episodes against the live web, **grade**
scores the terminal answers, and **report** aggregates. `run` does all four.

One invocation is one evaluation, identified by one `eval_id`, over a
(benchmarks x models) matrix. Its manifest freezes everything a number depends
on. Comparisons are only valid within a single `eval_id` — which is the point,
because these runs cost real money and take hours, so the failure mode worth
engineering against is not a crash but a run that silently continues under
changed settings.

## Setup

```bash
pip install -e .
```

Two API keys are needed, in the environment or in a `.env` file in the working
directory:

```
PARALLEL_API_KEY=...     # the search and fetch backend
OPENROUTER_API_KEY=...   # the judge, and any hosted model you evaluate
```

`EXA_API_KEY` and `TAVILY_API_KEY` are optional, needed only for
`--search-mode exa-*` or `tavily-*`.

Artifacts land under `./nextsearch-runs` (override with `NEXTSEARCH_HOME`):
prepared datasets, per-episode rollouts, grade sidecars, and the judge cache.

## The reported protocol

```bash
nextsearch-eval run --benches main --models nextsearch-1-xs
```

`main` is the four-benchmark suite behind every headline number:

| Benchmark | Rows | Turn budget | Primary metric | Measures |
|---|---:|---:|---|---|
| `seal0` | 97 | 10 | judge-scored correctness | resolving fresh or conflicting evidence |
| `frames:100` | 100 | 10 | judge-scored correctness | satisfying several retrieval constraints |
| `deepsearchqa:100` | 100 | 20 | set-answer F1 | recovering a comprehensive answer set |
| `widesearch-sub` | 49 | 10 | table row-F1 | filling a structured table as a subagent |

Everything else is held fixed: the `solo` harness, Parallel Turbo search, one
sampled episode per question, and the `gpt-5.6-luna-low` judge. Residual model
or tool errors score zero rather than being dropped — an agent that crashes is
an agent that did not answer.

**Set the task date explicitly to reproduce a published number.** The date
goes into every system prompt and is what "current" and "latest" resolve
against, so it changes the answers:

```bash
nextsearch-eval run --benches main --models nextsearch-1-xs --date 2026-07-31
```

Without `--date`, today's date is used and frozen into the manifest. That is
the right default for evaluating a model now, and the wrong one for matching a
table computed months earlier — the world moved.

## The composite wide-research setting

WideSearch tasks ask for tables with many entities and fields, which a single
research context handles poorly. That row uses the orchestrated harness: an
orchestrator decomposes the task and assembles the final table, while the
model under test runs bounded research calls as its subagent.

```bash
nextsearch-eval run --benches composite \
  --harness orchestrated \
  --models deepseek-v4-flash \
  --subagent nextsearch-1-s \
  --max-wall 2400 --subagent-max-wall 420
```

Note which model is which here: `--models` is the orchestrator, `--subagent`
is the model being evaluated. Results are recorded under the composite name
`<orchestrator>+<subagent>`, so an orchestrated row can never be mistaken for
a solo run of either half.

The orchestrator matters as much as the subagent. It chooses the
decomposition, writes each research brief, controls how many calls are issued,
and reconciles conflicting reports — so a strong subagent can lose most of its
value under an orchestrator that under-delegates. Any claim about a research
model in this setting should name the orchestrator it was measured under.

## Cost and time

Rough per-model figures for the `main` suite at the reported protocol, at
2026 prices. Search is billed per call and the judge per graded episode, so
both scale with the matrix; model cost depends on what you are serving.

| Benchmark | Episodes | Search + judge | Notes |
|---|---:|---:|---|
| `seal0` | 97 | ~$1–2 | |
| `frames:100` | 100 | ~$1–2 | |
| `deepsearchqa:100` | 100 | ~$3–6 | 20-turn budget, longer episodes |
| `widesearch-sub` | 49 | ~$2–4 | table grading needs a judge call per column |
| `composite` (20 tasks) | 20 | ~$10–15 | each task fans out into many nested episodes |

Wall time at the default concurrency of 32 is roughly 20–40 minutes for the
solo suite. The composite row is far slower, because a turn blocks on the
slowest subagent in its batch.

Iterate with `--n 20` before paying for a full pass. Judge verdicts are cached
across evaluations keyed on the exact prompt, so re-grading unchanged answers
is free.

## Reading the results

`report` writes `summary.json` and `summary.md` under the eval directory, with
one row per (benchmark, model, grader) cell. Three columns deserve attention:

**The confidence interval, not just the score.** At these sample sizes the 95%
interval is roughly ±10 points, so two models three points apart are tied. If
you are reproducing a published number, treat anything inside that band as a
match and anything outside it as a real difference worth diagnosing.

**`fmt✗ (score)` on the table benchmarks.** A response that will not parse
into the required table scores zero for the whole task, so a model with a
formatting tic can post a low headline while researching perfectly well. That
cell gives the count of unparseable answers and the mean row-F1 over the ones
that did parse. If the count is large, read the conditional score.

**The cost split.** Model, search, subagent, and judge dollars are reported
separately, from write-time snapshots. A `≥` prefix marks a lower bound: some
episode costs were never reported by the provider and count as zero in the
sum, and the count of those travels with it.

## Where the published numbers live

The headline tables — every release against its comparators, with cost and
latency — are in the
[technical report](https://nexttoken.co/research/nextsearch-1), and each
model card on the Hub carries
its own results with the protocol caveats attached. They are not duplicated
here, because a number copied into a README goes stale silently.

What this repository guarantees is that the *protocol* is reproducible: the
same prepared rows (validated by content hash), the same prompts, the same
tool specifications, the same grader, and the same caps.

## Two caveats worth knowing

**Gold revisions.** We audited these benchmarks and corrected their reference
answers; those corrections are applied by default and are described in
[`nextsearch/benchmarks/revisions.py`](../nextsearch/benchmarks/revisions.py).
Scores computed with them are not comparable to scores published against the
unmodified upstream golds. Pass `--no-revisions` to evaluate against the
originals.

**One judge across all benchmarks.** Every reported number uses the same judge
rather than each benchmark's own official auto-rater. That keeps model
comparisons internally consistent, and it means absolute numbers here are not
directly comparable to numbers published under a different grader.

## Other things the harness can measure

The knobs that produced the report's analysis section are all exposed:

```bash
# search-backend sensitivity: the same model, a different retrieval stack
nextsearch-eval run --benches main --models nextsearch-1-s --search-mode exa-auto

# interaction-budget sensitivity
nextsearch-eval run --benches seal0 --models nextsearch-1-m --max-turns 40

# does the policy depend on the literal tool names it was trained on?
nextsearch-eval run --benches seal0 --models nextsearch-1-s \
  --tool-alias search=web_search,fetch=read_url

# how much of the gain came from the task date?
nextsearch-eval run --benches seal0 --models nextsearch-1-m --date off
```

Each is frozen in the manifest, so a run under a different setting is visibly
a different evaluation rather than a silent contamination of the first.
