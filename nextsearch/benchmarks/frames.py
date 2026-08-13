"""FRAMES: multi-constraint retrieval and reasoning.

Questions that each need several distinct sources reconciled before an answer
exists — the deep-research axis that stresses decomposition rather than
freshness. It is also the most widely reported held-out evaluation across the
research-agent literature, which makes it the natural cross-paper comparison
point.

Golds are sometimes multi-part (a name, an ordinal, a date). The judge handles
those; normalized exact match under-reads here and is recorded as a diagnostic
rather than a headline. Reference links travel in `gold["urls"]` for anyone
wanting to analyze retrieval quality separately.

Source: `google/frames-benchmark`, config `default`, split `test`. Our audited
gold revisions are applied on top at prepare time.

The full split is 824 questions and is the most token-expensive item in the
suite. The reported protocol uses a fixed 100-question prefix; iterate with a
smaller `--n` before running anything larger.
"""

from . import Benchmark, register
from ._common import (AGENTIC_QA_SYSTEM, load_split, qa_rows, resolve_revision,
                      write_prepared)

HF_DATASET = "google/frames-benchmark"
CONFIG = "default"
SPLIT = "test"
PINNED_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
PROMPT_VERSION = 1
MAX_TURNS = 10


def _gold_urls(d):
    """The clean per-row reference links, preferred over the single-quoted
    combined field, which is not valid JSON."""
    return [v for k, v in d.items()
            if k.startswith("wikipedia_link_") and isinstance(v, str)
            and v.strip()]


def build_rows(records):
    """Row construction, kept pure so it is testable without a dataset pull."""
    return qa_rows(
        records, "frames", "Prompt", "Answer", AGENTIC_QA_SYSTEM,
        gold_fn=lambda d: {"answer": d["Answer"], "urls": _gold_urls(d)},
        meta_fn=lambda d: {"reasoning_types": d.get("reasoning_types")})


def _prepare(bench, apply_revisions=True):
    revision = resolve_revision(HF_DATASET, PINNED_REVISION)
    rows = build_rows(load_split(HF_DATASET, CONFIG, SPLIT, revision))
    return write_prepared(bench, rows, revision, {"config": CONFIG},
                          PROMPT_VERSION, HF_DATASET, SPLIT,
                          apply_revisions=apply_revisions)


register(Benchmark(name="frames", prepare_fn=_prepare,
                   agentic=True, max_turns=MAX_TURNS,
                   has_revisions=True,
                   expected_revision=PINNED_REVISION,
                   expected_prompt_version=PROMPT_VERSION,
                   expected_builder={"config": CONFIG}))
