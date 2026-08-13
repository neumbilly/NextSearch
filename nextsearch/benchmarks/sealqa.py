"""SEAL-0: conflict robustness over live search.

Questions whose top search results are noisy, conflicting, or outdated, so a
model that trusts the first plausible page gets them wrong. This is the
deep-research axis where the harness's source-judging guidance and the task
date matter most.

Source: `vtllms/sealqa`, config `seal_0`, split `test`. Our audited gold
revisions are applied on top at prepare time (see `revisions.py`) — SEAL-0
carried by far the highest defect rate of the benchmarks we audited, which is
what its emphasis on conflicting and freshly changing evidence predicts.

Prepared row files are gitignored: this benchmark ships with a canary string
to detect training contamination, and while the canary column is not copied
into rows, the prepared file is still kept out of version control as a matter
of hygiene.
"""

from .. import io
from ..types import Row, system, user
from . import Benchmark, register
from ._common import parse_maybe_json, write_prepared

HF_DATASET = "vtllms/sealqa"
CONFIG = "seal_0"
SPLIT = "test"
PINNED_REVISION = "62efe18f229ce5b1e4e2a7b056c5dd2fe53223f6"
PROMPT_VERSION = 1
MAX_TURNS = 10

SEAL0_SYSTEM = (
    "You are a research assistant with a web search tool. Web sources may be "
    "conflicting, outdated, or unreliable — search as needed, cross-check, and "
    "reason carefully about which sources to trust. When you are confident, "
    "give a concise final answer."
)


def build_rows(records, bench_id_prefix="seal0"):
    """Row construction, kept pure so it is testable without a dataset pull."""
    rows = []
    for i, d in enumerate(records):
        rows.append(Row(
            messages=[system(SEAL0_SYSTEM), user(d["question"])],
            gold={"answer": d["answer"],
                  "urls": parse_maybe_json(d.get("urls"))},
            id=f"{bench_id_prefix}-{i:03d}",
            tools=[],
            meta={"question": d["question"], "freshness": d.get("freshness"),
                  "question_types": d.get("question_types"),
                  "search_results": d.get("search_results"),
                  "effective_year": d.get("effective_year"),
                  "topic": d.get("topic")},
        ))
    return rows


def _prepare(bench, apply_revisions=True):
    from ._common import load_split, resolve_revision
    revision = resolve_revision(HF_DATASET, PINNED_REVISION)
    rows = build_rows(load_split(HF_DATASET, CONFIG, SPLIT, revision))
    return write_prepared(bench, rows, revision, {"config": CONFIG},
                          PROMPT_VERSION, HF_DATASET, SPLIT,
                          apply_revisions=apply_revisions)


register(Benchmark(name="seal0", prepare_fn=_prepare,
                   agentic=True, max_turns=MAX_TURNS,
                   has_revisions=True,
                   expected_revision=PINNED_REVISION,
                   expected_prompt_version=PROMPT_VERSION,
                   expected_builder={"config": CONFIG}))
