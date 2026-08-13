"""DeepSearchQA: comprehensiveness, where the answer is a complete SET.

Time-anchored prompts whose answers are exhaustive sets of items. They test
collation across sources, deduplication and entity resolution, and knowing
when to stop — the closest public proxy for research-subagent utility, and the
wide-research axis a single-answer benchmark cannot reach.

This is the benchmark that uses the set grader (`grade_kind="set"`). The
official whole-response autorater receives the raw prompt, raw gold, answer
type, and raw model response, and returns per-gold-part correctness plus any
extraneous answers; precision, recall, and F1 are derived from that. The gold
is deliberately not split on delimiters — commas occur inside entity names and
composite attributes in the released data.

Source: `google/deepsearchqa`, config `deepsearchqa`, split `eval`. Our audited
gold revisions are applied on top at prepare time.

The turn budget is larger than the short-answer benchmarks': comprehensiveness
needs more searches.
"""

from . import Benchmark, register
from ._common import load_split, qa_rows, resolve_revision, write_prepared

HF_DATASET = "google/deepsearchqa"
CONFIG = "deepsearchqa"
SPLIT = "eval"
PINNED_REVISION = "b2623f8653065c2672de6d941fc5434cd652376c"
PROMPT_VERSION = 2
MAX_TURNS = 20

DEEPSEARCHQA_SYSTEM = (
    "You are a research assistant with a web search tool. Many questions ask "
    "for a COMPLETE set of items satisfying several constraints: search "
    "thoroughly, cross-check sources, and collate every item that qualifies "
    "while excluding those that do not. Do not pad the answer with "
    "low-confidence guesses — extra wrong items are penalized as much as "
    "missing ones. When done, give a clear, concise final answer containing "
    "only the requested result. Preserve any grouping or attributes needed to "
    "distinguish composite answer items."
)


def build_rows(records):
    """Row construction, kept pure so it is testable without a dataset pull."""
    return qa_rows(
        records, "deepsearchqa", "problem", "answer", DEEPSEARCHQA_SYSTEM,
        gold_fn=lambda d: {"answer": d["answer"],
                           "answer_type": d["answer_type"]},
        meta_fn=lambda d: {"problem_category": d.get("problem_category"),
                           "answer_type": d.get("answer_type")})


def _prepare(bench, apply_revisions=True):
    revision = resolve_revision(HF_DATASET, PINNED_REVISION)
    rows = build_rows(load_split(HF_DATASET, CONFIG, SPLIT, revision))
    return write_prepared(bench, rows, revision, {"config": CONFIG},
                          PROMPT_VERSION, HF_DATASET, SPLIT,
                          apply_revisions=apply_revisions)


register(Benchmark(name="deepsearchqa", prepare_fn=_prepare,
                   agentic=True, max_turns=MAX_TURNS,
                   grade_kind="set",
                   has_revisions=True,
                   expected_revision=PINNED_REVISION,
                   expected_prompt_version=PROMPT_VERSION,
                   expected_builder={"config": CONFIG}))
