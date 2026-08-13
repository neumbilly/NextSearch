"""WideSearch-sub: scoring the research subagent in its deployed role.

A composite wide-search task measures a whole orchestrator-plus-subagent
system, so a subagent's contribution reaches the metric only after being
filtered through the orchestrator's planning. That makes it an indirect and
noisy way to compare research models. WideSearch-sub scores the subagent's own
deliverable instead.

We collected the research calls our orchestrated system actually delegated on
WideSearch tasks, deduplicated them, removed prompts that depended on hidden
conversation context, and authored source-backed gold tables for what
remained: 49 self-contained enrichment tasks graded by row-F1 with the same
table grader WideSearch uses.

Every gold record had to carry an as-of date and at least two evidence URLs,
pass structural validation against the table grader, and score 1.0 when
replayed as a perfect answer. A record that could not win its own row was an
unwinnable task and was rejected rather than shipped.

These golds are authored by us, so unlike the other benchmarks the data file
ships here directly (`data/widesearch-sub.jsonl`) rather than being pulled.

**The output-format block.** Only a minority of the harvested tasks asked for
a markdown table, and many named no format at all — but the grader scores an
unparseable response as zero for the whole task. So each row's user message is
the verbatim task plus a standardized block naming that row's exact columns.
This mirrors WideSearch itself, whose queries embed the column list. The block
spells out the line structure explicitly, because a model that emits an entire
table on one line produces a single giant header the parser cannot map, and
that scores zero on formatting while burying real research quality.

Rows run under the subagent role prompt, so the `solo` harness reproduces
exactly what a nested subagent sees: role document, research guidelines, task
date. The 10-turn budget matches the nested subagent budget in deployment.
"""

import hashlib
import json
from pathlib import Path

from ..prompts import load_doc
from . import Benchmark, register
from ._common import write_prepared

DATA_FILE = Path(__file__).parent / "data" / "widesearch-sub.jsonl"
SUBAGENT_DOC = "subagent.md"
PROMPT_VERSION = 2
MAX_TURNS = 10

OUTPUT_FORMAT = (
    "\n\n---\n"
    "Output format: reply with ONLY a fenced ```markdown code block "
    "containing a pipe table, and nothing before or after it.\n"
    "- Line 1 is the header, exactly these columns in this order:\n"
    "  {columns}\n"
    "- Line 2 is the separator row (`|---|---|`).\n"
    "- Then ONE ROW PER LINE. Each row must be on its own line, separated "
    "by real newlines — a table written on a single line cannot be read.\n"
    "- Put \"not found\" in a cell you could not verify. No commentary "
    "inside the block."
)


def build_rows(records, doc):
    """Row construction from gold records, kept pure so it is testable
    without touching the filesystem."""
    from ..types import Row, system, user
    rows = []
    for rec in records:
        columns = rec["columns"]
        spec = {"unique_columns": rec["unique_columns"],
                "required": columns,
                "eval_pipeline": rec["eval_pipeline"]}
        question = rec["task"] + OUTPUT_FORMAT.format(
            columns=" | ".join(columns))
        provenance = rec.get("provenance") or {}
        rows.append(Row(
            messages=[system(doc), user(question)],
            gold={"table": rec["gold_table"], "evaluation": spec},
            id=rec["id"], tools=[],
            meta={"question": question,
                  "task": rec["task"],
                  "as_of": rec.get("as_of"),
                  "n_gold_rows": len(rec["gold_table"]),
                  "n_required_columns": len(columns),
                  # Where the task came from: which orchestrator delegated it,
                  # which upstream WideSearch task it was decomposed from, and
                  # the difficulty strata used when sampling the set.
                  "source_orchestrator":
                      provenance.get("source_orchestrator"),
                  "parent_task_id": provenance.get("parent_task_id"),
                  "size": provenance.get("size"),
                  "hard": provenance.get("hard")},
        ))
    return rows


def _doc_sha():
    return hashlib.sha256(load_doc(SUBAGENT_DOC).encode()).hexdigest()[:12]


def _prepare(bench, apply_revisions=True):
    records = [json.loads(line) for line in
               DATA_FILE.read_text().split("\n") if line.strip()]
    rows = build_rows(records, load_doc(SUBAGENT_DOC))
    data_sha = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    return write_prepared(
        bench, rows, revision=data_sha,
        builder={"data": DATA_FILE.name, "data_sha256": data_sha,
                 "subagent_doc": SUBAGENT_DOC,
                 "subagent_doc_sha": _doc_sha()},
        prompt_version=PROMPT_VERSION,
        hf_dataset=None, split="full", apply_revisions=apply_revisions)


# The role prompt is baked into each row's system message at prepare time, so
# its CONTENT hash is part of the dataset's identity, not just its filename.
# Expecting the CURRENT doc's hash is what makes that bite: edit the prompt
# and every previously prepared dataset fails validation until it is rebuilt.
# Without it, a prompt edit leaves stale rows silently valid and the benchmark
# keeps evaluating an older prompt than the one on disk, while every manifest
# still claims the two agree.
register(Benchmark(name="widesearch-sub", prepare_fn=_prepare,
                   agentic=True, max_turns=MAX_TURNS,
                   grade_kind="table",
                   expected_prompt_version=PROMPT_VERSION,
                   expected_builder={"data": DATA_FILE.name,
                                     "subagent_doc": SUBAGENT_DOC,
                                     "subagent_doc_sha": _doc_sha()}))
