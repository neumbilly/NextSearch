"""WideSearch: fill a predefined table schema with ALL qualifying entities.

The wide-research axis in its composite form — one task can require dozens or
hundreds of entities, each enriched with several precise fields. Each task's
query embeds the required columns and the output format (a fenced markdown
table); the gold is a table plus a per-column evaluation pipeline
(exact_match / url_match / number_near / date_near / llm_judge with a
criterion) that our table grader replays from the official evaluator.

The headline metric is row-F1: a row counts only if the entity key and every
enrichment cell are right. Item-F1 and the strict whole-table success rate are
reported alongside; the latter is near zero for every model we have measured,
which is expected.

English split only — the other half needs search quality in a language we do
not evaluate. Gold tables are per-instance CSVs in the dataset repository,
joined into each row's gold at prepare time, so grading needs no further
dataset access.

Source: `ByteDance-Seed/WideSearch`. Our audited gold revisions are applied on
top at prepare time.

This benchmark is the one where the orchestrated harness earns its keep: a
single research context handles the breadth poorly, and solo runs truncate
often. See docs/evals.md for the orchestrated invocation.
"""

import csv
import json

from ..grading import ws_norm_column
from . import Benchmark, register
from ._common import write_prepared

HF_DATASET = "ByteDance-Seed/WideSearch"
LANGUAGE = "en"
PINNED_REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
PROMPT_VERSION = 1
MAX_TURNS = 30

WIDESEARCH_SYSTEM = (
    "You are an expert web researcher with a search tool. The task asks for a "
    "COMPLETE table: find every entity that satisfies the constraints and "
    "fill in every requested column, searching as many times as needed and "
    "verifying values across sources. Missing rows, extra rows, and wrong "
    "cells are all penalized — do not stop at a partial list and do not pad "
    "with unverified entries. Reply with the final table only, in the exact "
    "output format the task specifies (a fenced ```markdown table with "
    "exactly the requested columns), with no commentary inside the block."
)


def load_gold_table(csv_path, required):
    """One gold CSV -> a list of row dicts with normalized column names,
    restricted to the required columns, values kept as raw strings."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    rows = [{ws_norm_column(k): ("" if v is None else str(v))
             for k, v in row.items() if k is not None} for row in raw]
    missing = [c for c in required if rows and c not in rows[0]]
    if missing or not rows:
        raise ValueError(f"{csv_path}: missing required columns {missing}")
    return [{c: row.get(c, "") for c in required} for row in rows]


def build_rows(records):
    """Row construction from (record, gold_table) pairs, kept pure so it is
    testable without a dataset pull."""
    from ..types import Row, system, user
    rows = []
    for record, gold_table in records:
        spec = record["evaluation"]
        if isinstance(spec, str):
            spec = json.loads(spec)
        spec = {"unique_columns": spec["unique_columns"],
                "required": spec["required"],
                "eval_pipeline": spec["eval_pipeline"]}
        rows.append(Row(
            messages=[system(WIDESEARCH_SYSTEM), user(record["query"])],
            gold={"table": gold_table, "evaluation": spec},
            id=record["instance_id"].replace("_", "-"), tools=[],
            meta={"question": record["query"],
                  "instance_id": record["instance_id"],
                  "language": record["language"],
                  "n_gold_rows": len(gold_table),
                  "n_required_columns": len(spec["required"])},
        ))
    return rows


def _prepare(bench, apply_revisions=True):
    from huggingface_hub import snapshot_download

    from ._common import resolve_revision
    revision = resolve_revision(HF_DATASET, PINNED_REVISION)
    root = snapshot_download(
        repo_id=HF_DATASET, repo_type="dataset", revision=revision,
        allow_patterns=["widesearch.jsonl", "widesearch_gold/*"])
    records = []
    with open(f"{root}/widesearch.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["language"] != LANGUAGE:
                continue
            spec = json.loads(record["evaluation"])
            gold = load_gold_table(
                f"{root}/widesearch_gold/{record['instance_id']}.csv",
                spec["required"])
            records.append((record, gold))
    rows = build_rows(records)
    return write_prepared(bench, rows, revision, {"language": LANGUAGE},
                          PROMPT_VERSION, HF_DATASET, "full",
                          apply_revisions=apply_revisions)


register(Benchmark(name="widesearch", prepare_fn=_prepare,
                   agentic=True, max_turns=MAX_TURNS,
                   grade_kind="table",
                   has_revisions=True,
                   expected_revision=PINNED_REVISION,
                   expected_prompt_version=PROMPT_VERSION,
                   expected_builder={"language": LANGUAGE}))
