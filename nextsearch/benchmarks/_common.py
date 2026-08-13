"""Shared prepare helpers.

Every benchmark module owns its prompts, gold shape, and protocol, but the
mechanical parts — pinned-revision resolution, gold-revision application,
manifest writing, JSON-column parsing — are identical and live here.

Revision policy: each module pins its upstream dataset commit in a
`PINNED_REVISION` constant. Pinning matters more than usual for this task
family, because several of these datasets are actively corrected upstream and
an unpinned pull would silently change what "the benchmark" means between two
of your own runs.
"""

import hashlib
import json
import time

from .. import io
from ..types import Row, system, user

# Agentic short-answer QA: the shared system prompt for benchmarks whose gold
# is a single concise answer reached via web search. The conflict-aware and
# set-answer benchmarks carry their own.
AGENTIC_QA_SYSTEM = (
    "You are a research assistant with a web search tool. Use the search tool "
    "as many times as needed to find the answer, cross-check across sources, "
    "and reason carefully about which to trust. When you are confident, reply "
    "with a short, direct final answer and nothing else — no explanation."
)


def resolve_revision(hf_dataset, pinned):
    """The pinned commit, or resolve-and-print the current one."""
    if pinned:
        return pinned
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(hf_dataset).sha
    print(f"[prepare] {hf_dataset} revision UNPINNED -> resolved {sha}")
    return sha


def load_split(hf_dataset, config, split, revision):
    from datasets import load_dataset
    return load_dataset(hf_dataset, config, split=split, revision=revision)


def parse_maybe_json(v):
    """Some dataset columns arrive as JSON-encoded strings."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return v or []


def qa_rows(records, bench_id_prefix, question_key, answer_key, system_prompt,
            meta_fn=None, gold_fn=None):
    """Row construction for single-answer agentic QA: one system+user message
    pair per record, gold={answer, ...}, stable ids `<prefix>-<i:04d>`.

    `tools=[]` on purpose — tool specs attach at rollout time from whichever
    harness is selected, so one prepared dataset serves every configuration.
    """
    rows = []
    for i, d in enumerate(records):
        gold = gold_fn(d) if gold_fn else {"answer": d[answer_key]}
        meta = {"question": d[question_key]}
        if meta_fn:
            meta.update(meta_fn(d))
        rows.append(Row(
            messages=[system(system_prompt), user(d[question_key])],
            gold=gold, id=f"{bench_id_prefix}-{i:04d}", tools=[], meta=meta,
        ))
    return rows


def write_prepared(bench, rows, revision, builder, prompt_version,
                   hf_dataset, split, apply_revisions=True):
    """Apply gold revisions if the benchmark has them, then write the prepared
    Row JSONL and its manifest.

    The manifest carries exactly the keys `Benchmark.dataset_manifest()`
    validates, plus provenance. When revisions are applied their file hash and
    counts are frozen into the builder block, so a revised dataset is exactly
    as reproducible as an unrevised one.
    """
    from ..manifest import git_sha
    from . import revisions as revisions_mod

    builder = dict(builder)
    if bench.has_revisions and apply_revisions:
        rows, rev_info = revisions_mod.apply(bench.name, rows)
        builder["revisions"] = rev_info
    else:
        builder["revisions"] = None
    path = io.write_rows(bench.rows_path, rows)
    manifest = {
        "bench": bench.name, "hf_dataset": hf_dataset, "revision": revision,
        "split": split, "n_rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_version": prompt_version, "builder": builder,
        "code": git_sha(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    bench.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rev = builder.get("revisions")
    note = (f" ({rev['n_revised']} golds revised, {rev['n_dropped']} dropped)"
            if rev else "")
    print(f"[prepare] {bench.name}: {len(rows)} rows{note} -> {path}")
    return manifest
