"""Audited gold revisions, applied over the upstream benchmarks at prepare time.

Reviewing losses on these benchmarks, we repeatedly found the model had
returned a well-sourced current answer while the stored reference was wrong.
The defects fall into three classes: answers overtaken by events, extraction
errors where the reference answers a different question than the one asked,
and questions with several defensible readings. Left uncorrected, those rows
penalize the right behavior and add noise exactly at the top of the score
range, where model comparisons are decided.

We audited a fixed subset of each benchmark and evaluated on that same subset.
Each audited task went through corroboration across historical model answers,
verification against current primary sources, an independent model review of
every proposed change, and a human review of the final correction set.

This directory ships the audit verdicts only — never upstream questions or
answers. Each line of `revisions/<bench>-revisions.jsonl` is one verdict:

  {"id": "seal0-002", "verdict": "GOLD_STALE",
   "old_answer": "...", "new_answer": "...", "as_of": "2026-07-29",
   "evidence": ["url", ...], "note": "...", "confidence": "high"}

How each verdict is applied to the prepared rows:

  GOLD_STALE / GOLD_WRONG   replace the gold answer with `new_answer`
  TIME_SENSITIVE            with `new_answer`: re-gold as of the audit date.
                            Without one: drop the row — reserved for questions
                            whose sources conflict simultaneously, so no
                            "answer as of today" exists to gold against.
  AMBIGUOUS                 drop the row (a question defect, not drift)
  GOLD_OK / JUDGE_ERROR     no change; kept in the file as the audit trail

Because golds carry an as-of date, a source that moves again later is visible
rather than silent. Revisions are applied on top of freshly pulled upstream
rows and never edited into them, so running with `--no-revisions` reproduces
the unmodified upstream benchmark.
"""

import hashlib
import json
from pathlib import Path

REVISIONS_DIR = Path(__file__).parent / "revisions"

REVISING_VERDICTS = {"GOLD_STALE", "GOLD_WRONG"}
DROPPING_VERDICTS = {"AMBIGUOUS"}
NOOP_VERDICTS = {"GOLD_OK", "JUDGE_ERROR"}
# TIME_SENSITIVE is bimodal: with a new answer it revises, without one it drops.
ALL_VERDICTS = REVISING_VERDICTS | DROPPING_VERDICTS | NOOP_VERDICTS \
    | {"TIME_SENSITIVE"}


def path_for(bench_name):
    return REVISIONS_DIR / f"{bench_name}-revisions.jsonl"


def _action(record):
    """'revise' | 'drop' | None for one validated verdict record."""
    v = record["verdict"]
    if v in REVISING_VERDICTS:
        return "revise"
    if v in DROPPING_VERDICTS:
        return "drop"
    if v == "TIME_SENSITIVE":
        return "revise" if record.get("new_answer") else "drop"
    return None


def load(path):
    """Parse and validate one revision file -> {id: record}."""
    records = {}
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        verdict = r.get("verdict")
        if verdict not in ALL_VERDICTS:
            raise ValueError(f"{Path(path).name}: unknown verdict "
                             f"{verdict!r} for id {r.get('id')!r}")
        if verdict in REVISING_VERDICTS and not r.get("new_answer"):
            raise ValueError(f"{Path(path).name}: {verdict} without "
                             f"new_answer for id {r.get('id')!r}")
        if r["id"] in records:
            raise ValueError(f"{Path(path).name}: duplicate id {r['id']!r}")
        records[r["id"]] = r
    return records


def apply(bench_name, rows):
    """Apply this benchmark's revisions to freshly prepared rows.

    Returns (rows, info) where info is the provenance block frozen into the
    dataset manifest. An id in the revision file that is not in the prepared
    rows is an error: it means the upstream dataset moved under the audit, and
    silently ignoring it would produce a dataset matching neither.
    """
    path = path_for(bench_name)
    if not path.exists():
        raise FileNotFoundError(f"no revision file for {bench_name}: {path}")
    revisions = load(path)
    unknown = set(revisions) - {r.id for r in rows}
    if unknown:
        raise ValueError(
            f"{path.name}: {len(unknown)} revised ids are not in the prepared "
            f"{bench_name} rows (e.g. {sorted(unknown)[:3]}) — the upstream "
            "dataset has changed; re-audit before evaluating")
    out, n_revised, n_dropped = [], 0, 0
    for row in rows:
        rec = revisions.get(row.id)
        action = _action(rec) if rec else None
        if action == "drop":
            n_dropped += 1
            continue
        if action == "revise":
            # Single-answer golds revise "answer"; table golds revise "table"
            # with a corrected row list.
            key = "table" if isinstance(rec["new_answer"], list) else "answer"
            row.gold = dict(row.gold, **{key: rec["new_answer"]})
            row.meta = dict(row.meta or {}, audit={
                "verdict": rec["verdict"],
                "old_answer": rec.get("old_answer"),
                "as_of": rec.get("as_of"),
                "confidence": rec.get("confidence")})
            n_revised += 1
        out.append(row)
    info = {"file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_audited": len(revisions),
            "n_revised": n_revised, "n_dropped": n_dropped}
    return out, info
