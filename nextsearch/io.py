"""JSONL persistence: one JSON object per line.

Datasets are one file per benchmark — a frozen, hashable artifact. Rollout
files are append-only, so a job killed halfway keeps everything flushed so
far and can resume against it.
"""

import json
from pathlib import Path

from .types import Rollout, Row


def dump_line(d) -> str:
    return json.dumps(d, ensure_ascii=False, default=str)


def _write(path, dicts, append):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for d in dicts:
            f.write(dump_line(d) + "\n")
    return path


def write_rollouts(path, rollouts, append=False):
    return _write(path, (r.to_json() for r in rollouts), append)


def write_rows(path, rows, append=False):
    return _write(path, (r.to_json() for r in rows), append)


def read_jsonl(path):
    # split("\n") rather than splitlines(): JSON strings legally carry raw
    # U+2028/U+2029, which judges do emit when quoting web text, and
    # splitlines() would tear those records in half.
    for line in Path(path).read_text().split("\n"):
        line = line.strip()
        if line:
            yield json.loads(line)


def read_rollouts(path):
    return [Rollout.from_json(d) for d in read_jsonl(path)]


def read_rows(path):
    return [Row.from_json(d) for d in read_jsonl(path)]
