"""Benchmark registry: one entry owns its prepare step, prompts, and protocol.

Registered benchmarks are data plus callables — the pipeline stages never
special-case a benchmark by name.

Preparation always pulls the source dataset from its upstream home at a pinned
revision and writes canonical Row JSONL locally. This repository redistributes
no upstream benchmark data; what it does ship is our audited gold *revisions*
(see `revisions.py`), which are applied on top at prepare time.

Prepared row files are validated against a committed manifest on every load:
content hash, row count, revision, prompt version, and builder settings. A
stale or hand-edited dataset fails loudly instead of silently producing a
number that cannot be compared to anything.
"""

import hashlib
import json
from dataclasses import dataclass

from .. import io
from ..paths import DATASETS_DIR


@dataclass
class Benchmark:
    name: str
    prepare_fn: object              # (bench, apply_revisions) -> dataset manifest
    # What the benchmark NEEDS. Agentic benchmarks get whatever web toolset the
    # selected harness defines; the harness decides what that is.
    agentic: bool = False
    max_turns: int = 1              # 1 = single-shot, no tool loop
    n_default: "int | None" = None
    # Which grader scores this benchmark: "single" judge, "set" for the
    # set-answer autorater, "table" for the table evaluator.
    grade_kind: str = "single"
    has_revisions: bool = False     # a gold-revision file ships for this bench
    expected_revision: "str | None" = None
    expected_prompt_version: "int | None" = None
    expected_builder: "dict | None" = None

    @property
    def rows_path(self):
        return DATASETS_DIR / f"{self.name}.jsonl"

    @property
    def manifest_path(self):
        return DATASETS_DIR / f"{self.name}.manifest.json"

    def prepare(self, apply_revisions=True):
        return self.prepare_fn(self, apply_revisions)

    def dataset_manifest(self):
        if not self.manifest_path.exists() or not self.rows_path.exists():
            raise FileNotFoundError(
                f"{self.name} is not prepared — run "
                f"`nextsearch-eval prepare --benches {self.name}`")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("bench") != self.name:
            raise ValueError(
                f"dataset manifest benchmark mismatch for {self.name}")
        if (self.expected_revision is not None
                and manifest.get("revision") != self.expected_revision):
            raise ValueError(f"prepared dataset revision is stale for "
                             f"{self.name}; re-run prepare")
        if (self.expected_prompt_version is not None
                and manifest.get("prompt_version")
                != self.expected_prompt_version):
            raise ValueError(f"prepared prompt version is stale for "
                             f"{self.name}; re-run prepare")
        builder = manifest.get("builder") or {}
        if self.expected_builder and any(
                builder.get(k) != v for k, v in self.expected_builder.items()):
            raise ValueError(f"prepared dataset builder is stale for "
                             f"{self.name}; re-run prepare")
        actual_hash = hashlib.sha256(self.rows_path.read_bytes()).hexdigest()
        if manifest.get("sha256") != actual_hash:
            raise ValueError(f"prepared dataset hash mismatch for "
                             f"{self.name}; re-run prepare")
        return manifest

    def load_rows(self, n=None):
        if n is not None and n <= 0:
            raise ValueError("n must be positive")
        manifest = self.dataset_manifest()
        rows = io.read_rows(self.rows_path)
        if len(rows) != manifest.get("n_rows"):
            raise ValueError(f"prepared dataset row-count mismatch for "
                             f"{self.name}; re-run prepare")
        ids = [r.id for r in rows]
        if None in ids or len(ids) != len(set(ids)):
            raise ValueError(
                f"prepared dataset {self.name} has missing or duplicate ids")
        return rows[:n] if n is not None else rows


BENCHMARKS = {}


def register(b: Benchmark) -> Benchmark:
    BENCHMARKS[b.name] = b
    return b


def get(name) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(
            f"unknown benchmark {name!r}; registered: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


# Importing a module registers its benchmarks.
from . import sealqa         # noqa: E402,F401
from . import frames         # noqa: E402,F401
from . import deepsearchqa   # noqa: E402,F401
from . import widesearch     # noqa: E402,F401
from . import widesearch_sub  # noqa: E402,F401
