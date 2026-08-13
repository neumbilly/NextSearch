"""Where artifacts live, and how API keys get loaded.

Everything is anchored to a single working root so the package behaves the
same from any current directory. The root defaults to `./nextsearch-runs`
and can be moved with the `NEXTSEARCH_HOME` environment variable.
"""

import os
from pathlib import Path

HOME = Path(os.environ.get("NEXTSEARCH_HOME") or "nextsearch-runs").resolve()

DATASETS_DIR = HOME / "datasets"   # prepared Row JSONL + dataset manifests
RUNS_DIR = HOME / "runs"           # runs/<eval_id>/...
CACHE_DIR = HOME / "cache"         # judge verdicts + raw page extractions


def load_env(path=None):
    """Load KEY=VALUE lines from a .env file into the environment.

    Existing environment variables always win. Hand-rolled so the package
    needs no dotenv dependency; the file is plain assignments with optional
    `#` comments.
    """
    path = Path(path) if path else Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
