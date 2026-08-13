"""The prompt-layer documents.

Which document a run uses is decided by the harness (`harnesses.py`); this
package just holds the markdown and loads it.
"""

from pathlib import Path

_DIR = Path(__file__).parent


def load_doc(fname) -> str:
    return (_DIR / fname).read_text().strip()
