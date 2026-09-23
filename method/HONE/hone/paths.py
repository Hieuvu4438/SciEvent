"""Resolution of paths to the SciEvent benchmark.

CARVE treats the benchmark as a read-only dependency. Nothing in this repository
modifies it. Two locations are needed:

  * the prepared ONEIE-format splits, and
  * the official scorer, which we import rather than reimplement.

Both default to ``third_party/SciEvent`` inside the repository (populated by
``scripts/setup_data.sh``) and can be overridden with the ``SCIEVENT_ROOT``
environment variable.
"""

import os
from pathlib import Path

_SETUP_HINT = (
    "Run `bash scripts/setup_data.sh` from the repository root to fetch and "
    "prepare the SciEvent benchmark, or point the SCIEVENT_ROOT environment "
    "variable at an existing checkout."
)


def repo_root() -> Path:
    """Method root (the directory containing the ``hone`` package)."""
    return Path(__file__).resolve().parents[1]


def scievent_root() -> Path:
    """Root of the SciEvent benchmark checkout."""
    env = os.environ.get("SCIEVENT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    local = repo_root() / "third_party" / "SciEvent"
    if local.is_dir():
        return local
    # research-workspace layout: <workspace>/method/HONE and <workspace>/third_party/SciEvent
    return repo_root().parents[1] / "third_party" / "SciEvent"


def default_data_dir() -> str:
    """Directory holding ``{train,dev,test}.oneie.json``."""
    path = scievent_root() / "SciEvent_data" / "ONEIE" / "all_splits"
    if not path.is_dir():
        raise FileNotFoundError(f"SciEvent splits not found at {path}.\n{_SETUP_HINT}")
    return str(path)


def official_scorer_path() -> str:
    """The benchmark's own scorer, imported verbatim by :mod:`hone.evaluate`."""
    path = scievent_root() / "baselines" / "ONEIE" / "EM_overlap_eval.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official scorer not found at {path}.\n{_SETUP_HINT}")
    return str(path)


def split_path(split: str, data_dir: str | None = None) -> str:
    """Absolute path to one split file."""
    return os.path.join(data_dir or default_data_dir(), f"{split}.oneie.json")
