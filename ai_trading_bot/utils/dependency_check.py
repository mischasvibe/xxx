"""Runtime dependency checks for core modules."""

from __future__ import annotations

from importlib import util as importlib_util
from typing import Iterable

_HELP_MESSAGE = (
    "Missing required packages: {missing}.\n"
    "Activate your virtual environment and run `pip install -r requirements.txt`.\n"
    "If you previously installed optional ML libraries, ensure you are on Python 3.10-3.12 "
    "or install them via `pip install -r requirements-ml.txt`."
)


def ensure_required_packages(packages: Iterable[str]) -> None:
    """Exit early with a helpful message when core dependencies are absent."""

    missing = [pkg for pkg in packages if importlib_util.find_spec(pkg) is None]
    if missing:
        formatted = ", ".join(sorted(set(missing)))
        raise SystemExit(_HELP_MESSAGE.format(missing=formatted))
