"""Runtime dependency and interpreter checks for the trading bot."""

from __future__ import annotations

import sys
from importlib import util as importlib_util
from typing import Iterable, Tuple
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

_PYTHON_HELP = (
    "This project currently supports Python versions between {min_ver} (inclusive) and {max_ver} "
    "(exclusive).\n"
    "Install a supported interpreter (for example Python 3.11 via Homebrew on macOS) "
    "and recreate your virtual environment before installing dependencies."
)


def ensure_python_version(
    minimum: Tuple[int, int],
    maximum_exclusive: Tuple[int, int],
) -> None:
    """Abort execution when the interpreter version is outside the supported range."""

    current = sys.version_info[:2]
    if current < minimum or current >= maximum_exclusive:
        min_str = ".".join(map(str, minimum))
        max_str = ".".join(map(str, maximum_exclusive))
        raise SystemExit(_PYTHON_HELP.format(min_ver=min_str, max_ver=max_str))



def ensure_required_packages(packages: Iterable[str]) -> None:
    """Exit early with a helpful message when core dependencies are absent."""

    missing = [pkg for pkg in packages if importlib_util.find_spec(pkg) is None]
    if missing:
        formatted = ", ".join(sorted(set(missing)))
        raise SystemExit(_HELP_MESSAGE.format(missing=formatted))
