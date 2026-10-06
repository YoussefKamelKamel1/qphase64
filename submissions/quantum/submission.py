"""Competition entry point: quantum brickwork model, flat impostor threshold."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from xxz_quantum import classify  # noqa: E402,F401
