"""Competition entry point: shadows baseline."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from baselines import load  # noqa: E402

_CLF = None


def classify(oracle, state_ids):
    global _CLF
    if _CLF is None:
        _CLF = load("shadows")
    return _CLF.run(oracle, state_ids)
