"""Quantum phase recognition from at most 64 copies per state."""

from __future__ import annotations

__version__ = "1.0.0"

COPY_BUDGET = 64
SCORED_K = (4, 8, 16)
REPORTED_K = (4, 8, 16, 32, 64)

from qphase64 import readout  # noqa: E402

__all__ = ["COPY_BUDGET", "SCORED_K", "REPORTED_K", "readout"]
