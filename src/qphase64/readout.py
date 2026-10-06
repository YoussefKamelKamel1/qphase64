"""Site-averaged statistics computed from single-shot measurement outcomes."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

ALL_STATS: tuple[str, ...] = ("m_abs", "c1", "c2", "kink", "stag", "m2", "m4")

DEFAULT_STATS: tuple[str, ...] = ("m_abs", "c1", "c2", "stag")

#: The single-statistic arm of the B4 ablation.
SINGLE_STAT: tuple[str, ...] = ("m_abs",)

DAMPING_ROBUST_STATS: tuple[str, ...] = ("c1", "c2", "kink")


def as_pm1(shots: np.ndarray, encoding: str = "auto") -> np.ndarray:
    """Coerce shots to a float ``(n_shots, N)`` array of +/-1."""
    s = np.asarray(shots)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError(f"expected shots of shape (n_shots, N), got {s.shape}")
    s = s.astype(float)

    uniq = np.unique(s)
    if encoding == "pm1":
        as_bits = False
    elif encoding == "bits":
        as_bits = True
    elif encoding == "auto":
        as_bits = bool(uniq.size) and set(uniq.tolist()) <= {0.0, 1.0} and 0.0 in uniq
    else:
        raise ValueError(
            f"encoding must be 'auto', 'pm1' or 'bits', got {encoding!r}"
        )

    if as_bits:
        if not set(uniq.tolist()) <= {0.0, 1.0}:
            raise ValueError(f"encoding='bits' but values are not 0/1: {uniq}")
        return 1.0 - 2.0 * s  # bit 0 -> +1, bit 1 -> -1

    if uniq.size and not set(uniq.tolist()) <= {-1.0, 1.0}:
        raise ValueError(f"expected +/-1 samples, got values {uniq}")
    return s


def _magnetization(s: np.ndarray) -> np.ndarray:
    return s.mean(axis=1)


def _staggered(s: np.ndarray) -> np.ndarray:
    n = s.shape[1]
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return (s * sign).mean(axis=1)


def _corr(s: np.ndarray, d: int) -> np.ndarray:
    """Translation-averaged distance-``d`` ZZ correlation on an open chain."""
    n = s.shape[1]
    if n <= d:
        return np.zeros(s.shape[0])
    return (s[:, : n - d] * s[:, d:]).mean(axis=1)


_STAT_FNS = {
    "m_abs": lambda s: np.abs(_magnetization(s)),
    "m2": lambda s: _magnetization(s) ** 2,
    "m4": lambda s: _magnetization(s) ** 4,
    "stag": lambda s: np.abs(_staggered(s)),
    "c1": lambda s: _corr(s, 1),
    "c2": lambda s: _corr(s, 2),
    "kink": lambda s: (1.0 - _corr(s, 1)) / 2.0,
}


def statistic(
    shots: np.ndarray, name: str, encoding: str = "auto"
) -> np.ndarray:
    """One named statistic, evaluated per shot -> shape ``(n_shots,)``."""
    try:
        fn = _STAT_FNS[name]
    except KeyError:
        raise KeyError(
            f"unknown statistic {name!r}; available: {sorted(_STAT_FNS)}"
        ) from None
    return fn(as_pm1(shots, encoding))


def batch(
    shots: np.ndarray,
    stats: Sequence[str] = DEFAULT_STATS,
    encoding: str = "auto",
) -> np.ndarray:
    """Readout matrix ``X`` of shape ``(n_shots, len(stats))``."""
    s = as_pm1(shots, encoding)
    return np.stack([_STAT_FNS[name](s) for name in stats], axis=1)


def windows(
    shots: np.ndarray,
    width: int,
    stats: Sequence[str] = DEFAULT_STATS,
    *,
    rng: np.random.Generator | None = None,
    n_windows: int = 1,
    encoding: str = "auto",
) -> np.ndarray:
    """Sub-window augmentation (shared_size_transfer.md D3)."""
    s = as_pm1(shots, encoding)
    n = s.shape[1]
    if not 1 <= width <= n:
        raise ValueError(f"width must be in [1, {n}], got {width}")
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for _ in range(n_windows):
        start = int(rng.integers(0, n - width + 1))
        out.append(batch(s[:, start : start + width], stats, encoding="pm1"))
    return np.concatenate(out, axis=0)


def binder(shots: np.ndarray, encoding: str = "auto") -> float:
    """Binder cumulant U = 1 - <m^4> / (3 <m^2>^2), averaged over the given shots."""
    s = as_pm1(shots, encoding)
    m = _magnetization(s)
    m2 = float((m**2).mean())
    m4 = float((m**4).mean())
    if m2 <= 0.0:
        return 0.0
    return 1.0 - m4 / (3.0 * m2**2)


def total_sz(shots: np.ndarray, encoding: str = "auto") -> np.ndarray:
    """Raw sum_i s_i per shot - the U(1) symmetry witness of F2."""
    return as_pm1(shots, encoding).sum(axis=1)


def feature_names(stats: Sequence[str] = DEFAULT_STATS) -> list[str]:
    return list(stats)
