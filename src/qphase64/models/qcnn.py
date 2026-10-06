"""Quantum convolutional network with parameters tied across coarse-graining rounds."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qphase64.models.kernels import (
    EQUIVARIANT_KERNELS,
    apply_kernel,
    kernel_n_params,
)

class SizeTransferError(RuntimeError):
    """Raised when a model's parameters cannot be applied at a larger N."""


POOL_SIZES = {"z2pool": 4, "cry": 3}
EQUIVARIANT_POOLS = ("z2pool",)


def pool_n_params(name: str) -> int:
    try:
        return POOL_SIZES[name]
    except KeyError:
        raise ValueError(
            f"unknown pool {name!r}; available: {sorted(POOL_SIZES)}"
        ) from None


def apply_pool(name: str, theta, keep: int, drop: int) -> None:
    if name == "z2pool":
        # Commutes with X (x) X, so the whole flow stays Z2-equivariant.
        qml.IsingXX(theta[0], wires=[keep, drop])
        qml.IsingYY(theta[1], wires=[keep, drop])
        qml.IsingZZ(theta[2], wires=[keep, drop])
        qml.RX(theta[3], wires=keep)
    elif name == "cry":
        qml.CRY(theta[0], wires=[drop, keep])
        qml.CRZ(theta[1], wires=[drop, keep])
        qml.RY(theta[2], wires=keep)
    else:
        raise ValueError(f"unknown pool {name!r}")


class SharedQCNN:
    """QCNN whose conv and pool parameters are shared across all levels."""

    def __init__(
        self,
        n_qubits: int,
        kernel: str = "z2",
        pool: str = "z2pool",
        n_out: int = 2,
        conv_reps: int = 1,
        tied: bool = True,
        periodic: bool = False,
    ) -> None:
        if n_out not in (1, 2):
            raise ValueError(f"n_out must be 1 or 2, got {n_out}")
        self.n_qubits = int(n_qubits)
        self.kernel = kernel
        self.pool = pool
        self.n_out = int(n_out)
        self.conv_reps = int(conv_reps)
        self.tied = bool(tied)
        self.periodic = bool(periodic)
        self._k = kernel_n_params(kernel)
        self._p = pool_n_params(pool)
        self._f = kernel_n_params(kernel) if n_out == 2 else 3

    # ------------------------------------------------------------------ shape

    def levels(self) -> list[list[int]]:
        """Active wires at the start of each level, then the surviving output."""
        active = list(range(self.n_qubits))
        out = []
        while len(active) > self.n_out:
            out.append(list(active))
            nxt = active[::2]
            if len(nxt) < self.n_out:  # never pool below n_out
                break
            active = nxt
        return out + [list(active)]

    def n_levels(self) -> int:
        return len(self.levels()) - 1

    def n_params(self) -> int:
        """Constant in ``n_qubits`` when ``tied`` (the legal configuration)."""
        per_level = self.conv_reps * self._k + self._p
        if self.tied:
            return per_level + self._f
        return self.n_levels() * per_level + self._f  # diagnostic only

    @property
    def equivariant(self) -> bool:
        return self.kernel in EQUIVARIANT_KERNELS and self.pool in EQUIVARIANT_POOLS

    def output_wires(self) -> list[int]:
        return self.levels()[-1]

    def readout_spec(self) -> tuple[str, ...]:
        from qphase64 import readout

        return readout.DEFAULT_STATS

    def init_params(self, seed: int = 0, scale: float = 0.1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, size=self.n_params())

    # ---------------------------------------------------------------- circuit

    def _level_slice(self, theta, level: int):
        per_level = self.conv_reps * self._k + self._p
        base = 0 if self.tied else level * per_level
        if not self.tied and base + per_level > len(theta) - self._f:
            raise SizeTransferError(
                f"untied SharedQCNN has parameters for "
                f"{(len(theta) - self._f) // per_level} level(s) but N="
                f"{self.n_qubits} needs {self.n_levels()}. This is the point of "
                "the tied-vs-untied ablation: an untied QCNN cannot transfer to a "
                "larger system at all, which is why it is not submission-legal."
            )
        conv = [
            theta[base + r * self._k : base + (r + 1) * self._k]
            for r in range(self.conv_reps)
        ]
        pool = theta[base + self.conv_reps * self._k : base + per_level]
        return conv, pool

    def apply(self, theta) -> None:
        """Queue conv/pool levels then the final block (state already prepared)."""
        per_level = self.conv_reps * self._k + self._p
        final = theta[-self._f :] if self.tied else theta[
            self.n_levels() * per_level :
        ]

        stages = self.levels()
        for level, active in enumerate(stages[:-1]):
            conv_thetas, pool_theta = self._level_slice(theta, level)

            for conv_theta in conv_thetas:
                for parity in (0, 1):
                    for i in range(parity, len(active) - 1, 2):
                        apply_kernel(self.kernel, conv_theta,
                                     (active[i], active[i + 1]))
                    if self.periodic and len(active) > 2 and \
                            (len(active) - 1) % 2 == parity:
                        apply_kernel(self.kernel, conv_theta,
                                     (active[-1], active[0]))

            for i in range(0, len(active) - 1, 2):
                apply_pool(self.pool, pool_theta, active[i], active[i + 1])

        out = stages[-1]
        if self.n_out == 2 and len(out) >= 2:
            apply_kernel(self.kernel, final, (out[0], out[1]))
        else:
            qml.Rot(final[0], final[1], final[2], wires=out[0])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SharedQCNN(n_qubits={self.n_qubits}, levels={self.n_levels()}, "
            f"kernel={self.kernel!r}, pool={self.pool!r}, n_out={self.n_out}, "
            f"tied={self.tied}, n_params={self.n_params()}, "
            f"out_wires={self.output_wires()})"
        )
