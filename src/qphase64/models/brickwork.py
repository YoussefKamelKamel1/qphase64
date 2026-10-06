"""Translation-invariant brickwork circuit with shared two-qubit kernels."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qphase64.models.kernels import (
    EQUIVARIANT_KERNELS,
    apply_kernel,
    apply_single,
    bonds,
    kernel_n_params,
    single_n_params,
)


class Brickwork:
    """Translation-invariant brickwork classifier."""

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        kernel: str = "z2",
        single: str = "rx",
        periodic: bool = False,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.kernel = kernel
        self.single = single
        self.periodic = bool(periodic)
        self._k = kernel_n_params(kernel)
        self._s = single_n_params(single)

    # ------------------------------------------------------------------ shape

    def n_params(self) -> int:
        """Independent of ``n_qubits`` by construction."""
        return self.layers * (self._k + self._s)

    @property
    def equivariant(self) -> bool:
        return self.kernel in EQUIVARIANT_KERNELS and self.single in ("rx", "none")

    def readout_spec(self) -> tuple[str, ...]:
        from qphase64 import readout

        return readout.DEFAULT_STATS

    def init_params(self, seed: int = 0, scale: float = 0.1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, size=self.n_params())

    def _slice(self, theta, layer: int):
        w = self._k + self._s
        base = layer * w
        return theta[base : base + self._k], theta[base + self._k : base + w]

    # ---------------------------------------------------------------- circuit

    def apply(self, theta) -> None:
        """Queue the brickwork onto the active tape (state already prepared)."""
        wires = range(self.n_qubits)
        for layer in range(self.layers):
            k_theta, s_phi = self._slice(theta, layer)
            for parity in (0, 1):
                for bond in bonds(self.n_qubits, parity, self.periodic):
                    apply_kernel(self.kernel, k_theta, bond)
            apply_single(self.single, s_phi, wires)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Brickwork(n_qubits={self.n_qubits}, layers={self.layers}, "
            f"kernel={self.kernel!r}, single={self.single!r}, "
            f"n_params={self.n_params()}, equivariant={self.equivariant})"
        )
