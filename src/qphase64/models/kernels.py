"""Two-qubit kernels shared by the brickwork and the convolution layers."""

from __future__ import annotations

import pennylane as qml

#: name -> number of parameters. Every entry is independent of n_qubits.
KERNEL_SIZES = {"z2": 7, "general": 9, "general15": 15}

#: Kernels that commute with prod_i X_i.
EQUIVARIANT_KERNELS = ("z2",)


def kernel_n_params(name: str) -> int:
    try:
        return KERNEL_SIZES[name]
    except KeyError:
        raise ValueError(
            f"unknown kernel {name!r}; available: {sorted(KERNEL_SIZES)}"
        ) from None


def apply_kernel(name: str, theta, wires) -> None:
    """Apply the two-qubit kernel ``name`` with ``theta`` on ``wires``."""
    a, b = wires
    if name == "z2":
        # Commutes with X (x) X: RX and the three Ising couplings only.
        qml.RX(theta[0], wires=a)
        qml.RX(theta[1], wires=b)
        qml.IsingXX(theta[2], wires=[a, b])
        qml.IsingYY(theta[3], wires=[a, b])
        qml.IsingZZ(theta[4], wires=[a, b])
        qml.RX(theta[5], wires=a)
        qml.RX(theta[6], wires=b)
    elif name == "general":
        qml.RY(theta[0], wires=a)
        qml.RY(theta[1], wires=b)
        qml.IsingXX(theta[2], wires=[a, b])
        qml.IsingYY(theta[3], wires=[a, b])
        qml.IsingZZ(theta[4], wires=[a, b])
        qml.RY(theta[5], wires=a)
        qml.RY(theta[6], wires=b)
        qml.RZ(theta[7], wires=a)
        qml.RZ(theta[8], wires=b)
    elif name == "general15":
        qml.Rot(theta[0], theta[1], theta[2], wires=a)
        qml.Rot(theta[3], theta[4], theta[5], wires=b)
        qml.IsingXX(theta[6], wires=[a, b])
        qml.IsingYY(theta[7], wires=[a, b])
        qml.IsingZZ(theta[8], wires=[a, b])
        qml.Rot(theta[9], theta[10], theta[11], wires=a)
        qml.Rot(theta[12], theta[13], theta[14], wires=b)
    else:
        raise ValueError(f"unknown kernel {name!r}")


#: Single-qubit layer applied identically to every wire (shared angles).
SINGLE_SIZES = {"rx": 1, "ry_rz": 2, "none": 0}


def single_n_params(name: str) -> int:
    try:
        return SINGLE_SIZES[name]
    except KeyError:
        raise ValueError(
            f"unknown single-qubit layer {name!r}; available: {sorted(SINGLE_SIZES)}"
        ) from None


def apply_single(name: str, phi, wires) -> None:
    """Same angles on every wire, so the count stays independent of n_qubits."""
    if name == "none":
        return
    for w in wires:
        if name == "rx":
            qml.RX(phi[0], wires=w)
        elif name == "ry_rz":
            qml.RY(phi[0], wires=w)
            qml.RZ(phi[1], wires=w)
        else:
            raise ValueError(f"unknown single-qubit layer {name!r}")


def bonds(n_qubits: int, parity: int, periodic: bool) -> list[tuple[int, int]]:
    """Brickwork bonds of one sublayer."""
    out = []
    for i in range(parity, n_qubits - 1, 2):
        out.append((i, i + 1))
    if periodic and n_qubits > 2 and (n_qubits - 1) % 2 == parity:
        out.append((n_qubits - 1, 0))
    return out
