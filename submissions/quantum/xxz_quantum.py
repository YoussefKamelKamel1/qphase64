"""Quantum submission: trained brickwork circuit with the flat impostor threshold."""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pennylane as qml

PHASES = ["FM", "XY", "NEEL"]
N_QUBITS = 16
BUDGET = 64

THRESHOLD_MODE = os.environ.get("QPHASE_THRESHOLD_MODE", "sqrtk")
FLAT_THRESHOLD = 0.0
FALSE_REJECT_SIGMAS = 3.0      # only used by the "sqrtk" mode

_HERE = pathlib.Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "models"

# =============================================================== architectures

def apply_kernel(name, theta, wires):
    """Two-qubit block, reused at every bond. `theta` is indexed positionally."""
    a, b = wires
    if name == "z2":
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
    else:
        raise ValueError(f"unknown kernel {name!r}")


KERNEL_SIZES = {"z2": 7, "general": 9}
SINGLE_SIZES = {"rx": 1, "ry_rz": 2, "none": 0}
POOL_SIZES = {"z2pool": 4, "cry": 3}


def apply_single(name, phi, wires):
    """Same angle on every wire, so the count stays independent of n_qubits."""
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


def apply_pool(name, theta, keep, drop):
    """Move information from `drop` into `keep`; `drop` is then unused."""
    if name == "z2pool":
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


def bonds(n_qubits, parity, periodic):
    """Brickwork bonds of one sublayer: parity 0 -> (0,1),(2,3)...; 1 -> (1,2)..."""
    out = [(i, i + 1) for i in range(parity, n_qubits - 1, 2)]
    if periodic and n_qubits > 2 and (n_qubits - 1) % 2 == parity:
        out.append((n_qubits - 1, 0))
    return out


class Brickwork:
    """One two-qubit gate applied at every bond, `layers` times."""

    def __init__(self, n_qubits=N_QUBITS, layers=2, kernel="z2", single="rx",
                 periodic=True):
        self.n_qubits, self.layers = n_qubits, layers
        self.kernel, self.single, self.periodic = kernel, single, periodic
        self._k, self._s = KERNEL_SIZES[kernel], SINGLE_SIZES[single]

    def n_params(self):
        return self.layers * (self._k + self._s)

    def apply(self, theta):
        for layer in range(self.layers):
            w = self._k + self._s
            base = layer * w
            k_theta = theta[base: base + self._k]
            s_phi = theta[base + self._k: base + w]
            for parity in (0, 1):
                for bond in bonds(self.n_qubits, parity, self.periodic):
                    apply_kernel(self.kernel, k_theta, bond)
            apply_single(self.single, s_phi, range(self.n_qubits))

    def __repr__(self):
        return (f"Brickwork(n_qubits={self.n_qubits}, layers={self.layers}, "
                f"kernel={self.kernel!r}, n_params={self.n_params()})")


class SharedQCNN:
    """Convolve, drop half the wires, repeat — every round shares one parameter set."""

    def __init__(self, n_qubits=N_QUBITS, kernel="z2", pool="z2pool", n_out=2,
                 conv_reps=1, periodic=True):
        self.n_qubits, self.kernel, self.pool = n_qubits, kernel, pool
        self.n_out, self.conv_reps, self.periodic = n_out, conv_reps, periodic
        self._k, self._p = KERNEL_SIZES[kernel], POOL_SIZES[pool]
        self._f = self._k if n_out == 2 else 3

    def levels(self):
        active, out = list(range(self.n_qubits)), []
        while len(active) > self.n_out:
            out.append(list(active))
            nxt = active[::2]           # keeps the trailing wire when odd
            if len(nxt) < self.n_out:
                break
            active = nxt
        return out + [list(active)]

    def n_levels(self):
        return len(self.levels()) - 1

    def output_wires(self):
        return self.levels()[-1]

    def n_params(self):
        return self.conv_reps * self._k + self._p + self._f

    def apply(self, theta):
        per_level = self.conv_reps * self._k + self._p
        final = theta[-self._f:]
        stages = self.levels()
        for active in stages[:-1]:
            conv = [theta[r * self._k: (r + 1) * self._k]
                    for r in range(self.conv_reps)]
            pool_theta = theta[self.conv_reps * self._k: per_level]
            for conv_theta in conv:
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

    def __repr__(self):
        return (f"SharedQCNN(n_qubits={self.n_qubits}, rounds={self.n_levels()}, "
                f"n_out={self.n_out}, n_params={self.n_params()}, "
                f"out_wires={self.output_wires()})")


ARCHITECTURES = {"brickwork": Brickwork, "qcnn": SharedQCNN}
# ==================================================================== readout

def readout(shots, features):
    """Site-averaged statistics from raw 0/1 measurements."""
    s = 1.0 - 2.0 * np.atleast_2d(np.asarray(shots, dtype=float))
    n = s.shape[1]
    cols = {
        "m_abs": np.abs(s.mean(1)),                              # magnetization
        "stag": np.abs((s * (-1.0) ** np.arange(n)).mean(1)),    # Neel order
        "c1": (s * np.roll(s, -1, axis=1)).mean(1),              # neighbours
        "pair": (s[:, 0::2] * s[:, 1::2]).mean(1),               # wire pairs
    }
    return np.stack([cols[f] for f in features], axis=1)


def family_gap(shots):
    """`pair - c1`: the post-circuit impostor statistic."""
    r = readout(shots, ("pair", "c1"))
    return float(np.mean(r[:, 0] - r[:, 1]))


# ================================================================== the model

class XXZClassifier:
    """A trained circuit plus its phase tables and its impostor threshold."""

    def __init__(self, model, theta, table, symbols, log_prior, features,
                 out_wires, gap_mu, gap_sd, meta):
        self.model, self.theta = model, theta
        self.table, self.symbols = table, symbols
        self.log_prior, self.features = log_prior, tuple(features)
        self.out_wires = tuple(out_wires) if out_wires else None
        self.gap_mu, self.gap_sd, self.meta = gap_mu, gap_sd, meta

    # ---- the callable the oracle wants ------------------------------------
    def ops_fn(self, wires=None):
        self.model.apply(self.theta)

    # ---- phase ------------------------------------------------------------
    def _symbols_of(self, shots):
        if self.out_wires is not None:                # QCNN: read output wires
            bits = np.atleast_2d(np.asarray(shots))[:, list(self.out_wires)]
            w = 2 ** np.arange(bits.shape[1] - 1, -1, -1)
            return bits.astype(int) @ w
        x = readout(shots, self.features)             # brickwork: nearest symbol
        return np.argmin(((x[:, None, :] - self.symbols[None, :, :]) ** 2).sum(-1),
                         axis=1)

    def log_likelihood(self, shots):
        """(k, 3) per-copy log P(readout | phase)."""
        return np.log(self.table[:, self._symbols_of(shots)]).T

    def phase(self, shots):
        """Accumulate the evidence over copies and take the best phase."""
        total = self.log_prior + self.log_likelihood(shots).sum(axis=0)
        return PHASES[int(np.argmax(total))]

    # ---- family -----------------------------------------------------------
    def threshold(self, k):
        """Rejection bound for `pair - c1`."""
        if THRESHOLD_MODE == "flat":
            return FLAT_THRESHOLD
        return self.gap_mu - FALSE_REJECT_SIGMAS * self.gap_sd / np.sqrt(max(k, 1))

    def is_impostor(self, shots):
        return family_gap(shots) < self.threshold(len(shots))

    # ---- both -------------------------------------------------------------
    def predict(self, shots):
        """One label for one state, from its measurements."""
        shots = np.atleast_2d(np.asarray(shots))
        if len(shots) == 0:
            return "UNKNOWN"
        return "UNKNOWN" if self.is_impostor(shots) else self.phase(shots)

    def __repr__(self):
        return (f"XXZClassifier({self.model!r}, features={self.features}, "
                f"reject below {self.threshold(BUDGET):+.4f} at {BUDGET} copies)")


def load(which="brickwork", path=None):
    """Load a checkpoint. `which` is "brickwork" or "qcnn", or give a path."""
    if path is None:
        path = _MODEL_DIR / f"xxz_{which}.npz"
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected the checkpoints next to this file in "
            f"models/: xxz_brickwork.npz, xxz_qcnn.npz")

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    model = ARCHITECTURES[meta["model"]](n_qubits=N_QUBITS, **meta["model_kwargs"])

    gap_mu, gap_sd = meta.get("gap_mu"), meta.get("gap_sd")
    if gap_mu is None:                       # older checkpoints: measured values
        gap_mu, gap_sd = (0.0865, 0.2427) if meta["model"] == "brickwork" \
            else (0.0, 0.2500)
    return XXZClassifier(model, z["theta"], z["table"], z["symbols"],
                         z["log_prior"], meta["features"],
                         meta.get("out_wires"), gap_mu, gap_sd, meta)


# ============================================================== oracle driver

def classify(oracle, state_ids, clf=None, copies=BUDGET):
    """Label every id."""
    if clf is None:
        clf = load("brickwork")

    out = {}
    for sid in state_ids:
        budget = min(copies, getattr(oracle, "copy_budget", copies))
        shots = []
        for _ in range(budget):
            try:
                shots.append(oracle.measure(sid, ops_fn=clf.ops_fn))
            except Exception:            # budget exhausted or oracle refused
                break
        out[sid] = clf.predict(np.stack(shots)) if shots else "UNKNOWN"
    return out


if __name__ == "__main__":
    print(f"threshold mode: {THRESHOLD_MODE}"
          f"{'  (flat 0.0)' if THRESHOLD_MODE == 'flat' else '  (mu - 3 sd/sqrt(k))'}")
    for which in ("brickwork", "qcnn"):
        try:
            clf = load(which)
        except FileNotFoundError as e:
            print(f"{which}: {e}")
            continue
        print(f"{which:10s} {clf.model!r}")
        print(f"{'':10s} theta {len(clf.theta)} values, tables {clf.table.shape}, "
              f"trained at noise_p={clf.meta.get('noise_p')}")
        print(f"{'':10s} reject below "
              + ", ".join(f"k={k}: {clf.threshold(k):+.4f}" for k in (4, 16, 64)))
