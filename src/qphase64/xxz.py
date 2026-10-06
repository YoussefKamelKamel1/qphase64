"""Quantum models for the organizer XXZ task."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

PHASES = ["FM", "XY", "NEEL"]
N_QUBITS = 16
BUDGET = 64
FEATURES = ("m_abs", "stag")

KIT = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "SEND_TO_PARTICIPANTS-1"
TRAIN_NPZ = KIT / "xxz_public_train.npz"

_TAB: dict | None = None




# --------------------------------------------------------------------- basis


def basis_tables(n: int = N_QUBITS) -> dict:
    """Per-basis-string readout values. Built once; constant in theta."""
    global _TAB
    if _TAB is not None and _TAB["_n"] == n:
        return _TAB
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)
    spin = (1 - 2 * bits).astype(np.float32)
    _TAB = {
        "_n": n,
        "bits": bits,
        "m_abs": np.abs(spin.mean(1)),
        "stag": np.abs((spin * (-1.0) ** np.arange(n)).mean(1)),
        "c1": (spin * np.roll(spin, -1, axis=1)).mean(1),
        "pair": (spin[:, 0::2] * spin[:, 1::2]).mean(1),
        "sz": spin.sum(1),
    }
    return _TAB


_GRIDS: dict = {}


def symbol_grid(features=FEATURES, n: int = N_QUBITS):
    """Distinct readout values, and the map from basis string to symbol index."""
    key = (tuple(features), n)
    if key not in _GRIDS:
        tab = basis_tables(n)
        cols = np.stack([tab[f] for f in features], axis=1).round(9)
        _GRIDS[key] = np.unique(cols, axis=0, return_inverse=True)
    return _GRIDS[key]


def bitflip_channel(prob_vec: np.ndarray, q: float, n: int = N_QUBITS) -> np.ndarray:
    """Independent bit-flip channel of rate q on a distribution over 2^n strings."""
    if q <= 0:
        return prob_vec
    t = prob_vec.reshape((2,) * n).copy()
    m = np.array([[1 - q, q], [q, 1 - q]])
    for ax in range(n):
        t = np.moveaxis(np.tensordot(m, np.moveaxis(t, ax, 0), axes=(1, 0)), 0, ax)
    return t.reshape(-1)


def flip_rate(noise_p: float) -> float:
    return 2.0 * noise_p / 3.0


# ---------------------------------------------------------------------- data


@dataclass
class XXZData:
    psi: np.ndarray            # (n_states, 2**16) real amplitudes
    delta: np.ndarray
    label: np.ndarray          # strings
    ids: list[str]
    meta: dict = field(default_factory=dict)

    def y(self) -> np.ndarray:
        return np.array([PHASES.index(l) for l in self.label])

    def __len__(self) -> int:
        return len(self.psi)


def phase_of(delta: float) -> str:
    return "FM" if delta < -1 else ("NEEL" if delta > 1 else "XY")


def load_public() -> XXZData:
    d = np.load(TRAIN_NPZ, allow_pickle=True)
    return XXZData(
        psi=d["states"].astype(np.float64),
        delta=d["deltas"],
        label=np.array([str(x) for x in d["labels"]]),
        ids=[str(x) for x in d["ids"]],
        meta={"source": "organizer kit", "n_qubits": int(d["num_qubits"])},
    )


def xxz_hamiltonian(delta: float, h: float = 1e-4, n: int = N_QUBITS):
    """XXZ chain with periodic boundaries and the kit's symmetry-breaking field."""
    import scipy.sparse as sp

    dim = 2 ** n
    ix = np.arange(dim)
    sgn = 1 - 2 * ((ix[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)

    diag = np.zeros(dim)
    for i in range(n):
        diag += 0.25 * delta * sgn[:, i] * sgn[:, (i + 1) % n]
    diag -= h * 0.5 * sgn.sum(1)
    ham = sp.diags(diag).tocsr()

    rows, cols = [], []
    for i in range(n):
        j = (i + 1) % n
        bi, bj = 1 << (n - 1 - i), 1 << (n - 1 - j)
        opp = ((ix & bi) != 0) != ((ix & bj) != 0)
        rows.append(ix[opp])
        cols.append(ix[opp] ^ bi ^ bj)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return (ham + sp.csr_matrix((np.full(len(r), 0.5), (r, c)),
                                shape=(dim, dim))).tocsr()


def ground_state(delta: float, n: int = N_QUBITS) -> np.ndarray:
    import scipy.sparse.linalg as sla

    _, v = sla.eigsh(xxz_hamiltonian(delta, n=n), k=1, which="SA", tol=1e-9)
    return v[:, 0] / np.linalg.norm(v[:, 0])


def generate(deltas) -> XXZData:
    """Build XXZ ground states at arbitrary anisotropy. Evaluation only."""
    deltas = np.asarray(deltas, dtype=float)
    psi = np.stack([ground_state(float(d)) for d in deltas])
    return XXZData(psi=psi, delta=deltas,
                   label=np.array([phase_of(float(d)) for d in deltas]),
                   ids=[f"G{k:03d}" for k in range(len(deltas))],
                   meta={"source": "exact diagonalization"})


def near_boundary(n_per_side: int = 20, width: float = 0.28,
                  margin: float = 0.02) -> XXZData:
    """States either side of both boundaries, excluding the boundaries themselves."""
    left = np.linspace(-1 - width, -1 - margin, n_per_side // 2)
    left2 = np.linspace(-1 + margin, -1 + width, n_per_side // 2)
    right = np.linspace(1 - width, 1 - margin, n_per_side // 2)
    right2 = np.linspace(1 + margin, 1 + width, n_per_side // 2)
    return generate(np.concatenate([left, left2, right, right2]))


def impostors(n_states: int = 25) -> np.ndarray:
    """Fermi-Hubbard ground states, Jordan-Wigner mapped to 16 qubits."""
    import pennylane as qml

    [ds] = qml.data.load("qspin", sysname="FermiHubbard", periodicity="open",
                         lattice="chain", layout="1x8")
    step = max(1, len(ds.ground_states) // n_states)
    out = []
    for k in range(0, len(ds.ground_states), step):
        v = np.asarray(ds.ground_states[k], dtype=complex)
        out.append(v / np.linalg.norm(v))
        if len(out) >= n_states:
            break
    return np.stack(out)


# ---------------------------------------------------------------- likelihood


@dataclass
class PhaseLikelihood:
    """P(readout symbol | phase), built from training states under known noise."""

    table: np.ndarray          # (3, n_symbols)
    symbols: np.ndarray        # (n_symbols, d)
    features: tuple
    log_prior: np.ndarray
    temperature: float = 1.0
    out_wires: tuple | None = None      # set for the QCNN symbol readout

    def assign(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.argmin(((X[:, None, :] - self.symbols[None, :, :]) ** 2).sum(-1),
                         axis=1)

    def log_prob(self, shots: np.ndarray) -> np.ndarray:
        """(k, n) shots of 0/1 -> (k, 3) per-copy log-likelihoods."""
        if self.out_wires is not None:
            bits = np.atleast_2d(np.asarray(shots))[:, list(self.out_wires)]
            w = 2 ** np.arange(bits.shape[1] - 1, -1, -1)
            sym = (bits.astype(int) @ w)
        else:
            sym = self.assign(readout_matrix(shots, self.features))
        return np.log(self.table[:, sym]).T / self.temperature

    def posterior_path(self, shots: np.ndarray) -> np.ndarray:
        return self.log_prior[None, :] + np.cumsum(self.log_prob(shots), axis=0)

    def predict(self, shots: np.ndarray) -> int:
        return int(np.argmax(self.posterior_path(shots)[-1]))


def readout_matrix(shots: np.ndarray, features=FEATURES) -> np.ndarray:
    s = 1.0 - 2.0 * np.atleast_2d(np.asarray(shots, dtype=float))
    n = s.shape[1]
    cols = {
        "m_abs": np.abs(s.mean(1)),
        "stag": np.abs((s * (-1.0) ** np.arange(n)).mean(1)),
        "c1": (s * np.roll(s, -1, axis=1)).mean(1),
        "pair": (s[:, 0::2] * s[:, 1::2]).mean(1),
    }
    return np.stack([cols[f] for f in features], axis=1)


def fit_likelihood(class_probs: np.ndarray, noise_p: float, y: np.ndarray,
                   features=FEATURES, smoothing: float = 1e-3,
                   out_wires=None, n_out: int = 2) -> PhaseLikelihood:
    """Build the per-phase table from exact output distributions."""
    q = flip_rate(noise_p)
    if out_wires is not None:
        tab = basis_tables()
        bits = tab["bits"][:, list(out_wires)]
        w = 2 ** np.arange(len(out_wires) - 1, -1, -1)
        sym_of_basis = bits.astype(int) @ w
        symbols = np.arange(2 ** len(out_wires))[:, None].astype(float)
        n_sym = 2 ** len(out_wires)
    else:
        symbols, sym_of_basis = symbol_grid(features)
        n_sym = len(symbols)

    table = np.zeros((len(PHASES), n_sym))
    for c in range(len(PHASES)):
        np.add.at(table[c], sym_of_basis, bitflip_channel(class_probs[c], q))
    table += smoothing
    table /= table.sum(1, keepdims=True)

    counts = np.bincount(y, minlength=len(PHASES)).astype(float)
    return PhaseLikelihood(table=table, symbols=symbols, features=tuple(features),
                           log_prior=np.log(counts / counts.sum()),
                           out_wires=tuple(out_wires) if out_wires else None)


# -------------------------------------------------------------- oracle glue


def make_ops_fn(model, theta):
    """Wrap a model as the ``ops_fn(wires)`` callable the oracle expects."""
    def ops_fn(wires):
        model.apply(theta)
    return ops_fn


# ------------------------------------------------------------- checkpointing


def save_model(path, model_name: str, model_kwargs: dict, theta, likelihood,
               noise_p: float, extra: dict | None = None) -> None:
    """Persist everything needed to rebuild the classifier at submission time."""
    import json
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        theta=np.asarray(theta, dtype=float),
        table=likelihood.table,
        symbols=likelihood.symbols,
        log_prior=likelihood.log_prior,
        meta=json.dumps({
            "model": model_name,
            "model_kwargs": model_kwargs,
            "features": list(likelihood.features),
            "out_wires": list(likelihood.out_wires) if likelihood.out_wires else None,
            "temperature": float(likelihood.temperature),
            "noise_p": float(noise_p),
            **(extra or {}),
        }),
    )


def load_model(path):
    """Rebuild ``(model, theta, likelihood, meta)`` from a checkpoint."""
    import json

    from qphase64.models.brickwork import Brickwork
    from qphase64.models.qcnn import SharedQCNN

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    cls = {"brickwork": Brickwork, "qcnn": SharedQCNN}[meta["model"]]
    model = cls(n_qubits=N_QUBITS, **meta["model_kwargs"])
    lik = PhaseLikelihood(
        table=z["table"], symbols=z["symbols"],
        features=tuple(meta["features"]), log_prior=z["log_prior"],
        temperature=meta["temperature"],
        out_wires=tuple(meta["out_wires"]) if meta["out_wires"] else None,
    )
    return model, z["theta"], lik, meta


def sample_local(psi, model, theta, k, noise_p, rng, n=N_QUBITS):
    """Draw k noisy copies of one state by exact sampling instead of the oracle."""
    return sample_batch(psi[None, :], model, theta, k, noise_p, rng, n)[0]


def sample_batch(psis, model, theta, k, noise_p, rng, n=N_QUBITS,
                 chunk: int = 16):
    """Draw k noisy copies for each of many states, running the circuit once each."""
    psis = np.asarray(psis)
    bits = basis_tables(n)["bits"]
    q = flip_rate(noise_p)
    out = []
    for start in range(0, len(psis), chunk):
        probs = circuit_probs(psis[start:start + chunk], model, theta, n)
        for row in probs:
            pr = np.clip(bitflip_channel(row, q, n), 0, None)
            draws = rng.choice(len(pr), size=k, p=pr / pr.sum())
            out.append(bits[draws])
    return np.stack(out)


def circuit_probs(psi_batch, model, theta, n: int = N_QUBITS) -> np.ndarray:
    """Exact output distribution of the model, no gradient. Training states only."""
    import pennylane as qml
    import torch

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def qnode(states, th):
        qml.StatePrep(states, wires=range(n))
        model.apply(th)
        return qml.probs(wires=range(n))

    with torch.no_grad():
        out = qnode(torch.tensor(np.asarray(psi_batch, dtype=np.complex128)),
                    torch.tensor(np.asarray(theta, dtype=np.float64)))
    return np.asarray(out, dtype=float)
