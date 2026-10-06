"""The three classical baselines, frozen and standalone."""

from __future__ import annotations

import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_MODELS = _HERE / "models"
PHASES = ["FM", "XY", "NEEL"]
DEFAULT_THRESHOLD = -0.0816      # calibrated at 64 copies; every
                                 # baseline now spends the same budget


# ------------------------------------------------------------------ shared

def readout(shots):
    """(k, n) of 0/1 -> per-copy site-averaged statistics."""
    s = 1.0 - 2.0 * np.atleast_2d(np.asarray(shots, dtype=float))
    n = s.shape[1]
    return {
        "m_abs": np.abs(s.mean(1)),
        "stag": np.abs((s * (-1.0) ** np.arange(n)).mean(1)),
        "c1": (s * np.roll(s, -1, axis=1)).mean(1),
        "pair": (s[:, 0::2] * s[:, 1::2]).mean(1),
    }


def family_gap(shots):
    """mean(pair - c1). Around 0 for XXZ, around -0.25 for Fermi-Hubbard."""
    r = readout(shots)
    return float(np.mean(r["pair"] - r["c1"]))


class _Base:
    """Shared plumbing: per-copy log-likelihoods in, one label out."""

    name = "base"
    applies_gates = False

    def __init__(self, z):
        self.log_prior = z["log_prior"]
        self.phases = [str(p) for p in z["phases"]]
        self.copies = int(z["copies"])
        self.noise_p = float(z["noise_p"])
        self.n_qubits = int(z["n_qubits"])
        self.threshold = (float(z["threshold"]) if "threshold" in z
                          else DEFAULT_THRESHOLD)

    def log_lik(self, shots):
        """(k, n) shots -> (k, 3) per-copy log P(reading | phase)."""
        raise NotImplementedError

    def phase(self, shots):
        # The prior is added ONCE; the per-copy terms are summed.
        total = self.log_prior + np.asarray(self.log_lik(shots)).sum(axis=0)
        return self.phases[int(np.argmax(total))]

    def is_impostor(self, shots):
        return family_gap(shots) < self.threshold

    def predict(self, shots):
        shots = np.atleast_2d(np.asarray(shots))
        if len(shots) == 0:
            return "UNKNOWN"
        return "UNKNOWN" if self.is_impostor(shots) else self.phase(shots)

    def measure_one(self, oracle, sid, k):
        """Collect up to k copies. The shadow baseline overrides this."""
        shots = []
        for _ in range(k):
            try:
                shots.append(oracle.measure(sid))
            except Exception:
                break               # out of budget: decide on what arrived
        return np.stack(shots) if shots else np.zeros((0, self.n_qubits), int)

    def predict_measured(self, data):
        return self.predict(data)

    def run(self, oracle, state_ids, copies=None):
        want = self.copies if copies is None else int(copies)
        out = {}
        for sid in state_ids:
            k = min(want, getattr(oracle, "copy_budget", want))
            out[sid] = self.predict_measured(self.measure_one(oracle, sid, k))
        return out

    def __repr__(self):
        return (f"{type(self).__name__}(copies={self.copies}, "
                f"reject below {self.threshold:+.4f}, "
                f"applies_gates={self.applies_gates})")


# ------------------------------------------------- order-parameter grid (G2)

class OrderParam(_Base):
    """Nearest-symbol lookup on (|m|, staggered magnetization)."""

    name = "orderparam"

    def __init__(self, z):
        super().__init__(z)
        self.symbols, self.table = z["symbols"], z["table"]
        self.features = tuple(str(f) for f in z["features"])

    def log_lik(self, shots):
        r = readout(shots)
        x = np.stack([r[f] for f in self.features], axis=1)
        sym = np.argmin(((x[:, None, :] - self.symbols[None, :, :]) ** 2).sum(-1),
                        axis=1)
        return np.log(self.table[:, sym]).T


# -------------------------------------------------- CNN on raw bitstrings (G3)

class ShotCNN(_Base):
    """1D CNN over sites, circular padding, global mean pooling."""

    name = "cnn"

    def __init__(self, z):
        super().__init__(z)
        self.w0, self.b0 = z["w0"], z["b0"]      # (ch, 1, 3), (ch,)
        self.w1, self.b1 = z["w1"], z["b1"]      # (ch, ch, 3), (ch,)
        self.wh, self.bh = z["wh"], z["bh"]      # (3, ch), (3,)

    @staticmethod
    def _conv1d_circular(x, w, b):
        """x (batch, cin, n) -> (batch, cout, n). Kernel size 3, circular pad."""
        xp = np.concatenate([x[:, :, -1:], x, x[:, :, :1]], axis=2)
        n, ksz = x.shape[2], w.shape[2]
        idx = np.arange(n)[:, None] + np.arange(ksz)[None, :]
        win = xp[:, :, idx]                       # (batch, cin, n, ksz)
        return np.einsum("bcnk,ock->bon", win, w) + b[None, :, None]

    def log_lik(self, shots):
        x = (1.0 - 2.0 * np.atleast_2d(np.asarray(shots, dtype=float)))[:, None, :]
        h = np.maximum(self._conv1d_circular(x, self.w0, self.b0), 0.0)
        h = np.maximum(self._conv1d_circular(h, self.w1, self.b1), 0.0)
        logits = h.mean(axis=2) @ self.wh.T + self.bh
        logits = logits - logits.max(axis=1, keepdims=True)
        post = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return post - self.log_prior[None, :]


# ------------------------------------- shadows + logistic regression (G1)

class ShadowsLR(_Base):
    """Random-basis measurements -> Pauli estimates -> logistic regression."""

    name = "shadows"
    applies_gates = True

    def __init__(self, z):
        super().__init__(z)
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.mean, self.scale = z["mean"], z["scale"]

    @staticmethod
    def features(bits, base):
        """Translation-averaged weight-1 and weight-2 Pauli estimates."""
        s = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        base = np.asarray(base)
        f = []
        for b in range(3):                                  # <X_i>, <Y_i>, <Z_i>
            m = (base == b).astype(float)
            f.append(float((3.0 * s * m).sum(0).mean() / max(m.sum(0).mean(), 1e-9)))
        for d in (1, 2):                                    # <P_i P_{i+d}>
            for b in range(3):
                m = ((base == b) & (np.roll(base, -d, 1) == b)).astype(float)
                pr = s * np.roll(s, -d, axis=1)
                f.append(float((9.0 * pr * m).sum(0).mean()
                               / max(m.sum(0).mean(), 1e-9)))
        f.append(float(((3.0 * s * (base == 2)).mean(1) ** 2).mean()))
        return np.array(f)

    def basis_ops(self, basis):
        import pennylane as qml        # only this baseline needs PennyLane

        def ops(wires):
            for w, b in zip(wires, basis):
                if b == 0:
                    qml.Hadamard(w)
                elif b == 1:
                    qml.adjoint(qml.S(w))
                    qml.Hadamard(w)
        return ops

    def measure_one(self, oracle, sid, k):
        rng = np.random.default_rng(abs(hash(str(sid))) % (2 ** 32))
        bits, base = [], []
        for _ in range(k):
            b = rng.integers(0, 3, size=self.n_qubits)
            try:
                bits.append(oracle.measure(sid, ops_fn=self.basis_ops(b)))
            except Exception:
                break
            base.append(b)
        return (np.stack(bits).astype(float), np.stack(base)) if bits else None

    @staticmethod
    def shadow_family_gap(bits, base):
        """`pair - c1` estimated from randomly-rotated snapshots."""
        s = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        base = np.asarray(base)
        z = (base == 2)

        def corr(shift_bits, shift_z):
            m = (z & shift_z).astype(float)
            tot = m.sum()
            if tot < 1:
                return None
            return float((s * shift_bits * m).sum() / tot)

        c1 = corr(np.roll(s, -1, axis=1), np.roll(z, -1, axis=1))
        # wire pairs (0,1), (2,3), ...: compare even wires against their partner
        me, mo = z[:, 0::2], z[:, 1::2]
        m = (me & mo).astype(float)
        tot = m.sum()
        pair = float((s[:, 0::2] * s[:, 1::2] * m).sum() / tot) if tot >= 1 else None
        if c1 is None or pair is None:
            return None
        return pair - c1

    def predict_measured(self, data):
        if data is None:
            return "UNKNOWN"
        bits, base = data
        gap = self.shadow_family_gap(bits, base)
        if gap is not None and gap < self.threshold * 3.0:
            return "UNKNOWN"
        x = (self.features(bits, base) - self.mean) / self.scale
        return self.phases[int(np.argmax(self.coef @ x + self.intercept))]


# ------------------------------------------------------------------ loading

_REGISTRY = {"orderparam": (OrderParam, "classical_fit.npz"),
             "cnn": (ShotCNN, "cnn_fit.npz"),
             "shadows": (ShadowsLR, "shadows_fit.npz")}


def load(which="orderparam", path=None):
    if which not in _REGISTRY:
        raise ValueError(f"unknown baseline {which!r}; available: {sorted(_REGISTRY)}")
    cls, fname = _REGISTRY[which]
    p = pathlib.Path(path) if path else _MODELS / fname
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Keep models/ next to this file.")
    return cls(np.load(p, allow_pickle=False))


def classify(oracle, state_ids, which="orderparam", copies=None):
    """Competition-shaped entry point."""
    return load(which).run(oracle, state_ids, copies)


if __name__ == "__main__":
    for w in ("orderparam", "cnn", "shadows"):
        try:
            print(f"{w:11s} {load(w)}")
        except Exception as exc:
            print(f"{w:11s} FAILED: {exc}")
