"""
Copy-metered oracle for 'The 64-Copy Problem'.

Each call to `measure(...)` prepares ONE fresh copy of the requested state,
optionally corrupted by the noise channel, runs YOUR operations on it, and
returns a single measurement sample. The copy is then gone. You have 64
copies per state, total, across everything you do. Exceeding the budget
raises CopyBudgetExceeded.

Practice usage (with the public training set, where labels are known):

    from oracle import CopyOracle
    oracle = CopyOracle("xxz_public_train.npz", copy_budget=64, noise_p=0.02, seed=1)
    bits = oracle.measure("T00")                    # computational-basis sample
    bits = oracle.measure("T00", ops_fn=my_ops)     # your gates run before measurement
    oracle.remaining("T00")                         # copies left for this state

`ops_fn(wires)` is a callable that applies PennyLane operations (no
measurements) on the 16 wires; `wires` is range(16), chain site i = wire i.
Final evaluation uses this same class pointed at the hidden test set, with
noise parameters set by the organizers. Interface is identical.
"""
import warnings

import numpy as np

try:
    import pennylane as qml
except ImportError as e:
    raise ImportError("oracle.py requires PennyLane: pip install pennylane") from e

warnings.filterwarnings(
    "ignore", message=".*Setting shots on device.*", category=UserWarning
)


class CopyBudgetExceeded(RuntimeError):
    pass


class CopyOracle:
    def __init__(self, npz_path, copy_budget=64, noise_p=0.0, seed=None,
                 unlimited=False):
        """
        npz_path    : dataset file (public train for practice, hidden for eval)
        copy_budget : copies available per state id
        noise_p     : per-qubit depolarizing probability applied to every copy
        unlimited   : practice mode only - disables the budget (final eval: False)
        """
        data = np.load(npz_path, allow_pickle=True)
        self.n = int(data["num_qubits"])
        self.ids = [str(x) for x in data["ids"]]
        self._states = {sid: data["states"][k].astype(np.complex128)
                        for k, sid in enumerate(self.ids)}
        for sid in self.ids:
            v = self._states[sid]
            self._states[sid] = v / np.linalg.norm(v)
        self.copy_budget = int(copy_budget)
        self.noise_p = float(noise_p)
        self.unlimited = bool(unlimited)
        self._remaining = {sid: self.copy_budget for sid in self.ids}
        self._rng = np.random.default_rng(seed)
        self._dev = qml.device("default.qubit", wires=self.n, shots=1)

    def state_ids(self):
        return list(self.ids)

    def remaining(self, state_id):
        return self._remaining[state_id]

    def usage_report(self):
        return {sid: self.copy_budget - r for sid, r in self._remaining.items()}

    def measure(self, state_id, ops_fn=None, wires=None):
        """Consume one copy; return one sample (0/1 array) over `wires`
        (default: all 16, computational basis unless ops_fn rotates it)."""
        if state_id not in self._states:
            raise KeyError(f"unknown state id {state_id}")
        if not self.unlimited:
            if self._remaining[state_id] <= 0:
                raise CopyBudgetExceeded(
                    f"copy budget ({self.copy_budget}) exhausted for {state_id}")
            self._remaining[state_id] -= 1

        vec = self._states[state_id]
        noise_ops = []
        if self.noise_p > 0:
            hit = self._rng.random(self.n) < self.noise_p
            paulis = self._rng.integers(0, 3, size=self.n)
            for w in range(self.n):
                if hit[w]:
                    noise_ops.append((w, int(paulis[w])))
        meas_wires = list(range(self.n)) if wires is None else list(wires)

        @qml.qnode(self._dev)
        def _circ():
            qml.StatePrep(vec, wires=range(self.n))
            for (w, p) in noise_ops:
                (qml.PauliX, qml.PauliY, qml.PauliZ)[p](wires=w)
            if ops_fn is not None:
                ops_fn(wires=range(self.n))
            return qml.sample(wires=meas_wires)

        out = np.asarray(_circ())
        return out.reshape(-1)
