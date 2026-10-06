"""Training for the XXZ task (16 qubits, three phases)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from qphase64 import xxz as X


@dataclass
class TrainResult:
    theta: np.ndarray
    likelihood: object
    history: list = field(default_factory=list)
    best_val: float = 0.0
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def _prob_fn(model, n: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(states, theta):
        qml.StatePrep(states, wires=range(n))
        model.apply(theta)
        return qml.probs(wires=range(n))

    return qnode


def _symbol_matrix(features, out_wires, n) -> torch.Tensor:
    """(2**n, n_symbols) indicator matrix mapping basis strings to symbols."""
    tab = X.basis_tables(n)
    if out_wires is not None:
        bits = tab["bits"][:, list(out_wires)].astype(int)
        w = 2 ** np.arange(len(out_wires) - 1, -1, -1)
        sym = bits @ w
        n_sym = 2 ** len(out_wires)
    else:
        _, sym = X.symbol_grid(features, n)
        n_sym = int(sym.max()) + 1
    m = np.zeros((2 ** n, n_sym))
    m[np.arange(2 ** n), sym] = 1.0
    return torch.tensor(m, dtype=torch.float64)


def _noise_matrix(q: float, n: int) -> torch.Tensor:
    return torch.tensor([[1 - q, q], [q, 1 - q]], dtype=torch.float64)


def _apply_noise(probs: torch.Tensor, nm: torch.Tensor, n: int) -> torch.Tensor:
    """Differentiable bit-flip channel on a batch of distributions."""
    t = probs.reshape((-1,) + (2,) * n)
    for ax in range(1, n + 1):
        t = torch.movedim(torch.tensordot(nm, torch.movedim(t, ax, 0), dims=([1], [0])),
                          0, ax)
    return t.reshape(probs.shape[0], -1)


def chernoff_loss(sym_probs: torch.Tensor, y: torch.Tensor, n_classes: int = 3,
                  n_grid: int = 21) -> torch.Tensor:
    """Negative worst-pair Chernoff exponent between per-phase symbol distributions."""
    eps = 1e-12
    rows = []
    for c in range(n_classes):
        m = y == c
        if not bool(m.any()):
            continue
        p = sym_probs[m].mean(0)
        rows.append(p / p.sum())
    s = torch.linspace(0.0, 1.0, n_grid, dtype=sym_probs.dtype)
    worst = None
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            pa = torch.clamp(rows[a], min=eps)
            pb = torch.clamp(rows[b], min=eps)
            terms = torch.exp(s[:, None] * torch.log(pa)[None, :]
                              + (1 - s)[:, None] * torch.log(pb)[None, :]).sum(-1)
            xi = -torch.log(torch.clamp(terms.min(), min=eps))
            worst = xi if worst is None else torch.minimum(worst, xi)
    return -worst


def bce_loss(sym_probs: torch.Tensor, y: torch.Tensor, n_classes: int = 3):
    """Cross-entropy on the mean symbol distribution -- the mean-separation arm."""
    rows = []
    for c in range(n_classes):
        m = y == c
        p = sym_probs[m].mean(0)
        rows.append(p / p.sum())
    tab = torch.stack(rows)
    ll = torch.log(torch.clamp(tab, min=1e-12))
    return -(ll[y] * sym_probs).sum(-1).mean()


def kshot_loss(sym_probs: torch.Tensor, y: torch.Tensor, n_classes: int = 3,
               k: int = 8) -> torch.Tensor:
    """Accuracy at a *chosen* copy count, rather than the asymptotic decay rate."""
    eps = 1e-12
    rows = []
    for c in range(n_classes):
        m = y == c
        if not bool(m.any()):
            continue
        p = sym_probs[m].mean(0)
        rows.append(p / p.sum())

    worst = None
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            pa = torch.clamp(rows[a], min=eps)
            pb = torch.clamp(rows[b], min=eps)
            llr = torch.log(pa) - torch.log(pb)
            mu = (pa * llr).sum()
            var = torch.clamp((pa * llr ** 2).sum() - mu ** 2, min=eps)
            z = torch.sqrt(torch.tensor(float(k))) * mu / torch.sqrt(var)
            acc = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
            worst = acc if worst is None else torch.minimum(worst, acc)
    return -torch.log(torch.clamp(worst, min=eps))


LOSSES = {"C2": chernoff_loss, "C1": bce_loss, "C3": kshot_loss}


def train(model, data, *, noise_p: float = 0.02, steps: int = 300, batch: int = 6,
          lr: float = 0.05, lr_final: float = 0.005, seed: int = 0,
          loss_name: str = "C2", features=X.FEATURES, out_wires=None,
          eval_every: int = 50, verbose: bool = True,
          n: int = X.N_QUBITS) -> TrainResult:
    """Optimise the circuit so single copies separate the three phases."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_start = time.time()

    prob_fn = _prob_fn(model, n)
    symmat = _symbol_matrix(features, out_wires, n)
    nm = _noise_matrix(X.flip_rate(noise_p), n)

    psi = torch.tensor(np.asarray(data.psi, dtype=np.complex128))
    y_all = torch.tensor(data.y(), dtype=torch.long)

    theta = torch.tensor(model.init_params(seed), dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1),
                                                       eta_min=lr_final)
    lossf = LOSSES[loss_name]
    best = (theta.detach().numpy().copy(), -np.inf)
    result = TrainResult(theta=best[0], likelihood=None)

    for step in range(steps):
        pick = []
        per = max(1, batch // len(X.PHASES))
        for c in range(len(X.PHASES)):
            members = np.flatnonzero(data.y() == c)
            pick.extend(rng.choice(members, size=min(per, len(members)),
                                   replace=False))
        pick = np.array(pick)

        opt.zero_grad()
        probs = prob_fn(psi[pick], theta)
        probs = _apply_noise(probs, nm, n)
        loss = lossf(probs @ symmat, y_all[pick])
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            score = -float(loss.item())
            result.history.append({"step": step, "loss": float(loss.item())})
            if score > best[1]:
                best = (theta.detach().numpy().copy(), score)
            if verbose:
                print(f"    step {step:4d}  loss {loss.item():+.4f}  "
                      f"({time.time()-t_start:.0f}s)")

    result.theta = best[0]
    result.likelihood = fit_at(model, data, best[0], noise_p, features, out_wires, n)
    result.seconds = time.time() - t_start
    result.best_val = best[1]
    return result


def fit_at(model, data, theta, noise_p, features=X.FEATURES, out_wires=None,
           n: int = X.N_QUBITS):
    """Build the phase likelihood at a given theta, from training states only."""
    probs = X.circuit_probs(data.psi, model, theta, n)
    y = data.y()
    class_probs = np.stack([probs[y == c].mean(0) for c in range(len(X.PHASES))])
    class_probs /= class_probs.sum(1, keepdims=True)
    return X.fit_likelihood(class_probs, noise_p, y, features=features,
                            out_wires=out_wires)
