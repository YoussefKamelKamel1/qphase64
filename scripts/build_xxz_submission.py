"""Train the submission model, save it, and validate it end to end."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "datasets" / "SEND_TO_PARTICIPANTS-1"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from qphase64 import xxz as X          # noqa: E402
from qphase64 import xxz_train as T    # noqa: E402
from qphase64.models.brickwork import Brickwork   # noqa: E402

MODELS = REPO / "models"
KS = (1, 2, 4, 8, 16, 32, 64)


def curve(model, theta, lik, data, noise_p, seed=0, budget=64):
    rng = np.random.default_rng(seed)
    shots = X.sample_batch(data.psi, model, theta, budget, noise_p, rng)
    return {k: float(np.mean([X.PHASES[lik.predict(s[:k])] for s in shots]
                             == data.label)) for k in KS}


def build_mixed_npz(near, imp, path):
    """A hidden-set stand-in: near-boundary XXZ plus Fermi-Hubbard impostors."""
    states, ids, deltas, labels, families, tiers = [], [], [], [], [], []
    for i, (psi, d, lab) in enumerate(zip(near.psi, near.delta, near.label)):
        states.append(np.real(psi).astype(np.float32))
        ids.append(f"N{i:03d}")
        deltas.append(float(d))
        labels.append(str(lab))
        families.append("XXZ")
        tiers.append("near_boundary")
    for i, psi in enumerate(imp):
        states.append(np.real(psi).astype(np.float32))
        ids.append(f"I{i:03d}")
        deltas.append(0.0)
        labels.append("UNKNOWN")
        families.append("FERMIHUBBARD")
        tiers.append("impostor")

    np.savez_compressed(path, states=np.stack(states), ids=np.array(ids),
                        deltas=np.array(deltas), labels=np.array(labels),
                        num_qubits=X.N_QUBITS,
                        model="mixed validation set",
                        convention="near-boundary XXZ + Fermi-Hubbard impostors")
    key = path.with_suffix(".key.csv")
    with key.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "family", "label", "tier",
                                          "delta", "U"])
        w.writeheader()
        for sid, fam, lab, tier, d in zip(ids, families, labels, tiers, deltas):
            w.writerow(dict(id=sid, family=fam, label=lab, tier=tier,
                            delta=d, U=""))
    return path, key


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--noise-p", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-loss-study", action="store_true")
    args = p.parse_args(argv)

    train = X.load_public()
    near = X.near_boundary(n_per_side=24, width=0.28)
    print(f"train {len(train)} | near-boundary {len(near)}")

    # ------------------------------------------------ 1. objective comparison
    if not args.skip_loss_study:
        print("\n=== training objective: asymptotic rate vs fixed-k ===")
        rows = {}
        for loss_name in ("C2", "C3"):
            model = Brickwork(n_qubits=X.N_QUBITS, layers=2, kernel="z2",
                              single="rx", periodic=True)
            res = T.train(model, train, noise_p=args.noise_p, steps=args.steps,
                          seed=args.seed, loss_name=loss_name,
                          eval_every=args.steps, verbose=False)
            rows[loss_name] = curve(model, res.theta, res.likelihood, near,
                                    args.noise_p)
            print(f"  {loss_name}: " + " ".join(f"k={k}:{rows[loss_name][k]:.3f}"
                                                for k in (1, 4, 8, 16, 64)))
        (REPO / "results" / "xxz").mkdir(parents=True, exist_ok=True)
        (REPO / "results" / "xxz" / "loss_study.json").write_text(
            json.dumps(rows, indent=1), encoding="utf-8")

    # ------------------------------------------------ 2. train and save final
    print("\n=== final model ===")
    kwargs = dict(layers=2, kernel="z2", single="rx", periodic=True)
    model = Brickwork(n_qubits=X.N_QUBITS, **kwargs)
    res = T.train(model, train, noise_p=args.noise_p, steps=args.steps,
                  seed=args.seed, loss_name="C2",
                  eval_every=max(args.steps // 3, 1), verbose=True)
    c_near = curve(model, res.theta, res.likelihood, near, args.noise_p)
    c_train = curve(model, res.theta, res.likelihood, train, args.noise_p)
    print(f"  near-boundary: " + " ".join(f"k={k}:{c_near[k]:.3f}"
                                          for k in (4, 16, 64)))

    path = MODELS / "xxz_brickwork.npz"
    X.save_model(path, "brickwork", kwargs, res.theta, res.likelihood,
                 args.noise_p, extra={"steps": args.steps, "seed": args.seed,
                                      "near_boundary_curve": c_near,
                                      "train_curve": c_train})
    print(f"  saved {path.relative_to(REPO)}")

    m2, th2, lik2, meta = X.load_model(path)
    reload_ok = curve(m2, th2, lik2, near, args.noise_p)[16]
    print(f"  reload check: near-boundary k=16 {reload_ok:.3f} "
          f"(trained {c_near[16]:.3f})")

    # ------------------------------------------------ 3. mixed set end to end
    print("\n=== mixed set through the oracle ===")
    imp = X.impostors(n_states=25)
    mixed = REPO / "results" / "xxz" / "mixed_validation.npz"
    npz, key = build_mixed_npz(near, imp, mixed)
    print(f"  {len(near)} near-boundary XXZ + {len(imp)} impostors -> "
          f"{npz.relative_to(REPO)}")

    out = subprocess.run(
        [sys.executable, str(KIT / "evaluate.py"), str(REPO / "submission.py"),
         "--data", str(npz), "--key", str(key),
         "--noise_p", str(args.noise_p), "--budget", "64"],
        cwd=str(KIT), capture_output=True, text=True)
    print(out.stdout[-2200:])
    if out.returncode != 0:
        print("scorer stderr:", out.stderr[-800:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
