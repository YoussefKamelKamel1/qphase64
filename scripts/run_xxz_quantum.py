"""Quantum models on the organizer XXZ task: training, evaluation, ablations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "datasets" / "SEND_TO_PARTICIPANTS-1"))

from qphase64 import xxz as X          # noqa: E402
from qphase64 import xxz_train as T    # noqa: E402
from qphase64.models.brickwork import Brickwork   # noqa: E402
from qphase64.models.qcnn import SharedQCNN       # noqa: E402

OUT = REPO / "results" / "xxz"
KS = (1, 2, 4, 8, 16, 32, 64)
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)


MODEL_KWARGS = {
    "M1": dict(layers=2, kernel="z2", single="rx", periodic=True),
    "M2": dict(kernel="z2", pool="z2pool", n_out=2, periodic=True),
}


def m1(**kw):
    base = dict(layers=2, kernel="z2", single="rx", periodic=True)
    base.update(kw)
    return Brickwork(n_qubits=X.N_QUBITS, **base), None


def m2(**kw):
    base = dict(kernel="z2", pool="z2pool", n_out=2, periodic=True)
    base.update(kw)
    model = SharedQCNN(n_qubits=X.N_QUBITS, **base)
    return model, tuple(model.output_wires())


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float),
                                      encoding="utf-8")
    print(f"    wrote results/xxz/{name}.json")


def accuracy_vs_k(psis, truth, model, theta, lik, noise_p, seed, budget=64):
    """Draw the budget once per state, then replay truncated at each k."""
    rng = np.random.default_rng(seed)
    shots = X.sample_batch(psis, model, theta, budget, noise_p, rng)
    out = {}
    for k in KS:
        pred = np.array([X.PHASES[lik.predict(s[:k])] for s in shots])
        out[k] = float(np.mean(pred == truth))
    return out, shots


def oracle_check(model, theta, lik, data, noise_p, seed, budget=64, k_use=32):
    """Headline numbers through the organizer's own metered oracle."""
    from oracle import CopyOracle

    o = CopyOracle(str(X.TRAIN_NPZ), copy_budget=budget, noise_p=noise_p, seed=seed)
    ops = X.make_ops_fn(model, theta)
    preds = []
    for sid in data.ids:
        shots = np.stack([o.measure(sid, ops_fn=ops) for _ in range(k_use)])
        preds.append(X.PHASES[lik.predict(shots)])
    used = list(o.usage_report().values())
    return (float(np.mean(np.array(preds) == data.label)),
            {"mean": float(np.mean(used)), "max": int(np.max(used))})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--ablation-steps", type=int, default=150)
    p.add_argument("--noise-p", type=float, default=0.02)
    p.add_argument("--skip-ablations", action="store_true")
    args = p.parse_args(argv)

    print("loading data ...")
    train_data = X.load_public()
    near = X.near_boundary(n_per_side=24, width=0.28)
    imp = X.impostors(n_states=25)
    print(f"  train {len(train_data)} | near-boundary {len(near)} | impostors {len(imp)}")
    save("datasets", {"train": len(train_data), "near_boundary": len(near),
                      "impostors": len(imp),
                      "near_deltas": near.delta.tolist()})

    trained, headline = {}, {}
    for name, build in (("M1", m1), ("M2", m2)):
        print(f"\n=== {name} ===")
        for seed in args.seeds:
            model, out_wires = build()
            print(f"  seed {seed} (params={model.n_params()}, "
                  f"out_wires={out_wires})")
            res = T.train(model, train_data, noise_p=args.noise_p,
                          steps=args.steps, seed=seed, out_wires=out_wires,
                          eval_every=max(args.steps // 4, 1), verbose=True)
            trained[(name, seed)] = (model, out_wires, res)
            print(f"    trained in {res.seconds:.0f}s")

            ckpt = REPO / "models" / f"xxz_{name.lower()}_seed{seed}.npz"
            X.save_model(ckpt, "brickwork" if name == "M1" else "qcnn",
                         MODEL_KWARGS[name], res.theta, res.likelihood,
                         args.noise_p, extra={"seed": seed, "steps": args.steps})
            print(f"    saved {ckpt.relative_to(REPO)}")

            acc_tr, _ = accuracy_vs_k(train_data.psi, train_data.label, model,
                                      res.theta, res.likelihood, args.noise_p, seed)
            acc_nb, _ = accuracy_vs_k(near.psi, near.label, model, res.theta,
                                      res.likelihood, args.noise_p, seed)
            headline.setdefault(name, []).append(
                {"seed": seed, "params": int(model.n_params()),
                 "train_deltas": acc_tr, "near_boundary": acc_nb,
                 "seconds": res.seconds})
            print(f"    train-deltas  k=16 {acc_tr[16]:.3f} | "
                  f"near-boundary k=16 {acc_nb[16]:.3f}")
            save("headline", headline)

        # Oracle confirmation on one seed
        model, out_wires, res = trained[(name, args.seeds[0])]
        acc, usage = oracle_check(model, res.theta, res.likelihood, train_data,
                                  args.noise_p, args.seeds[0])
        print(f"  ORACLE ({name}): accuracy {acc:.3f}, copies {usage}")
        headline[name + "_oracle"] = {"accuracy": acc, "usage": usage}
        save("headline", headline)

    # ------------------------------------------------------------- noise sweep
    print("\n=== noise sweep ===")
    noise = {}
    for name in ("M1", "M2"):
        model, out_wires, res = trained[(name, args.seeds[0])]
        for aware in (True, False):
            for p_noise in NOISE_LEVELS:
                lik = (res.likelihood if aware else
                       T.fit_at(model, train_data, res.theta, 0.0,
                                out_wires=out_wires))
                a, _ = accuracy_vs_k(near.psi, near.label, model, res.theta,
                                     lik, p_noise, args.seeds[0])
                noise.setdefault(name, {}).setdefault(
                    "noise_aware" if aware else "noise_blind", {})[p_noise] = a
                print(f"  {name} aware={aware} p={p_noise:.2f} k=16 {a[16]:.3f}")
    save("noise", noise)

    # --------------------------------------------------------------- impostors
    print("\n=== impostors ===")
    tab = X.basis_tables()
    imp_out = {}
    for name in ("M1", "M2"):
        model, out_wires, res = trained[(name, args.seeds[0])]
        rng = np.random.default_rng(0)
        # raw computational-basis copies carry the Jordan-Wigner signature
        def gap(psi):
            pr = X.bitflip_channel(np.abs(psi) ** 2, X.flip_rate(args.noise_p))
            d = rng.choice(len(pr), size=32, p=np.clip(pr, 0, None) / pr.sum())
            return float(np.mean(tab["pair"][d] - tab["c1"][d]))
        xxz_gap = np.array([gap(p) for p in train_data.psi])
        nb_gap = np.array([gap(p) for p in near.psi])
        fh_gap = np.array([gap(p) for p in imp])
        thr = float(np.percentile(np.concatenate([xxz_gap, nb_gap]), 1.0)) - 0.02
        imp_out[name] = {
            "threshold": thr,
            "recall": float(np.mean(fh_gap < thr)),
            "false_reject": float(np.mean(np.concatenate([xxz_gap, nb_gap]) < thr)),
            "xxz_gap_mean": float(xxz_gap.mean()),
            "fh_gap_mean": float(fh_gap.mean()),
            "xxz_gaps": xxz_gap.tolist(), "fh_gaps": fh_gap.tolist(),
            "nb_gaps": nb_gap.tolist(),
        }
        print(f"  {name}: recall {imp_out[name]['recall']:.3f}, "
              f"false-reject {imp_out[name]['false_reject']:.3f}")
    save("impostor", imp_out)

    # --------------------------------------------------------------- ablations
    if not args.skip_ablations:
        print("\n=== ablations ===")
        configs = [
            ("M1 depth", "L=1", lambda: m1(layers=1), "C2", X.FEATURES),
            ("M1 depth", "L=2", lambda: m1(layers=2), "C2", X.FEATURES),
            ("M1 depth", "L=3", lambda: m1(layers=3), "C2", X.FEATURES),
            ("M1 kernel", "general (9)", lambda: m1(kernel="general"), "C2", X.FEATURES),
            ("M1 readout", "m_abs only", lambda: m1(), "C2", ("m_abs",)),
            ("M1 readout", "m_abs+stag", lambda: m1(), "C2", X.FEATURES),
            ("M1 readout", "+c1", lambda: m1(), "C2", ("m_abs", "stag", "c1")),
            ("M1 loss", "C1", lambda: m1(), "C1", X.FEATURES),
            ("M1 loss", "C2", lambda: m1(), "C2", X.FEATURES),
            ("M2 n_out", "1", lambda: m2(n_out=1), "C2", X.FEATURES),
            ("M2 n_out", "2", lambda: m2(n_out=2), "C2", X.FEATURES),
            ("M2 tying", "untied", lambda: m2(tied=False), "C2", X.FEATURES),
        ]
        abl = []
        for group, label, build, loss_name, feats in configs:
            model, out_wires = build()
            t0 = time.time()
            try:
                res = T.train(model, train_data, noise_p=args.noise_p,
                              steps=args.ablation_steps, seed=args.seeds[0],
                              loss_name=loss_name, features=feats,
                              out_wires=out_wires,
                              eval_every=args.ablation_steps, verbose=False)
                a_tr, _ = accuracy_vs_k(train_data.psi, train_data.label, model,
                                        res.theta, res.likelihood, args.noise_p, 0)
                a_nb, _ = accuracy_vs_k(near.psi, near.label, model, res.theta,
                                        res.likelihood, args.noise_p, 0)
                flat = len({build()[0].n_params()} |
                           {Brickwork(n, layers=model.layers, periodic=True).n_params()
                            if isinstance(model, Brickwork) else
                            SharedQCNN(n, n_out=model.n_out, tied=model.tied,
                                       periodic=True).n_params()
                            for n in (8, 12, 16)}) == 1
                row = {"group": group, "config": label,
                       "params": int(model.n_params()), "size_independent": bool(flat),
                       "train_k16": a_tr[16], "near_k4": a_nb[4],
                       "near_k16": a_nb[16], "near_k64": a_nb[64],
                       "seconds": round(time.time() - t0, 1)}
            except Exception as exc:
                row = {"group": group, "config": label, "error": str(exc)[:120]}
            abl.append(row)
            print(f"  {group:12s} {label:14s} " +
                  (f"params={row.get('params','-'):>3} near k4/16/64 = "
                   f"{row.get('near_k4',float('nan')):.3f}/"
                   f"{row.get('near_k16',float('nan')):.3f}/"
                   f"{row.get('near_k64',float('nan')):.3f}"
                   if "error" not in row else f"FAILED {row['error']}"))
            save("ablations", abl)

    print("\ndone ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
