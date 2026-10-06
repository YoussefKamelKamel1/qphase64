"""
Scoring script for 'The 64-Copy Problem'. Fully transparent: this is the
EXACT script organizers run against the hidden set. You can (and should)
run it yourself against the public training set — see self_score.py.

Usage:
    python evaluate.py submission.py --data <states.npz> --key <answer_key.csv> \
                       [--noise_p 0.02] [--seed 123] [--budget 64]

Your submission.py must define classify(oracle, state_ids) -> {id: label},
label in {"FM", "XY", "NEEL", "UNKNOWN"}.
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import CopyOracle, CopyBudgetExceeded  # noqa: E402

PHASES = ["FM", "XY", "NEEL"]
VALID = set(PHASES) | {"UNKNOWN"}


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


def score(preds, key, usage, budget):
    conf = defaultdict(int)
    tier_hit, tier_tot = defaultdict(int), defaultdict(int)
    for sid, row in key.items():
        truth = row["label"] if row["family"] == "XXZ" else "IMPOSTOR"
        pred = preds.get(sid, "MISSING")
        pred = pred if pred in VALID else "MISSING"
        tier_tot[row["tier"]] += 1
        correct = (pred == row["label"]) if row["family"] == "XXZ" else (pred == "UNKNOWN")
        tier_hit[row["tier"]] += int(correct)
        conf[(truth, pred)] += 1

    print("\n--- per-class metrics (XXZ states only) ---")
    f1s = []
    for c in PHASES:
        tp = conf[(c, c)]
        fp = sum(conf[(t, c)] for t in PHASES + ["IMPOSTOR"] if t != c)
        fn = sum(conf[(c, p)] for p in VALID | {"MISSING"} if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print(f"{c:5s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    print(f"MACRO-F1 (headline metric): {np.mean(f1s):.3f}")

    has_impostors = any(r["family"] != "XXZ" for r in key.values())
    if has_impostors:
        tp = conf[("IMPOSTOR", "UNKNOWN")]
        fp = sum(conf[(c, "UNKNOWN")] for c in PHASES)
        fn = sum(conf[("IMPOSTOR", p)] for p in VALID | {"MISSING"} if p != "UNKNOWN")
        iprec = tp / (tp + fp) if tp + fp else 0.0
        irec = tp / (tp + fn) if tp + fn else 0.0
        print(f"\nimpostor detection: P={iprec:.3f}  R={irec:.3f}")

    print("\n--- accuracy by tier ---")
    for t in sorted(tier_tot):
        print(f"{t:9s} {tier_hit[t]}/{tier_tot[t]} = {tier_hit[t]/tier_tot[t]:.2f}")

    print("\n--- copy usage ---")
    u = np.array(list(usage.values()))
    print(f"mean {u.mean():.1f}, min {u.min()}, max {u.max()} of {budget}")

    print("\n--- confusion (truth -> pred) ---")
    cols = PHASES + ["UNKNOWN", "MISSING"]
    print("truth\\pred " + " ".join(f"{c:>8s}" for c in cols))
    for t in PHASES + (["IMPOSTOR"] if has_impostors else []):
        print(f"{t:10s} " + " ".join(f"{conf[(t, c)]:8d}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--data", required=True, help="states .npz")
    ap.add_argument("--key", required=True, help="answer key .csv")
    ap.add_argument("--noise_p", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--budget", type=int, default=64)
    a = ap.parse_args()

    key = {}
    with open(a.key) as f:
        for row in csv.DictReader(f):
            key[row["id"]] = row

    oracle = CopyOracle(a.data, copy_budget=a.budget, noise_p=a.noise_p, seed=a.seed)
    classify = load_submission(a.submission)
    try:
        preds = classify(oracle, oracle.state_ids())
    except CopyBudgetExceeded as e:
        print(f"FATAL: submission exceeded the budget and crashed: {e}")
        return
    score(preds, key, oracle.usage_report(), a.budget)


if __name__ == "__main__":
    main()
