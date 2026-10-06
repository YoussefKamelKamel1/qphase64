"""
Self-scoring on the PUBLIC training set — the same experience organizers
get on the hidden set. Builds an answer key from the public labels, then
runs evaluate.py on your submission.py.

    python self_score.py [path/to/submission.py]
"""
import csv
import subprocess
import sys

import numpy as np

data = np.load("xxz_public_train.npz", allow_pickle=True)
with open("train_answer_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "family", "label", "tier", "delta", "U"])
    w.writeheader()
    for sid, lab, d in zip(data["ids"], data["labels"], data["deltas"]):
        w.writerow(dict(id=str(sid), family="XXZ", label=str(lab),
                        tier="train", delta=float(d), U=""))

sub = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
subprocess.run([sys.executable, "evaluate.py", sub,
                "--data", "xxz_public_train.npz",
                "--key", "train_answer_key.csv"], check=False)
