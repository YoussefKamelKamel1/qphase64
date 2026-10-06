"""
Starter kit for 'The 64-Copy Problem'.

What this file shows:
  1. loading the public training set
  2. what the arrays contain
  3. how to talk to the copy-metered oracle (API demo ONLY - the example
     'protocol' below is deliberately useless and is not a strategy hint)
  4. the exact submission interface your team must implement

Runs on free Google Colab. Dependencies: numpy, matplotlib, pennylane.
"""
import numpy as np
import matplotlib.pyplot as plt

from oracle import CopyOracle

# ---------------------------------------------------------------- 1. load data
data = np.load("xxz_public_train.npz", allow_pickle=True)
states = data["states"]        # (48, 65536) float32 - ground state vectors
ids = [str(x) for x in data["ids"]]      # "T00".."T47"
deltas = data["deltas"]        # XXZ anisotropy of each state
labels = [str(x) for x in data["labels"]]  # "FM" / "XY" / "NEEL"
print(str(data["model"]))
print(str(data["convention"]))
print(f"{len(ids)} training states, {int(data['num_qubits'])} qubits each")

# ---------------------------------------------------------------- 2. a look at it
# Amplitude structure of three examples (one per phase). What you compute
# from these vectors, and how, is entirely up to you.
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, k in zip(axes, [0, 24, 47]):
    ax.plot(np.abs(states[k]) ** 2)
    ax.set_title(f"delta={deltas[k]:.2f}  label={labels[k]}")
    ax.set_xlabel("basis index")
    ax.set_ylabel("|amplitude|^2")
plt.tight_layout()
plt.savefig("train_examples.png", dpi=120)
print("wrote train_examples.png")

# ---------------------------------------------------------------- 3. oracle API demo
# Practice oracle wraps the PUBLIC states, so you can develop and self-score.
# The final evaluation uses the SAME class on the hidden set - identical API.
oracle = CopyOracle("xxz_public_train.npz", copy_budget=64, noise_p=0.02, seed=0)

bits = oracle.measure("T00")          # one copy consumed, computational-basis sample
print("one sample of T00:", bits, "| copies left:", oracle.remaining("T00"))


def example_ops(wires):
    """Your gates go in a function like this (applied to each fresh copy
    before measurement). This one does nothing - API demo only."""
    pass


bits = oracle.measure("T00", ops_fn=example_ops)
print("another sample:", bits, "| copies left:", oracle.remaining("T00"))

# ---------------------------------------------------------------- 4. submission interface
# Your repository must contain a file `submission.py` defining exactly:
#
#     def classify(oracle, state_ids):
#         """Return {state_id: label} with label in
#         {"FM", "XY", "NEEL", "UNKNOWN"} for every id in state_ids.
#         `oracle` is a CopyOracle over states you have never seen,
#         with a budget of 64 copies per state. Exceeding the budget
#         on a state forfeits that state."""
#
# The random guesser below is the minimal valid submission - it consumes one
# copy per state and then guesses. It exists only to show the contract.


def classify(oracle, state_ids):
    rng = np.random.default_rng(0)
    out = {}
    for sid in state_ids:
        _ = oracle.measure(sid)  # you have 64 of these per state - spend wisely
        out[sid] = rng.choice(["FM", "XY", "NEEL", "UNKNOWN"])
    return out


if __name__ == "__main__":
    practice = CopyOracle("xxz_public_train.npz", copy_budget=64, noise_p=0.02, seed=7)
    preds = classify(practice, practice.state_ids())
    acc = np.mean([preds[sid] == lab for sid, lab in zip(ids, labels)])
    print(f"random-guess practice accuracy: {acc:.2f}  (your job: beat this, a lot)")
