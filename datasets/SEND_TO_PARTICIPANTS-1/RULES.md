# The 64-Copy Problem — Competition Rules

## 1. What you receive and what you don't

- You receive: the public training set (`xxz_public_train.npz`, fully labeled),
  the oracle (`oracle.py`), the starter kit, and the **exact scoring script**
  (`evaluate.py`) used for final evaluation.
- You do NOT receive the hidden test states or their answer key, in any form.
  Hidden states are accessed only when the organizers run your submission on
  their machine after the deadline.

## 2. The copy budget

- 64 copies per hidden state, enforced by the oracle. One `measure(...)` call
  consumes one copy. Exceeding the budget forfeits that state.
- Adaptive protocols are allowed: what you do with copy *k* may depend on
  copies 1..k−1.
- Your report must account for how your protocol spends its copies.

## 3. Submissions and checkpoints

- Your repository must contain `submission.py` defining
  `classify(oracle, state_ids)` exactly as specified in the README. Everything
  your protocol needs (including pre-trained model files) must live in the repo
  and load inside that call.
- Practice and self-score locally: `python self_score.py submission.py` runs
  the real scoring pipeline against the public training set.
- Checkpoint evaluations: at the announced checkpoint times, organizers pull
  your repository, run it against the hidden set, and return your scores.
  Between checkpoints, nobody sees the hidden set — including you.
- Frameworks: any (PennyLane, Qiskit, ...), as long as `submission.py` speaks
  the oracle's PennyLane-callable interface at the boundary.

## 4. Fair play

- Do not attempt to extract, reconstruct, fingerprint, or brute-force hidden
  states or the answer key beyond your 64 metered copies per state. Protocol
  cleverness inside the budget is the whole game and is encouraged; attacking
  the harness or the commitment files is disqualification.
- All copies you burn count, including debugging runs during checkpoint
  evaluations. There are no refunds.
- Classical baselines in your report must obey the same 64-copy budget as your
  main protocol — that is what makes your comparison meaningful.

## 5. Integrity commitment (why there are encrypted files in your kit)

The `commitment/` folder contains:

- `answer_key.csv.enc` — the real hidden-set answer key, encrypted with
  AES-256. You cannot open it, and that is the point.
- `MANIFEST.sha256` — cryptographic hashes of the hidden test file and the
  encrypted key, fixed before the competition starts.

At the closing ceremony, organizers reveal the decryption password. You can
then decrypt the key yourself:

    openssl enc -d -aes-256-cbc -pbkdf2 -in commitment/answer_key.csv.enc \
        -out answer_key.csv

and verify the hashes match the manifest. This proves the ground truth was
fixed before anyone submitted and was never altered afterward. Until the
reveal, do not waste your hackathon trying to crack AES-256 — spend the time
on the actual problem.

## 6. Scoring

Exactly what `evaluate.py` prints, weighted per the track document: macro-F1
over the three phases (headline), impostor detection precision/recall,
per-tier breakdown, copy usage, plus the judged report deliverables
(protocol justification, budget-matched baselines, robustness analysis).
Identical `noise_p` and random seed are used for every team.
