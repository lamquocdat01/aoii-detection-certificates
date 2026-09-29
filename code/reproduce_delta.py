#!/usr/bin/env python3
"""Regenerate every number of the paper from the released data, in order, and compare the
result key by key with expected/numbers_delta.json.  seed = 42.

    python code/reproduce_delta.py            # full run (about 1 h on a laptop CPU)
    python code/reproduce_delta.py --from markov_baseline.py   # resume at a step

Exit code 0 only if every macro value matches.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = [
    "aoii_replay.py",          # trace replay of the policy zoo -> results/aoii_scores.json (gate thresholds)
    "energy_frontier.py",      # measured power model  -> results/energy_frontier.json
    "t2star_verify.py",        # Theorem 1: exhaustive, attainment, Monte Carlo, Cor. 1
    "t2_refined_corpus.py",    # Theorem 1 on the workload
    "sanity_delta.py",         # Lemma 1 and Proposition 2 (both directions)
    "optim_delta.py",          # Algorithm 1 on the workload (price of the certificate)
    "holdout_delta.py",        # 60/40 hold-out, semi-Markov model, oracle
    "markov_baseline.py",      # Markov-optimal baseline (raw and budget-corrected)
    "summary_delta.py",        # nested optimum J*(A), Markov c-independence
    "prop34_check.py",         # Lemma S1 and the monotonicity observation
    "cor22_tail.py",           # distributional certificate (Remark 1)
    "figs_delta.py",           # figures and figD1 statistics
    "make_numbers_delta.py",   # every printed number -> results/numbers_delta.json
]


def main():
    # optional: --from <script> resumes a run that stopped at that step
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0]
    for s in STEPS[STEPS.index(start):]:
        t0 = time.time()
        print(f"== {s}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=ROOT)
        if r.returncode:
            sys.exit(f"step {s} failed with code {r.returncode}")
        print(f"   {time.time() - t0:.0f} s", flush=True)
    got = json.load(open(ROOT / "results" / "numbers_delta.json", encoding="utf-8"))
    exp = json.load(open(ROOT / "expected" / "numbers_delta.json", encoding="utf-8"))
    diff = {k: (exp.get(k), got.get(k)) for k in sorted(set(exp) | set(got)) if exp.get(k) != got.get(k)}
    json.dump(dict(n_expected=len(exp), n_got=len(got), mismatches=diff),
              open(ROOT / "results" / "reproduce_check.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(exp) - len(diff)}/{len(exp)} macros match")
    for k, v in diff.items():
        print("MISMATCH", k, v)
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
