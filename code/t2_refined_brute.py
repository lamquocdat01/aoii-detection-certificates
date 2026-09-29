#!/usr/bin/env python3
"""Does the refined bound n <= N*K + d - 1 (d <= K) also hold for NON-periodic samplers
with gaps <= K? Random search; records the worst counterexample.
Output: results/t2_refined_brute.json"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from spike_lagged import runs, transitions
rng = np.random.default_rng(42)
T = 60
worst = dict(ratio=0.0)
nviol = ntr = 0
for it in range(60000):
    K = int(rng.integers(2, 7)); d = int(rng.integers(1, K + 1))
    g = (np.cumsum(rng.random(T) < rng.uniform(0.05, 0.6)) % 2).astype(np.int8)
    gaps = rng.integers(1, K + 1, T); first = int(rng.integers(0, K))
    idx = first + np.concatenate(([0], np.cumsum(gaps))); s = np.zeros(T, bool); s[idx[idx < T]] = True
    st, L = runs(C.belief(s, g, d) != g)
    ctr = np.concatenate(([0], np.cumsum(transitions(g))))
    N = ctr[st + L] - ctr[np.maximum(st - d + 1, 0)]
    b = np.where(N == 0, K - 1, N * K + d - 1)
    ntr += 1
    if (L > b).any():
        nviol += 1
        j = int(np.argmax(L / b))
        r = float(L[j] / b[j])
        if r > worst["ratio"]:
            worst = dict(ratio=r, K=K, d=d, n=int(L[j]), N=int(N[j]), bound=int(b[j]),
                         g=g.tolist(), s=s.astype(int).tolist(), start=int(st[j]))
res = dict(trials=ntr, trials_with_violation=nviol, worst=worst)
C.jdump(res, HERE.parent / "results" / "t2_refined_brute.json")
print(ntr, nviol, {k: v for k, v in worst.items() if k not in ("g", "s")})
