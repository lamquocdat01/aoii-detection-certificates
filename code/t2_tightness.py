#!/usr/bin/env python3
"""Exhaustive tightness of the transition-count certificate (T2) for periodic-K:
max mismatch-run length n as a function of N (transitions of X in (s-d, t]),
over ALL binary sources of length T and all phases. Output: results/t2_tightness.json"""
import sys, itertools, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from spike_lagged import runs, transitions
T = 18
out = []
for K in (2, 3, 4):
    for d in (1, 2, 3):
        best = {}
        for bits in itertools.product((0, 1), repeat=T):
            g = np.array(bits, dtype=np.int8)
            ctr = np.concatenate(([0], np.cumsum(transitions(g))))
            for phi in range(K):
                s = C.periodic(T, K, phi)
                xh = C.belief(s, g, d)
                st, L = runs(xh != g)
                for a, n in zip(st, L):
                    if a + n >= T:          # right-censored run: skip
                        continue
                    N = int(ctr[a + n] - ctr[max(a - d + 1, 0)])
                    best[N] = max(best.get(N, 0), int(n))
        rows = {N: dict(max_run=best[N], bound=(K - 1) + N * (K - 1 + d)) for N in sorted(best)}
        out.append(dict(K=K, d=d, T=T, by_N=rows))
        print(K, d, {N: (r["max_run"], r["bound"]) for N, r in rows.items()})
C.jdump(out, HERE.parent / "results" / "t2_tightness.json")
