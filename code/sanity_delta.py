#!/usr/bin/env python3
"""Sanity Monte-Carlo (>= 1e4 random sources each) for Lemma L and Prop 3.1.  seed = 42.
 S1  Lemma L: any sampler with gaps <= K, first invocation <= K-1, any d:
     sup AoII_lag <= K-1, and AoII_lag(t; d) == AoII(t-d; 0).
 S2  Prop 3.1 (<=): HIST tables with every entry <= A+1 never violate sup AoII_lag <= A.
 S3  Prop 3.1 (=>): for every HIST table with a reachable entry K(b,c) >= A+2, the
     adversarial source "X = b for the first c invocations, then 1-b" violates it.
Output: results/sanity_delta.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import optim_delta as O  # noqa: E402
from spike_lagged import lagged_ref  # noqa: E402

rng = np.random.default_rng(C.SEED)


def rand_source(T):
    kind = rng.integers(0, 3)
    if kind == 0:
        L = rng.geometric(rng.uniform(0.02, 0.7), T)
    elif kind == 1:
        L = np.maximum(1, (rng.pareto(rng.uniform(0.8, 2.5), T) + 1).astype(int))
    else:
        L = rng.integers(1, 20, T)
    g = np.repeat(np.arange(len(L)) % 2, L)[:T].astype(np.int8)
    return g if rng.random() < 0.5 else (1 - g).astype(np.int8)


def main():
    out = {}
    # S1
    n = 12000; v = ident_bad = 0
    for _ in range(n):
        T = 300; K = int(rng.integers(1, 17)); d = int(rng.integers(0, 9))
        g = rand_source(T)
        gaps = rng.integers(1, K + 1, T); first = int(rng.integers(0, K))
        idx = first + np.concatenate(([0], np.cumsum(gaps)))
        s = np.zeros(T, bool); s[idx[idx < T]] = True
        Al = C.aoii_trace(C.belief(s, g, d), lagged_ref(g, d))
        A0 = C.aoii_trace(C.belief(s, g, 0), g)
        v += int(Al.max() > K - 1)
        ident_bad += int(not (np.array_equal(Al[d:], A0[:T - d]) and not Al[:d].any()))
    out["S1_lemmaL"] = dict(n_sources=n, violations=v, identity_mismatches=ident_bad)
    print(out["S1_lemmaL"])
    # S2 / S3
    n = 12000; v2 = 0; n3 = v3_ok = 0
    for _ in range(n):
        A = int(rng.integers(1, 20)); d = int(rng.integers(0, 4)); T = 400
        kmin = max(1, d)
        if A + 1 < kmin:
            continue
        tab = rng.integers(kmin, A + 2, size=(2, O.CH))
        phi = int(rng.integers(0, A + 1))
        g = rand_source(T)
        s = O.schedule_hist(g, tab.astype(np.int64), O.CH, phi)
        r = O.evaluate(g, s, d, 7)
        v2 += int(r[4] > A)
        # S3: raise one entry above the cap and play the adversary
        b = int(rng.integers(0, 2)); c = int(rng.integers(1, O.CH + 1))
        tab3 = tab.copy(); tab3[b, c - 1] = int(rng.integers(A + 2, A + 12))
        # adversary: X = b through the c-th invocation (entries for state (b, c') with
        # c' < c keep their values), then flip to 1-b right after it
        g3 = np.full(T, b, np.int8)
        t = phi; cnt = 0; flip_at = None
        while t < T:
            cnt += 1
            cc = min(cnt, O.CH)
            if cc == c and (c < O.CH or cnt == c):
                flip_at = t + 1; break
            t += tab3[b, cc - 1]
        if flip_at is None or flip_at + tab3[b, c - 1] + d >= T:
            continue
        g3[flip_at:] = 1 - b
        s3 = O.schedule_hist(g3, tab3.astype(np.int64), O.CH, phi)
        r3 = O.evaluate(g3, s3, d, 7)
        n3 += 1; v3_ok += int(r3[4] > A)
    out["S2_cap_implies_cert"] = dict(n_sources=n, violations=v2)
    out["S3_gap_gt_cap_breaks_cert"] = dict(n_cases=n3, adversary_succeeds=v3_ok)
    print(out["S2_cap_implies_cert"], out["S3_gap_gt_cap_breaks_cert"])
    C.jdump(out, HERE.parent / "results" / "sanity_delta.json")


if __name__ == "__main__":
    main()
