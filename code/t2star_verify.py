#!/usr/bin/env python3
"""CP-delta1 / P1: machine checks for Theorem 2 (T2*) and its corollaries.  seed = 42.

Theorem 2 (periodic K, 1 <= d <= K):  every maximal mismatch run [s,t] of the
symmetric AoII contains N >= 1 source transitions in (s-d, t] and has length
    n <= N*K + d - 1,                                  attained for every N >= 1.
(d = 0: n <= K-1, Lemma 1.)  Suffix form used by Cor 2.2: every suffix [a,t] of a
run satisfies n <= (N'+1) K + d - 1, N' = transitions in (a, t].
Cor 2.1: if every sojourn of X is >= K+d, every run has N <= 1, so n <= K+d-1.

Checks
  E1  exhaustive: every binary source of length L = 3K, every phase, K <= 6, d <= K
      (plus d = K+1, K+2 to exhibit the failure beyond d = K)
  E2  attainment construction for K <= 12, 1 <= d <= K, N <= 6
  E3  Monte-Carlo sanity: 2e4 random sources (iid flips, heavy-tail, adversarial)
  E4  Cor 2.1 under min-sojourn K+d (1e4 sources) and a violation at K+d-1
  E5  suffix form (Cor 2.2) on every suffix of every run in E3
Output: results/t2star_verify.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

rng = np.random.default_rng(C.SEED)


# ----------------------------------------------------------------- kernels
@njit(cache=True)
def belief_periodic(g, K, phi, d):
    """Padding convention: X_u = 0 and virtual samples for u < 0 (belief 0)."""
    T = g.shape[0]
    xh = np.zeros(T, np.int8)
    for t in range(T):
        u = t - d                       # latest admissible sample slot
        if u < phi:
            xh[t] = 0
        else:
            p = phi + ((u - phi) // K) * K
            xh[t] = g[p]
    return xh


@njit(cache=True)
def runs_check(g, xh, K, d, maxN, out_max, out_viol, suffix_viol):
    """For every maximal mismatch run: N = #transitions of X in (s-d, t],
    update out_max[N] = max run length; count violations of N*K+d-1 (N>=1) /
    K-1 (N=0).  Also checks the suffix bound on every suffix of the run."""
    T = g.shape[0]
    tr = np.zeros(T, np.int64)          # tr[v] = 1 iff X_v != X_{v-1}, X_{-1} = 0
    prev = 0
    for v in range(T):
        if g[v] != prev:
            tr[v] = 1
        prev = g[v]
    cum = np.zeros(T + 1, np.int64)
    for v in range(T):
        cum[v + 1] = cum[v] + tr[v]
    t = 0
    nv = 0
    while t < T:
        if xh[t] != g[t]:
            s = t
            while t < T and xh[t] != g[t]:
                t += 1
            e = t - 1                   # run [s, e]
            n = e - s + 1
            lo = s - d + 1
            if lo < 0:
                lo = 0
            N = cum[e + 1] - cum[lo]
            b = K - 1 if N == 0 else N * K + d - 1
            if n > b:
                nv += 1
            NN = N if N < maxN else maxN - 1
            if n > out_max[NN]:
                out_max[NN] = n
            # suffix form: every suffix [a, e]
            for a in range(s, e + 1):
                Np = cum[e + 1] - cum[a + 1]   # transitions in (a, e]
                if (e - a + 1) > (Np + 1) * K + d - 1:
                    suffix_viol[0] += 1
        else:
            t += 1
    out_viol[0] += nv


@njit(cache=True)
def exhaustive(K, d, L, maxN):
    out_max = np.zeros(maxN, np.int64)
    viol = np.zeros(1, np.int64)
    sv = np.zeros(1, np.int64)
    g = np.zeros(L, np.int8)
    for code in range(1 << L):
        for i in range(L):
            g[i] = (code >> i) & 1
        for phi in range(K):
            xh = belief_periodic(g, K, phi, d)
            runs_check(g, xh, K, d, maxN, out_max, viol, sv)
    return out_max, viol[0], sv[0]


def one(g, K, phi, d, maxN=64):
    om = np.zeros(maxN, np.int64); v = np.zeros(1, np.int64); sv = np.zeros(1, np.int64)
    runs_check(g, belief_periodic(g, K, phi, d), K, d, maxN, om, v, sv)
    return om, int(v[0]), int(sv[0])


# ------------------------------------------------------------- E2 construction
def construction(K, d, N, tail=None):
    """Samples at 0, K, 2K, ... (phase 0).  X = 0 up to slot K, flips to 1 at K+1,
    then flips at jK+d for j = 2..N; afterwards constant.  Claimed run:
    [K+1, (N+1)K+d-1], length N K + d - 1, with exactly N transitions in (s-d, t]."""
    T = (N + 3) * K + d + 5
    g = np.zeros(T, np.int8)
    val = 1
    flips = [K + 1] + [j * K + d for j in range(2, N + 1)]
    cur = 0
    fi = 0
    for t in range(T):
        if fi < len(flips) and t == flips[fi]:
            cur = 1 - cur
            fi += 1
        g[t] = cur
    xh = belief_periodic(g, K, 0, d)
    m = (xh != g)
    s = K + 1
    e = s
    while e + 1 < T and m[e + 1]:
        e += 1
    n = e - s + 1 if m[s] else 0
    tr = np.concatenate(([g[0] != 0], g[1:] != g[:-1]))
    Nn = int(tr[max(s - d + 1, 0):e + 1].sum())
    return n, Nn


# --------------------------------------------------------------- random sources
def random_source(T, kind, K, d, min_soj=1):
    if kind == 0:
        p = rng.uniform(0.02, 0.7)
        L = rng.geometric(p, T)
    elif kind == 1:
        L = np.maximum(1, (rng.pareto(rng.uniform(0.8, 2.5), T) + 1).astype(int))
    else:                                    # near-adversarial: sojourns around K
        L = rng.integers(1, 2 * K + 2, T)
    L = np.maximum(L, min_soj)
    g = np.repeat(np.arange(len(L)) % 2, L)[:T].astype(np.int8)
    if rng.random() < 0.5:
        g = 1 - g
    return g


def main():
    out = {"seed": C.SEED}
    # ---------------- E1 exhaustive
    e1 = []
    for K in range(1, 7):
        for d in range(0, K + 3):
            # d <= K: horizon 3K as specified; d > K: longer horizon so the failure shows
            L = max(3 * K, 6) if d <= K else min(max(3 * K, 16), 18)
            om, v, sv = exhaustive(K, d, L, 16)
            rows = {int(N): int(om[N]) for N in range(16) if om[N] > 0}
            pred = {N: (K - 1 if N == 0 else N * K + d - 1) for N in rows}
            attained = {N: rows[N] == pred[N] for N in rows}
            e1.append(dict(K=K, d=d, L=L, regime=("d<=K" if d <= K else "d>K"),
                           violations=int(v), suffix_violations=int(sv),
                           max_run_by_N=rows, bound_by_N=pred, attained_by_N=attained))
            print(f"E1 K={K} d={d} L={L} viol={v} suffix_viol={sv} {rows}")
    out["E1_exhaustive"] = e1
    out["E1_summary"] = dict(
        n_cases_d_le_K=sum(1 for r in e1 if r["regime"] == "d<=K"),
        violations_d_le_K=sum(r["violations"] for r in e1 if r["regime"] == "d<=K"),
        suffix_violations_d_le_K=sum(r["suffix_violations"] for r in e1 if r["regime"] == "d<=K"),
        violations_d_gt_K=sum(r["violations"] for r in e1 if r["regime"] == "d>K"),
        cases_d_gt_K_with_violation=[(r["K"], r["d"]) for r in e1
                                     if r["regime"] == "d>K" and r["violations"] > 0])

    # explicit counterexample for d > K (smallest)
    ce = None
    for K in (2, 3):
        d = K + 1
        L = 3 * K + 3
        for code in range(1 << L):
            g = np.array([(code >> i) & 1 for i in range(L)], np.int8)
            for phi in range(K):
                om, v, _ = one(g, K, phi, d)
                if v:
                    ce = dict(K=K, d=d, phase=phi, g=g.tolist(),
                              belief=belief_periodic(g, K, phi, d).tolist())
                    break
            if ce:
                break
        if ce:
            break
    out["E1_counterexample_d_gt_K"] = ce

    # ---------------- E2 attainment
    e2 = []
    for K in range(1, 13):
        for d in range(1, K + 1):
            for N in range(1, 7):
                n, Nn = construction(K, d, N)
                e2.append(dict(K=K, d=d, N=N, n=n, N_counted=Nn,
                               ok=(n == N * K + d - 1 and Nn == N)))
    out["E2_attainment"] = dict(n=len(e2), all_ok=all(r["ok"] for r in e2),
                                failures=[r for r in e2 if not r["ok"]][:10])
    print("E2", out["E2_attainment"]["n"], out["E2_attainment"]["all_ok"])

    # ---------------- E3 Monte-Carlo sanity (>= 1e4 sources)
    n_src = 20000
    viol = sviol = 0
    runs_total = 0
    for i in range(n_src):
        K = int(rng.integers(1, 17)); d = int(rng.integers(0, K + 1))
        g = random_source(400, i % 3, K, d)
        phi = int(rng.integers(0, K))
        om, v, sv = one(g, K, phi, d)
        viol += v; sviol += sv
    out["E3_mc"] = dict(n_sources=n_src, T=400, violations=viol, suffix_violations=sviol)
    print("E3", out["E3_mc"])

    # ---------------- E4 Cor 2.1 (min sojourn K+d => N <= 1, n <= K+d-1)
    bad_hyp = 0
    viol_at_Kd_minus1 = 0
    for i in range(10000):
        K = int(rng.integers(1, 13)); d = int(rng.integers(1, K + 1))
        phi = int(rng.integers(0, K))
        g = random_source(600, i % 3, K, d, min_soj=K + d)
        om, v, _ = one(g, K, phi, d)
        if om[2:].max() > 0 or om[1] > K + d - 1:
            bad_hyp += 1
        g2 = random_source(600, 2, K, d, min_soj=K + d - 1)
        om2, _, _ = one(g2, K, phi, d)
        if om2[2:].max() > 0:
            viol_at_Kd_minus1 += 1
    out["E4_cor21"] = dict(n_sources=10000, runs_with_N_ge_2_or_n_gt_Kd1_under_min_sojourn_K_plus_d=bad_hyp,
                           sources_with_N_ge_2_under_min_sojourn_K_plus_d_minus_1=viol_at_Kd_minus1)
    print("E4", out["E4_cor21"])

    C.jdump(out, HERE.parent / "results" / "t2star_verify.json")


if __name__ == "__main__":
    main()
