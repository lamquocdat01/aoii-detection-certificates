#!/usr/bin/env python3
"""Đợt 1b / Việc 2: Prop 3.4 (narrow) — decreasing hazard => index thresholds K(b,c)
non-decreasing in c.  seed = 42.

Lemma M (proved in theory_delta.md §3.5).  Sojourn law F with DFR (F̄(x+j)/F̄(x)
non-decreasing in x).  The current sojourn started at an unknown slot S in a window
I (between the last differing verdict and the first identical one), with ANY prior
pi on I, and has survived until the last invocation t.  Then
    s(t) := P(no transition in (t, t+j] | S in I, survived to t)
          = E_{w_t}[ F̄(t+j-S) / F̄(t-S) ],   w_t(S) ∝ pi(S) F̄(t-S),
is non-decreasing in t.  One more identical verdict moves t to t' > t, so under the
no-double-flip condition ND (no sojourn shorter than a gap) the posterior mismatch
probability q(j,b,c) = 1 - s is non-increasing in c, and the index threshold
K(b,c) = min{j : q(j,b,c) >= theta} is non-decreasing in c.

Checks
 M1  Lemma M exactly (no Monte Carlo) on 1e4 random (DFR law, prior, t<t', j)
     DFR laws: discretised Pareto (alpha in [0.3,3]) and Weibull (k in [0.3,0.95]);
     negative control: IFR Weibull (k in [1.5,4]) - the inequality should FAIL often
 M2  Without ND: Monte-Carlo estimate of q(j,b,c) on alternating DFR sources under
     periodic sampling (1e4 sources in 200 settings); count settings where q increases
     with c by more than 2 standard errors
 M3  The empirical corpus: q(j,b,c) under periodic K = 8, d = 0 (all phases)
 M4  Exception d = 0, A = 15, abar = 0.25 (K(1,.) = (3,2,7)): coordinate descent from
     3 initialisations
Output: results/prop34_check.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

rng = np.random.default_rng(C.SEED)
XMAX = 4000


def survival(kind, par):
    x = np.arange(XMAX + 1, dtype=float)
    if kind == "pareto":
        Fb = (1 + x) ** (-par)
    else:                                        # weibull, scale 5
        Fb = np.exp(-(x / 5.0) ** par)
    return np.maximum(Fb, 1e-300)


def lemma_value(Fb, S, pi, t, j):
    a = t - S
    w = pi * Fb[a]
    w = w / w.sum()
    return float((w * Fb[a + j] / Fb[a]).sum())


def M1(n=10000):
    res = {}
    for fam in ("DFR", "IFR"):
        viol = 0; margin = []
        for _ in range(n):
            if fam == "DFR":
                if rng.random() < 0.5:
                    Fb = survival("pareto", rng.uniform(0.3, 3.0))
                else:
                    Fb = survival("weibull", rng.uniform(0.3, 0.95))
            else:
                Fb = survival("weibull", rng.uniform(1.5, 4.0))
            L = int(rng.integers(1, 30)); S = np.arange(0, L)       # window I = [0, L)
            pi = rng.random(L) + 1e-3
            t = int(rng.integers(L, L + 60)); tp = t + int(rng.integers(1, 40)); j = int(rng.integers(1, 40))
            v0, v1 = lemma_value(Fb, S, pi, t, j), lemma_value(Fb, S, pi, tp, j)
            margin.append(v1 - v0)
            viol += int(v1 < v0 - 1e-12)
        res[fam] = dict(n=n, violations=viol, min_margin=float(min(margin)))
    return res


def alt_source(T, lawD, lawG):
    n = T // 2 + 10
    L = np.empty(2 * n, int); L[0::2] = lawG(n); L[1::2] = lawD(n)
    return np.repeat(np.arange(2 * n) % 2, L)[:T].astype(np.int8)


def q_table(g, K, CH=3, phases=None):
    """q[b, c-1, j-1] = P(X_{t_c + j} != b | verdict b, c consecutive identical), d = 0."""
    num = np.zeros((2, CH, K)); den = np.zeros((2, CH))
    T = len(g)
    for phi in (range(K) if phases is None else phases):
        idx = np.arange(phi, T - K, K)
        v = g[idx]
        c = np.ones(len(idx), int)
        for i in range(1, len(idx)):
            c[i] = min(c[i - 1] + 1, CH) if v[i] == v[i - 1] else 1
        for b in (0, 1):
            for cc in range(1, CH + 1):
                sel = idx[(v == b) & (c == cc)]
                den[b, cc - 1] += len(sel)
                for j in range(1, K):
                    num[b, cc - 1, j - 1] += (g[sel + j] != b).sum()
    return num, den


def monotone_viol(num, den, K, CH=3, z=2.0):
    """# of (b, j, c) where q increases from c to c+1 by more than z SE."""
    v = 0; tot = 0
    for b in (0, 1):
        for j in range(1, K):
            for cc in range(CH - 1):
                n0, n1 = den[b, cc], den[b, cc + 1]
                if n0 < 30 or n1 < 30:
                    continue
                q0, q1 = num[b, cc, j - 1] / n0, num[b, cc + 1, j - 1] / n1
                se = np.sqrt(q0 * (1 - q0) / n0 + q1 * (1 - q1) / n1) + 1e-12
                tot += 1
                v += int(q1 - q0 > z * se)
    return v, tot


def M2(n_settings=200, per=50, T=20000):
    out = {"DFR": [0, 0, 0], "DFR_ND": [0, 0, 0], "IFR": [0, 0, 0]}
    for fam in out:
        for _ in range(n_settings):
            K = int(rng.integers(4, 13))
            if fam == "IFR":
                kD, kG = rng.uniform(1.5, 4, 2)
                lawD = lambda n, k=kD: np.maximum(1, np.round(8 * rng.weibull(k, n))).astype(int)
                lawG = lambda n, k=kG: np.maximum(1, np.round(8 * rng.weibull(k, n))).astype(int)
            else:
                aD, aG = rng.uniform(0.6, 2.5, 2)
                m = K if fam == "DFR_ND" else 1          # ND: no sojourn shorter than a gap
                lawD = lambda n, a=aD, m=m: (m + np.minimum(np.floor(rng.pareto(a, n) * 3), 50000)).astype(int)
                lawG = lambda n, a=aG, m=m: (m + np.minimum(np.floor(rng.pareto(a, n) * 3), 50000)).astype(int)
            num = np.zeros((2, 3, K)); den = np.zeros((2, 3))
            for _ in range(per):
                a, b = q_table(alt_source(T, lawD, lawG), K, phases=[int(rng.integers(0, K))])
                num += a; den += b
            v, tot = monotone_viol(num, den, K)
            out[fam][0] += 1; out[fam][1] += v; out[fam][2] += tot
    return {k: dict(settings=v[0], sources=v[0] * per, increasing_cells=v[1], tested_cells=v[2])
            for k, v in out.items()}


def M3():
    vids = C.load_videos()
    K = 8
    num = np.zeros((2, 3, K)); den = np.zeros((2, 3))
    for x in vids.values():
        a, b = q_table(x["g"], K)
        num += a; den += b
    v, tot = monotone_viol(num, den, K)
    q = num / np.maximum(den[:, :, None], 1)
    return dict(K=K, increasing_cells=v, tested_cells=tot,
                q_b0=q[0].round(4).tolist(), q_b1=q[1].round(4).tolist(), n=den.tolist())


def M4():
    import optim_delta as O
    vids = C.load_videos(); keys = sorted(vids)
    A, d, abar = 15, 0, 0.25
    sp = O.Space(vids, keys, d, A, {k: 0 for k in keys})
    lam_grid = np.geomspace(0.005, 5, 14)
    inits = {"bestBEL": None, "periodic4": [4] * 6, "random": list(rng.integers(1, A + 2, 6))}
    res = {}
    for name, init in inits.items():
        best = None
        for lam in lam_grid:
            if init is None:
                bel = {(a, b): sp.hist([a] * 3 + [b] * 3) for a in range(1, A + 2) for b in range(1, A + 2)}
                a0, b0 = min(bel, key=lambda k: bel[k]["mean_aoii"] + lam * bel[k]["activation"])
                tab = np.array([a0] * 3 + [b0] * 3)
            else:
                tab = np.array(init)
            f = lambda t: sp.hist(t)["mean_aoii"] + lam * sp.hist(t)["activation"]
            cur = f(tab)
            for _ in range(8):
                imp = False
                for i in range(6):
                    for v in range(1, A + 2):
                        t2 = tab.copy(); t2[i] = v
                        if f(t2) < cur - 1e-12:
                            cur, tab, imp = f(t2), t2, True
                if not imp:
                    break
            r = sp.hist(tab)
            if r["activation"] <= abar + O.TOL and (best is None or r["mean_aoii"] < best[1]):
                best = (tuple(int(v) for v in tab), r["mean_aoii"], r["activation"])
        mono = all(best[0][b * 3 + i] <= best[0][b * 3 + i + 1] for b in (0, 1) for i in (0, 1))
        res[name] = dict(table=best[0], mean_aoii=best[1], activation=best[2], monotone_in_c=mono)
        print("M4", name, res[name])
    return res


def main():
    out = {"seed": C.SEED}
    out["M1_lemma_exact"] = M1(); print("M1", out["M1_lemma_exact"])
    out["M3_corpus"] = M3(); print("M3", {k: out["M3_corpus"][k] for k in ("increasing_cells", "tested_cells")})
    out["M2_mc_without_ND"] = M2(); print("M2", out["M2_mc_without_ND"])
    out["M4_exception"] = M4()
    C.jdump(out, HERE.parent / "results" / "prop34_check.json")


if __name__ == "__main__":
    main()
