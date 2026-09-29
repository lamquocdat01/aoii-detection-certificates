#!/usr/bin/env python3
"""Cor 2.2: distributional certificate for the ORIGINAL AoII at d >= 1 from the
source's transition counts only.  seed = 42.

Pathwise (suffix form of Theorem 2, periodic K, 1 <= d <= K):
    A_t >= n   ==>   N_(t-n+1, t] >= k(n) := ceil((n-d+1)/K) - 1
so for a stationary source  P(A >= n) <= P(N_{n-1} >= k(n)),  N_w = #transitions in
a window of w slots.  The right side depends on the source law only.
Evaluated (i) empirically on the corpus (both sides, same traces) and (ii) under the
alternating-renewal (semi-Markov) model with the MEASURED duration and gap laws.
Output: results/cor22_tail.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

K, D = 8, 2
NS = np.arange(1, 61)
EPS = (1e-2, 1e-3, 1e-4)


def k_of(n):
    return np.maximum(np.ceil((n - D + 1) / K).astype(int) - 1, 0)


def window_counts(g, w):
    tr = np.concatenate(([int(g[0] != 0)], (g[1:] != g[:-1]).astype(int)))
    c = np.concatenate(([0], np.cumsum(tr)))
    T = len(g)
    t = np.arange(T)
    lo = np.maximum(t - w + 1, 0)          # (t-w, t]
    return c[t + 1] - c[lo]


def r_of(n):
    """# of consecutive complete sojourns of length exactly K forced by A_t >= n."""
    return np.ceil(n / K).astype(int) - 3


def exactK_chain_end(g):
    """chain[t] = length of the longest run of consecutive COMPLETE sojourns of
    length exactly K that ends at or before t and is followed (inside the window
    test) -- we return, for every slot t, the pair (chain length c, start slot of
    the chain) for the chain whose last sojourn ends at t-1 (i.e. transition at t)."""
    T = len(g)
    tr = np.flatnonzero(np.concatenate(([g[0] != 0], g[1:] != g[:-1])))
    return tr


def sharp_event(g, n):
    """True at slot t iff (t-n+1, t] contains >= r(n) consecutive complete sojourns
    of length exactly K (sojourn = [tr_i, tr_{i+1}-1])."""
    T = len(g); r = int(r_of(np.array([n]))[0])
    if r <= 0:
        return np.ones(T, bool)
    tr = exactK_chain_end(g)
    L = np.diff(tr)                          # complete sojourn lengths
    ok = (L == K).astype(int)
    ev = np.zeros(T, bool)
    # chains of r consecutive exact-K sojourns: sojourns i..i+r-1, span [tr_i, tr_{i+r}-1]
    if len(ok) >= r:
        cs = np.concatenate(([0], np.cumsum(ok)))
        good = np.flatnonzero(cs[r:] - cs[:-r] == r)   # i with sojourns i..i+r-1 all K
        for i in good:
            st, en = tr[i], tr[i + r] - 1              # chain occupies [st, en]
            # window (t-n+1, t] = [t-n+2, t] must contain [st, en]:  en <= t <= st+n-2
            lo, hi = en, min(st + n - 2, T - 1)
            if lo <= hi:
                ev[lo:hi + 1] = True
    return ev


def empirical(vids):
    tailA = np.zeros(len(NS)); tailB = np.zeros(len(NS)); tot = 0
    tailS = np.zeros(len(NS)); violS = 0
    viol = 0
    sharp = {k: [sharp_event(x["g"], int(n)) for n in NS] for k, x in vids.items()}
    for key, x in vids.items():
        g = x["g"]
        for phi in range(K):
            s = C.periodic(len(g), K, phi)
            A = C.aoii_trace(C.belief(s, g, D), g)
            tot += len(g)
            for i, n in enumerate(NS):
                ev = A >= n
                cnt = window_counts(g, n - 1) if n > 1 else np.zeros(len(g), int)
                bnd = cnt >= k_of(np.array([n]))[0]
                tailA[i] += ev.sum(); tailB[i] += bnd.sum()
                viol += int((ev & ~bnd).sum())          # pathwise implication
                sh = sharp[key][i]
                tailS[i] += sh.sum(); violS += int((ev & ~sh).sum())
    return tailA / tot, tailB / tot, viol, tailS / tot, violS


def semi_markov(vids, T=10_000_000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    Dlaw = np.array([dd for x in vids.values() for _, dd in x["ev"]])
    Glaw = np.concatenate([np.array(x["gaps"]) for x in vids.values() if x["gaps"]])
    n = int(T / (Dlaw.mean() + Glaw.mean())) + 10
    L = np.empty(2 * n, int)
    L[0::2] = rng.choice(Glaw, n); L[1::2] = rng.choice(Dlaw, n)
    g = np.repeat(np.arange(2 * n) % 2, L)[:T].astype(np.int8)
    tail = np.zeros(len(NS))
    for i, n_ in enumerate(NS):
        cnt = window_counts(g, n_ - 1) if n_ > 1 else np.zeros(T, int)
        tail[i] = (cnt >= k_of(np.array([n_]))[0]).mean()
    # model AoII tail (periodic K, random phase)
    s = C.periodic(T, K, int(rng.integers(0, K)))
    A = C.aoii_trace(C.belief(s, g, D), g)
    tailA = np.array([(A >= n_).mean() for n_ in NS])
    tailS = np.array([sharp_event(g, int(n_)).mean() for n_ in NS])
    # closed-form union bound: P <= (n-1) * lam * prod of alternating p_D(K), p_G(K)
    pD = float((Dlaw == K).mean()); pG = float((Glaw == K).mean())
    lam = 2.0 / (Dlaw.mean() + Glaw.mean())
    cf = []
    for n_ in NS:
        r = int(r_of(np.array([n_]))[0])
        if r <= 0:
            cf.append(1.0); continue
        a = pD ** ((r + 1) // 2) * pG ** (r // 2); b = pG ** ((r + 1) // 2) * pD ** (r // 2)
        cf.append(min(1.0, (n_ - 1) * lam * 0.5 * (a + b)))
    return tail, tailA, tailS, np.array(cf), dict(pD_K=pD, pG_K=pG, lam=lam)


def quant(tail, eps):
    idx = np.flatnonzero(tail <= eps)
    return int(NS[idx[0]]) if len(idx) else None


def main():
    vids = C.load_videos()
    eA, eB, viol, eS, violS = empirical(vids)
    mB, mA, mS, cf, par = semi_markov(vids)
    out = dict(K=K, d=D, n=NS.tolist(),
               empirical_tail_A=eA.tolist(), empirical_tail_bound=eB.tolist(),
               pathwise_violations=viol,
               model_tail_bound=mB.tolist(), model_tail_A=mA.tolist(),
               sharp_empirical_tail=eS.tolist(), sharp_pathwise_violations=violS,
               sharp_model_tail=mS.tolist(), sharp_closed_form_union=cf.tolist(),
               closed_form_params=par,
               quantiles={str(e): dict(emp_A=quant(eA, e), emp_bound=quant(eB, e),
                                       model_bound=quant(mB, e), model_A=quant(mA, e),
                                       sharp_emp=quant(eS, e), sharp_model=quant(mS, e),
                                       sharp_closed_form=quant(cf, e))
                          for e in EPS},
               note="P(A_t>=n) <= P(N_{n-1} >= ceil((n-d+1)/K)-1); model = alternating "
                    "renewal with measured duration/gap laws (i.i.d. resampling)")
    C.jdump(out, HERE.parent / "results" / "cor22_tail.json")
    print("pathwise violations:", viol, "sharp:", violS, par)
    for e in EPS:
        print(e, out["quantiles"][str(e)])
    for i in (0, 4, 9, 19, 20, 29, 39):
        print(NS[i], f"A {eA[i]:.2e} crude {eB[i]:.2e} sharp {eS[i]:.2e} | model A {mA[i]:.2e} sharp {mS[i]:.2e} cf {cf[i]:.2e}")


if __name__ == "__main__":
    main()
