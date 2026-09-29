#!/usr/bin/env python3
"""CP-delta1 / P2: certificate-constrained optimal sampling (problem P-M).  seed = 42.

  min_{pi in Pi} mean AoII(pi)   s.t.  (i)  sup AoII_lag(pi) <= A   (certificate)
                                        (ii) activation a(pi) <= abar (energy budget)

Policy classes (every one is forced to cap A+1, which is equivalent to (i), Prop 3.1):
  PER   periodic K,                      K <= A+1
  BEL   belief-dependent age threshold:  after an invocation whose verdict is b,
        invoke again after K_b slots,    max(1,d) <= K_b <= A+1        (causal: K_b >= d)
  HIST  history-dependent threshold:     K(b, c), c = # consecutive identical
        verdicts (capped at C = 3),      max(1,d) <= K(b,c) <= A+1
  GATE  capped content gate (S_t >= tau, forced refresh at K_cap = A+1)
Randomisation = time-sharing between two deterministic policies (both capped, the
switch happens at an invocation, so (i) is preserved).

Evaluation = trace replay on the 121-video corpus (the object reported), plus a
hold-out protocol (optimise on 60 % of the videos, evaluate on 40 %) and a
model-based protocol (optimise on an alternating-renewal source resampled from the
TRAIN duration/gap laws, evaluate on TEST traces).
Output: results/optim_delta.json
"""
from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

AS = (3, 7, 15, 31)
ABARS = (0.125, 0.25, 0.5)
DS = (0, 2)
CH = 3                               # history depth of HIST
TOL = 0.002                          # budget tolerance on measured activation
W = C.W_WINDOW
EF = json.load(open(HERE.parent / "results" / "energy_frontier.json", encoding="utf-8"))["fit"]
W0, W1 = EF["W0"], EF["slope_W"]
mJ = lambda a: (W0 + W1 * a) * C.SLOT_MS


# ------------------------------------------------------------------ kernels
@njit(cache=True)
def schedule_hist(g, Ktab, CH, phi):
    """Invocation indicator for a HIST table Ktab[b, c-1]; first invocation at phi."""
    T = g.shape[0]
    s = np.zeros(T, np.bool_)
    t = phi
    pb = -1
    c = 0
    while t < T:
        s[t] = True
        b = g[t]
        if b == pb:
            c = c + 1 if c < CH else CH
        else:
            c = 1
        pb = b
        t += Ktab[b, c - 1]
    return s


@njit(cache=True)
def evaluate(g, s, d, W):
    """Returns (slots, acts, sumA, supA, supAlag, n_ev, strict, window)."""
    T = g.shape[0]
    xh = np.zeros(T, np.int8)
    last = -1
    acts = 0
    for t in range(T):
        if s[t]:
            acts += 1
        u = t - d
        if u >= 0 and s[u]:
            last = u
        xh[t] = g[last] if last >= 0 else 0
    A = 0; Al = 0; sumA = 0.0; supA = 0; supAl = 0
    for t in range(T):
        A = A + 1 if xh[t] != g[t] else 0
        ref = g[t - d] if t >= d else 0
        Al = Al + 1 if xh[t] != ref else 0
        sumA += A
        if A > supA:
            supA = A
        if Al > supAl:
            supAl = Al
    nxt = np.full(T + 1, T + 10 ** 6, np.int64)
    for t in range(T - 1, -1, -1):
        nxt[t] = t if s[t] else nxt[t + 1]
    n_ev = 0; strict = 0; win = 0
    t = 0
    while t < T:
        if g[t] == 1:
            on = t
            while t < T and g[t] == 1:
                t += 1
            D = t - on
            n_ev += 1
            q = nxt[on]
            if q < on + D:
                strict += 1
            if q < on + (D if D > W else W):
                win += 1
        else:
            t += 1
    return T, acts, sumA, supA, supAl, n_ev, strict, win


def pooled(vids, keys, make_s, d):
    tot = np.zeros(8)
    supA = supAl = 0
    for k in keys:
        x = vids[k]
        r = evaluate(x["g"], make_s(k, x), d, W)
        tot[[0, 1, 2, 5, 6, 7]] += np.array([r[0], r[1], r[2], r[5], r[6], r[7]], float)
        supA = max(supA, r[3]); supAl = max(supAl, r[4])
    return dict(activation=tot[1] / tot[0], mean_aoii=tot[2] / tot[0], sup_aoii=int(supA),
                sup_aoii_lag=int(supAl), recall_strict=tot[6] / tot[5],
                recall_W=tot[7] / tot[5], n_slots=int(tot[0]))


# ------------------------------------------------------------------ helpers
def _first(s):
    s = s.copy(); s[0] = True
    return s


def lower_hull(pts):
    """pts: list of (a, J, payload). Lower convex hull in (a, J), a ascending."""
    P = sorted(pts, key=lambda p: (p[0], p[1]))
    H = []
    for p in P:
        while len(H) >= 2:
            (a1, j1, _), (a2, j2, _) = H[-2], H[-1]
            if (j2 - j1) * (p[0] - a1) >= (p[1] - j1) * (a2 - a1):
                H.pop()
            else:
                break
        H.append(p)
    # keep the decreasing part only (more activation never needed once J rises)
    out = []
    for p in H:
        if not out or p[1] < out[-1][1]:
            out.append(p)
    return out


def best_at(pts, abar):
    """Deterministic best with a <= abar, and randomised (hull) value at abar."""
    feas = [p for p in pts if p[0] <= abar + TOL]   # TOL: periodic 1/(A+1) measures 0.1253 (slot-0 start, clip rounding)
    if not feas:
        return None
    det = min(feas, key=lambda p: p[1])
    H = lower_hull(pts)
    rnd = None
    for (a1, j1, p1), (a2, j2, p2) in zip(H[:-1], H[1:]):
        if a1 <= abar <= a2:
            th = (abar - a1) / (a2 - a1) if a2 > a1 else 0.0
            rnd = dict(J=(1 - th) * j1 + th * j2, theta=th, lo=p1, hi=p2, a_lo=a1, a_hi=a2)
    if rnd is None or det[1] <= rnd["J"]:
        rnd = dict(J=det[1], theta=0.0, lo=det[2], hi=det[2], a_lo=det[0], a_hi=det[0])
    return dict(det_J=det[1], det_a=det[0], det=det[2], rnd=rnd)


class Space:
    """Evaluates policies on a set of videos with a common start-offset per video."""

    def __init__(self, vids, keys, d, A, offsets):
        self.vids, self.keys, self.d, self.A = vids, keys, d, A
        self.off = offsets
        self.cache = {}

    def phi(self, k):
        # every policy invokes at slot 0 (common warm start).  A random first
        # invocation in [0, A] adds a start-up transient whose quadratic AoII area
        # dominates the mean (found and fixed 29-09, see CP_DELTA1_REPORT).
        return 0

    def per(self, K):
        key = ("PER", K)
        if key not in self.cache:
            self.cache[key] = pooled(self.vids, self.keys,
                                     lambda k, x: C.periodic(len(x["g"]), K, self.phi(k) % K), self.d)
        return self.cache[key]

    def hist(self, tab):
        tab = tuple(int(v) for v in np.asarray(tab).ravel())
        key = ("HIST", tab)
        if key not in self.cache:
            T2 = np.array(tab, np.int64).reshape(2, CH)
            self.cache[key] = pooled(self.vids, self.keys,
                                     lambda k, x: schedule_hist(x["g"], T2, CH, self.phi(k)), self.d)
        return self.cache[key]

    def gate(self, tau):
        key = ("GATE", round(float(tau), 9))
        if key not in self.cache:
            Kc = self.A + 1
            self.cache[key] = pooled(self.vids, self.keys,
                                     lambda k, x: C.capped(_first(x["S"] >= tau), Kc), self.d)
        return self.cache[key]


def optimise(sp, taus, lambdas):
    A, d = sp.A, sp.d
    kmin = max(1, d)
    pts = {"PER": [], "BEL": [], "HIST": [], "GATE": []}
    for K in range(1, A + 2):
        r = sp.per(K); pts["PER"].append((r["activation"], r["mean_aoii"], ("PER", K)))
    bel = {}
    for K0 in range(kmin, A + 2):
        for K1 in range(kmin, A + 2):
            r = sp.hist([K0] * CH + [K1] * CH)
            bel[(K0, K1)] = r
            pts["BEL"].append((r["activation"], r["mean_aoii"], ("BEL", K0, K1)))
    pts["HIST"] += pts["BEL"]
    # HIST: coordinate descent on J + lam*a from the best BEL policy for each lam
    # multi-start (added in Đợt 1b after a local optimum was found at d=0, A=15,
    # abar=0.25): best BEL, best periodic, and two seeded random tables per lambda
    rs = np.random.default_rng(C.SEED)
    for lam in lambdas:
        K0, K1 = min(bel, key=lambda k: bel[k]["mean_aoii"] + lam * bel[k]["activation"])
        Kp = min(range(kmin, A + 2), key=lambda K: bel[(K, K)]["mean_aoii"] + lam * bel[(K, K)]["activation"])
        starts = [np.array([K0] * CH + [K1] * CH), np.array([Kp] * (2 * CH)),
                  rs.integers(kmin, A + 2, 2 * CH), rs.integers(kmin, A + 2, 2 * CH)]
        f = lambda t: sp.hist(t)["mean_aoii"] + lam * sp.hist(t)["activation"]
        for tab in starts:
            cur = f(tab)
            for _ in range(8):
                improved = False
                for i in range(2 * CH):
                    for v in range(kmin, A + 2):
                        t2 = tab.copy(); t2[i] = v
                        val = f(t2)
                        if val < cur - 1e-12:
                            cur, tab, improved = val, t2, True
                if not improved:
                    break
            r = sp.hist(tab)
            pts["HIST"].append((r["activation"], r["mean_aoii"], ("HIST", tuple(int(v) for v in tab))))
    for tau in taus:
        r = sp.gate(tau); pts["GATE"].append((r["activation"], r["mean_aoii"], ("GATE", float(tau))))
    return pts


def describe(sp, payload):
    kind = payload[0]
    if kind == "PER":
        return sp.per(payload[1])
    if kind == "BEL":
        return sp.hist([payload[1]] * CH + [payload[2]] * CH)
    if kind == "HIST":
        return sp.hist(payload[1])
    return sp.gate(payload[1])


def main():
    vids = C.load_videos()
    keys = sorted(vids)
    rng = np.random.default_rng(C.SEED)
    offsets = {k: int(rng.integers(0, 10 ** 6)) for k in keys}
    allS = np.concatenate([vids[k]["S"] for k in keys])
    taus = np.quantile(allS, 1 - np.linspace(0.01, 0.99, 50))
    lambdas = np.geomspace(0.005, 5, 14)
    out = {"seed": C.SEED, "classes": ["PER", "BEL", "HIST", "GATE"], "CH": CH,
           "A": list(AS), "abar": list(ABARS), "d": list(DS), "table": [], "cert": []}

    for d in DS:
        for A in AS:
            sp = Space(vids, keys, d, A, offsets)
            pts = optimise(sp, taus, lambdas)
            # certificate check on every evaluated policy
            ncheck = nviol = 0
            for v in sp.cache.values():
                ncheck += 1; nviol += int(v["sup_aoii_lag"] > A)
            out["cert"].append(dict(d=d, A=A, n_policies=ncheck, lag_violations=nviol))
            for abar in ABARS:
                row = dict(d=d, A=A, abar=abar)
                for cls in ("PER", "BEL", "HIST", "GATE"):
                    b = best_at(pts[cls], abar)
                    if b is None:
                        row[cls] = None
                        continue
                    r = describe(sp, b["det"])
                    row[cls] = dict(policy=list(map(str, b["det"])), det_a=b["det_a"],
                                    det_mean_aoii=b["det_J"], rnd_mean_aoii=b["rnd"]["J"],
                                    rnd_theta=b["rnd"]["theta"],
                                    rnd_pair=[list(map(str, b["rnd"]["lo"])), list(map(str, b["rnd"]["hi"]))],
                                    recall_strict=r["recall_strict"], recall_W=r["recall_W"],
                                    sup_aoii=r["sup_aoii"], sup_aoii_lag=r["sup_aoii_lag"],
                                    mJ_per_slot=mJ(b["det_a"]))
                out["table"].append(row)
                print(d, A, abar, {c: (None if row[c] is None else
                                       (round(row[c]["det_mean_aoii"], 4), round(row[c]["rnd_mean_aoii"], 4),
                                        row[c]["policy"], round(row[c]["det_a"], 3)))
                                   for c in ("PER", "BEL", "HIST", "GATE")})
            # price-of-certificate curve (randomised optimum, HIST class) for figures
            out.setdefault("curve", []).append(dict(
                d=d, A=A, hull_HIST=[(p[0], p[1]) for p in lower_hull(pts["HIST"])],
                hull_PER=[(p[0], p[1]) for p in lower_hull(pts["PER"])],
                hull_GATE=[(p[0], p[1]) for p in lower_hull(pts["GATE"])]))
    # unconstrained reference (A = infinity approximated by A = 127), SAME classes
    # (HIST incl. BEL/PER) so that the price of the certificate compares like with like
    for d in DS:
        sp = Space(vids, keys, d, 127, offsets)
        pts = optimise(sp, [], lambdas)
        out.setdefault("uncapped_BEL", []).append(
            dict(d=d, A_ref=127, at={str(ab): best_at(pts["HIST"], ab)["rnd"]["J"] for ab in ABARS},
                 at_det={str(ab): best_at(pts["HIST"], ab)["det_J"] for ab in ABARS},
                 policy={str(ab): list(map(str, best_at(pts["HIST"], ab)["det"])) for ab in ABARS}))
    C.jdump(out, HERE.parent / "results" / "optim_delta.json")
    for c in out["cert"]:
        print(c)


if __name__ == "__main__":
    main()
