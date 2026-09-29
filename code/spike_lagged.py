#!/usr/bin/env python3
"""Spike 3.2 (post-TNET reject, 29-09-2026): lagged AoII.  seed = 42.

Uses Submission_TNET/code as a READ-ONLY library; writes only to
PostReject_20260929/results/.

Definitions (padding convention: X_u = 0 for u < 0, consistent with the default
belief Xhat = 0 before the first delivery):
  belief     Xhat_t = X_{sigma(t)},  sigma(t) = max{u <= t-d : s_u = 1}
  AoII       A_t     = (A_{t-1}+1) 1{Xhat_t != X_t}          (as in the TNET paper)
  lagged     Alag_t  = (Alag_{t-1}+1) 1{Xhat_t != X_{t-d}}   (reference = submitted frame)

Checks
  (T1) sup Alag <= K-1 for every sampler with inter-invocation gap <= K and first
       invocation <= K-1, every d (time-shift of Lemma 1); tight.
  (T2) transition-count certificate for the ORIGINAL AoII at d >= 1: every maximal
       mismatch run of length n satisfies n <= (K-1) + N (K-1+d), N = number of
       source transitions in (s-d, t] of the run [s, t].
  (T3) gamma: among source-oblivious samplers, the constraint sup Alag <= A forces
       every gap <= A+1, hence activation >= 1/(A+1); periodic K* = A+1 attains it.
Brute force on random sources/samplers + replay on the 121-video corpus.
"""
from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
LIB = HERE
sys.path.insert(0, str(LIB))
import common as C  # noqa: E402

DS = (0, 1, 2, 3, 5)
KS = (2, 4, 8, 16)
KCAPS = (8, 16, 32)
rng = np.random.default_rng(C.SEED)


# ------------------------------------------------------------------ engine
def lagged_ref(g: np.ndarray, d: int) -> np.ndarray:
    y = np.zeros_like(g)
    y[d:] = g[:len(g) - d] if d else g
    return y


def runs(mask: np.ndarray):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    dm = np.diff(m)
    st = np.flatnonzero(dm == 1)
    en = np.flatnonzero(dm == -1)          # exclusive
    return st, en - st


def transitions(g: np.ndarray) -> np.ndarray:
    """tr[v] = 1 iff X_v != X_{v-1} (X_{-1} = 0)."""
    prev = np.concatenate(([0], g[:-1]))
    return (g != prev).astype(np.int64)


def score(g, s, d, K):
    xh = C.belief(s, g, d)
    A = C.aoii_trace(xh, g)
    Al = C.aoii_trace(xh, lagged_ref(g, d))
    # T2 on original mismatch runs
    st, L = runs(xh != g)
    ctr = np.concatenate(([0], np.cumsum(transitions(g))))   # ctr[i] = sum tr[0..i-1]
    lo = np.maximum(st - d + 1, 0)                          # (s-d, t] -> [s-d+1, t]
    hi = st + L                                             # t+1
    N = ctr[hi] - ctr[lo]
    bound = (K - 1) + N * (K - 1 + d)
    return dict(A=A, Al=Al, n=L, N=N, bound=bound)


# ------------------------------------------------------------- brute force
def brute(n_trials=4000, T=400):
    viol_lag = viol_t2 = 0
    tight_lag = 0
    worst_ratio_t2 = 0.0
    tight_t2 = {}
    cases = 0
    for _ in range(n_trials):
        K = int(rng.integers(2, 9))
        d = int(rng.integers(0, 7))
        kind = rng.integers(0, 3)
        if kind == 0:     # iid-ish flips
            p = rng.uniform(0.02, 0.6)
            g = (np.cumsum(rng.random(T) < p) % 2).astype(np.int8)
        elif kind == 1:   # adversarial 2K-periodic (Prop 2(ii)) with random shift
            sh = int(rng.integers(0, 2 * K))
            t = np.arange(T) + sh
            g = (((t - d) // K) % 2 == 0).astype(np.int8)
        else:             # heavy-tail sojourns
            L = np.maximum(1, (rng.pareto(1.2, T) + 1).astype(int))
            g = np.repeat(np.arange(len(L)) % 2, L)[:T].astype(np.int8)
        # sampler with gaps in [1, K], first invocation <= K-1
        gaps = rng.integers(1, K + 1, T)
        first = int(rng.integers(0, K))
        idx = first + np.concatenate(([0], np.cumsum(gaps)))
        s = np.zeros(T, dtype=bool)
        s[idx[idx < T]] = True
        r = score(g, s, d, K)
        cases += 1
        if r["Al"].max() > K - 1:
            viol_lag += 1
        if r["Al"].max() == K - 1:
            tight_lag += 1
        if len(r["n"]):
            if (r["n"] > r["bound"]).any():
                viol_t2 += 1
            ratio = float((r["n"] / r["bound"]).max())
            worst_ratio_t2 = max(worst_ratio_t2, ratio)
            key = f"d{d}"
            tight_t2[key] = max(tight_t2.get(key, 0.0), ratio)
    return dict(n_trials=cases, T=T, lagged_violations=viol_lag,
                lagged_attained_Kminus1=tight_lag, t2_violations=viol_t2,
                t2_max_ratio_n_over_bound=worst_ratio_t2, t2_max_ratio_by_d=tight_t2)


def brute_gamma(maxT=14):
    """Exhaustive: every oblivious sampler (bit-string) on horizon T with some gap
    >= A+2 admits a source with lagged run > A; periodic A+1 never does."""
    res = []
    for A in (1, 2, 3):
        for d in (0, 1, 2):
            T = maxT
            bad_found_all = True
            n_checked = 0
            for bits in itertools.product((0, 1), repeat=T):
                s = np.array(bits, dtype=bool)
                idx = np.flatnonzero(s)
                pts = np.concatenate(([-1], idx, [T]))
                # only interior-or-leading gaps that fit fully inside horizon (need room
                # for the adversary to be observed: gap ends before T-d)
                big = [(p, q) for p, q in zip(pts[:-1], pts[1:]) if q - p >= A + 2 and p + A + 1 + d < T]
                if not big:
                    continue
                n_checked += 1
                p, q = big[0]
                # adversary: X = belief complement after sample p (belief = X_p, or 0 if p = -1)
                g = np.zeros(T, dtype=np.int8)
                xp = 0
                g[p + 1:] = 1 - xp if p >= 0 else 1
                if p >= 0:
                    g[p] = xp
                Al = C.aoii_trace(C.belief(s, g, d), lagged_ref(g, d))
                if Al.max() <= A:
                    bad_found_all = False
            res.append(dict(A=A, d=d, T=T, samplers_with_gap_ge_A_plus_2=n_checked,
                            adversary_always_exceeds_A=bad_found_all))
    return res


# ------------------------------------------------------------------ corpus
def corpus():
    vids = C.load_videos()
    allS = np.concatenate([x["S"] for x in vids.values()])
    tauW = json.load(open(HERE.parent / "results/aoii_scores.json",
                          encoding="utf-8"))["iso_recall90"]["W"]["tau"]
    tau125 = float(np.quantile(allS, 1 - 0.125))
    pols = []
    for K in KS:
        for phi in range(K):
            pols.append((f"periodic_K{K}", K, phi, lambda x, K=K, phi=phi: C.periodic(len(x["g"]), K, phi)))
    for name, tau in (("iso90W", tauW), ("a0.125", tau125)):
        for Kc in KCAPS:
            pols.append((f"cappedsgate_{name}_Kcap{Kc}", Kc, 0,
                         lambda x, tau=tau, Kc=Kc: C.capped(x["S"] >= tau, Kc)))
        pols.append((f"sgate_{name}", None, 0, lambda x, tau=tau: x["S"] >= tau))

    agg = {}
    for name, K, phi, mk in pols:
        for d in DS:
            key = f"{name}_d{d}"
            a = agg.setdefault(key, dict(policy=name, K=K, d=d, sup_A=0, sup_Alag=0,
                                         sumA=0.0, sumAl=0.0, slots=0, acts=0,
                                         t2_viol=0, t2_runs=0, t2_max_ratio=0.0,
                                         t2_tight_runs=0, n_phase=0))
            a["n_phase"] += 1
            for x in vids.values():
                g = x["g"]
                s = mk(x)
                Kb = K if K is not None else 10 ** 9
                r = score(g, s, d, Kb if K is not None else 1)
                a["sup_A"] = max(a["sup_A"], int(r["A"].max()))
                a["sup_Alag"] = max(a["sup_Alag"], int(r["Al"].max()))
                a["sumA"] += float(r["A"].sum())
                a["sumAl"] += float(r["Al"].sum())
                a["slots"] += len(g)
                a["acts"] += int(s.sum())
                if K is not None and len(r["n"]):
                    a["t2_runs"] += len(r["n"])
                    a["t2_viol"] += int((r["n"] > r["bound"]).sum())
                    a["t2_max_ratio"] = max(a["t2_max_ratio"], float((r["n"] / r["bound"]).max()))
                    a["t2_tight_runs"] += int((r["n"] == r["bound"]).sum())
    rows = []
    for key, a in agg.items():
        rows.append(dict(
            key=key, policy=a["policy"], K=a["K"], d=a["d"],
            activation=a["acts"] / a["slots"],
            sup_aoii=a["sup_A"], sup_aoii_lag=a["sup_Alag"],
            mean_aoii=a["sumA"] / a["slots"], mean_aoii_lag=a["sumAl"] / a["slots"],
            bound_lag=(a["K"] - 1 if a["K"] is not None else None),
            lag_holds=(a["sup_Alag"] <= a["K"] - 1 if a["K"] is not None else None),
            lag_tightness=((a["sup_Alag"] / (a["K"] - 1)) if a["K"] is not None else None),
            nominal_Km1pd=(a["K"] - 1 + a["d"] if a["K"] is not None else None),
            nominal_violated=(a["sup_A"] > a["K"] - 1 + a["d"] if a["K"] is not None else None),
            t2_runs=a["t2_runs"], t2_violations=a["t2_viol"],
            t2_max_ratio=a["t2_max_ratio"], t2_tight_runs=a["t2_tight_runs"],
            n_phase=a["n_phase"]))
    return rows


def gamma_table(rows):
    ef = json.load(open(HERE.parent / "results/energy_frontier.json",
                        encoding="utf-8"))["fit"]
    W0, W1 = ef["W0"], ef["slope_W"]
    E = lambda a: (W0 + W1 * a) * C.SLOT_MS                    # mJ / slot
    out = []
    by = {r["key"]: r for r in rows}
    for A in (7, 15, 31):
        K = A + 1
        a_star = 1.0 / K
        cap_iso = by.get(f"cappedsgate_iso90W_Kcap{K}_d0")
        cap_125 = by.get(f"cappedsgate_a0.125_Kcap{K}_d0")
        per = by.get(f"periodic_K{K}_d0")
        out.append(dict(
            A=A, K_star=K, a_star=a_star, E_star_mJ=E(a_star),
            E_formula="(W0 + W1/(A+1)) * slot_ms", W0=W0, W1=W1,
            periodic_measured_activation=(per["activation"] if per else None),
            capped_iso90W_activation=(cap_iso["activation"] if cap_iso else None),
            capped_iso90W_E_mJ=(E(cap_iso["activation"]) if cap_iso else None),
            capped_a0125_activation=(cap_125["activation"] if cap_125 else None),
            capped_a0125_E_mJ=(E(cap_125["activation"]) if cap_125 else None)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"seed": C.SEED, "delays": list(DS)}
    res["brute_force"] = brute()
    res["brute_gamma"] = brute_gamma()
    rows = corpus()
    res["corpus"] = rows
    res["gamma"] = gamma_table(rows)
    cert = [r for r in rows if r["K"] is not None]
    dpos = [r for r in cert if r["d"] >= 1]
    res["summary"] = dict(
        n_certified_checkpoints=len(cert),
        lag_violations=sum(not r["lag_holds"] for r in cert),
        lag_tight=sum(r["lag_tightness"] == 1.0 for r in cert),
        n_checkpoints_d_ge_1=len(dpos),
        lag_violations_d_ge_1=sum(not r["lag_holds"] for r in dpos),
        nominal_Km1pd_violations_d_ge_1=sum(r["nominal_violated"] for r in dpos),
        t2_violations=sum(r["t2_violations"] for r in cert),
        t2_runs=sum(r["t2_runs"] for r in cert),
        t2_max_ratio=max(r["t2_max_ratio"] for r in cert),
        mean_lag_equals_mean_d0=None)
    # mean Alag(d) vs mean A(d=0): must agree up to the last-d-slot truncation
    diffs = []
    for r in cert:
        r0 = next(q for q in rows if q["policy"] == r["policy"] and q["d"] == 0)
        diffs.append(abs(r["mean_aoii_lag"] - r0["mean_aoii"]) / max(r0["mean_aoii"], 1e-12))
    res["summary"]["mean_lag_rel_diff_vs_d0_max"] = max(diffs)
    C.jdump(res, OUT / "spike_lagged.json")
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["brute_force"], indent=1))
    print(json.dumps(res["brute_gamma"], indent=1))


if __name__ == "__main__":
    main()
