#!/usr/bin/env python3
"""B3 sanity for Theorem 2 (impossibility for content-driven gates with unbounded
closed intervals) and its Corollary (capped gate).  seed = 42.
Output: results/sanity_thm.json

Three numerical checks:
  (T2-a) explicit adversarial construction: for every target B a realisation exists
         with A_t > B, on both the miss side and the false side;
  (T2-b) growth-with-horizon check on the measured corpus: the realised worst-case
         AoII of the S-gate keeps growing with the observation horizon (no plateau),
         whereas a capped gate saturates at K_cap - 1;
  (Cor)  capping any sampler at K_cap restores sup A = K_cap - 1 at d = 0 -- verified
         on 2000 random gates and on the corpus S-gate -- together with the activation
         (energy) price of the cap.
Also recomputes the cue-onset AUC of the frame score (draft v1 quoted 0.49).
"""
from __future__ import annotations
import numpy as np
import common as C

RNG = np.random.RandomState(C.SEED)


# ---------------------------------------------------------------- T2 (a)
def adversarial_instance(B, K_closed=None):
    """Explicit realisation reaching AoII > B under a content-driven gate.

    Score s_t = 1 on the first slot of every event and 0 elsewhere is a legitimate
    content function; the gate {s_t >= 1} therefore never fires inside a long event
    after its onset slot has passed, nor during a long silence.  Feeding it a source
    with one event of duration B+2 whose onset slot is suppressed (score 0, because
    the cue is only emitted after a warm-up of one slot) yields a mismatch run of
    length B+1 on the miss side; the mirrored construction does the same on the false
    side.  Nothing here depends on B, so no finite certificate exists.
    """
    L = B + 2
    # miss side: gate fires once at t = 0 (sees X = 0) then never again
    X = np.zeros(2 * L + 4, dtype=np.int8)
    X[2:2 + L] = 1
    s = np.zeros_like(X, dtype=bool)
    s[0] = True
    A_miss = C.aoii_trace(C.belief(s, X, 0), X).max()
    # false side: gate fires once inside the event then never again
    X2 = np.zeros(2 * L + 4, dtype=np.int8)
    X2[0:3] = 1
    s2 = np.zeros_like(X2, dtype=bool)
    s2[1] = True
    A_false = C.aoii_trace(C.belief(s2, X2, 0), X2).max()
    return int(A_miss), int(A_false)


# ---------------------------------------------------------------- T2 (b)
def horizon_growth(vids, tau, horizons):
    """Worst-case AoII of the S-gate as a function of the number of slots observed."""
    order = sorted(vids)
    out = []
    for H in horizons:
        budget = H
        mx_gate = 0
        mx_cap = {8: 0, 16: 0, 32: 0}
        for k in order:
            if budget <= 0:
                break
            x = vids[k]
            n = min(budget, len(x["g"]))
            g = x["g"][:n]
            s = x["S"][:n] >= tau
            mx_gate = max(mx_gate, int(C.aoii_trace(C.belief(s, g, 0), g).max()))
            for Kc in mx_cap:
                sc = C.capped(s, Kc)
                mx_cap[Kc] = max(mx_cap[Kc], int(C.aoii_trace(C.belief(sc, g, 0), g).max()))
            budget -= n
        out.append(dict(horizon=int(H), sgate_max_aoii=mx_gate,
                        capped_max_aoii={str(k): v for k, v in mx_cap.items()}))
    return out


# ---------------------------------------------------------------- Corollary
def corollary_random(n_cases=2000):
    bad = []
    for _ in range(n_cases):
        T = RNG.randint(20, 400)
        # bursty binary source
        X = np.zeros(T, dtype=np.int8)
        i = 0
        while i < T:
            g = RNG.geometric(1 / RNG.uniform(1.2, 30))
            d = RNG.geometric(1 / RNG.uniform(1.2, 30))
            i += g
            X[i:i + d] = 1
            i += d
        s = RNG.rand(T) < RNG.uniform(0.0, 0.5)
        Kc = int(RNG.randint(2, 40))
        A = C.aoii_trace(C.belief(C.capped(s, Kc), X, 0), X)
        if A.max() > Kc - 1:
            bad.append(dict(T=T, K_cap=Kc, max_aoii=int(A.max())))
    return bad


# ---------------------------------------------------------------- cue-onset AUC
def cue_onset_auc(vids, keys=None):
    """AUC of the frame score for predicting an onset one slot ahead, restricted to
    background slots: positives are background slots immediately preceding an onset."""
    sc, lab = [], []
    for k, x in (vids.items() if keys is None else ((k, vids[k]) for k in keys)):
        g = x["g"].astype(bool)
        S = x["S"]
        bg = ~g[:-1]
        onset_next = g[1:] & ~g[:-1]
        sc.append(S[:-1][bg])
        lab.append(onset_next[bg])
    sc = np.concatenate(sc)
    lab = np.concatenate(lab).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return None
    order = np.argsort(sc, kind="mergesort")
    ranks = np.empty(len(sc), dtype=float)
    ranks[order] = np.arange(1, len(sc) + 1)
    # average ranks for ties
    s_sorted = sc[order]
    r_sorted = ranks[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r_sorted[i:j + 1] = r_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = r_sorted
    n1 = int(lab.sum()); n0 = len(lab) - n1
    auc = (ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(auc=float(auc), n_pos=n1, n_neg=n0)


def main():
    vids = C.load_videos()
    allS = np.concatenate([x["S"] for x in vids.values()])
    out = {"seed": C.SEED}

    out["T2a_adversarial"] = [dict(B=B, miss_side_aoii=m, false_side_aoii=f,
                                   exceeds_B=bool(m > B and f > B))
                              for B in (10, 100, 1000, 10000)
                              for m, f in [adversarial_instance(B)]]
    out["T2a_all_exceed"] = all(r["exceeds_B"] for r in out["T2a_adversarial"])

    tau = float(np.quantile(allS, 1 - 0.125))
    out["T2b_horizon_growth"] = dict(
        tau=tau, a_target=0.125,
        rows=horizon_growth(vids, tau, (5_000, 20_000, 50_000, 100_000, 166_589)))

    bad = corollary_random()
    out["corollary_random_gates"] = dict(n_cases=2000, n_violations=len(bad), violations=bad[:10])

    # activation price of the cap, measured on the corpus
    price = {}
    for a_t in (0.125, 0.5):
        t = float(np.quantile(allS, 1 - a_t))
        base = np.mean([ (x["S"] >= t).mean() for x in vids.values() ])
        n_tot = sum(len(x["g"]) for x in vids.values())
        a_base = sum(int((x["S"] >= t).sum()) for x in vids.values()) / n_tot
        row = {}
        for Kc in (8, 16, 32):
            a_cap = sum(int(C.capped(x["S"] >= t, Kc).sum()) for x in vids.values()) / n_tot
            row[str(Kc)] = dict(a_capped=a_cap, delta_a=a_cap - a_base,
                                relative_increase=a_cap / a_base - 1.0,
                                certificate=Kc - 1)
        price[f"a={a_t}"] = dict(tau=t, a_uncapped=a_base, by_K_cap=row)
    out["corollary_energy_price"] = price

    sparse = [k for k, x in vids.items() if x["g"].mean() < 0.3]
    out["cue_onset_auc"] = dict(full=cue_onset_auc(vids), sparse=cue_onset_auc(vids, sparse))

    C.jdump(out, C.RESULTS / "sanity_thm.json")
    print(f"T2(a) adversarial construction exceeds every target B : {out['T2a_all_exceed']}")
    rows = out["T2b_horizon_growth"]["rows"]
    print("T2(b) horizon -> S-gate max AoII : " +
          ", ".join(f"{r['horizon']}:{r['sgate_max_aoii']}" for r in rows))
    print("      capped K_cap=32          : " +
          ", ".join(f"{r['horizon']}:{r['capped_max_aoii']['32']}" for r in rows))
    print(f"Corollary violations on 2000 random gates : {out['corollary_random_gates']['n_violations']}")
    print(f"cue-onset AUC full = {out['cue_onset_auc']['full']['auc']:.4f}, "
          f"sparse = {out['cue_onset_auc']['sparse']['auc']:.4f}")
    return out


if __name__ == "__main__":
    main()
