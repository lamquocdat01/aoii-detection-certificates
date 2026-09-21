#!/usr/bin/env python3
"""B2 sanity: Monte-Carlo check of the closed-form mean symmetric AoII (d = 0) and of
the worst-case certificate of Lemma 1.  seed = 42.  -> results/sanity_lemma1.json

Three source families: geometric (memoryless), the measured full-corpus duration law,
and the measured sparse-subset law.  The closed form assumes R(K) (every gap >= K), so
each family is simulated twice: R(K)-conditioned (validates the algebra, target < 1%)
and unconditional (quantifies the modelling error of the assumption).
"""
from __future__ import annotations
import numpy as np
import common as C
import theory as TH

RNG = np.random.RandomState(C.SEED)
N_SLOTS = 4_000_000
MC_REPS = 8           # permutation replicates for the low-variance check
N_REP = 5             # i.i.d. replicates for the standard-error check
KS = (2, 4, 8, 16)


def build_sequence(D_law, G_law, n_slots, rng):
    X = np.empty(n_slots + 20_000, dtype=np.int8); i = 0
    nD, nG = len(D_law), len(G_law)
    while i < n_slots:
        g = int(G_law[rng.randint(nG)]); d = int(D_law[rng.randint(nD)])
        if i + g + d >= len(X):
            X = np.concatenate([X, np.empty(g + d + 20_000, dtype=np.int8)])
        X[i:i + g] = 0; i += g
        X[i:i + d] = 1; i += d
    return X[:i]


def build_sequence_perm(D_law, G_law, reps, rng):
    """Low-variance variant: both empirical laws are reproduced by tiling random
    permutations, so each duration is used (almost exactly) equally often and the
    simulated laws match the target laws.  The residual gap to the closed form is
    then algebra error, not sampling noise."""
    def tile(arr, L):
        k = int(np.ceil(L / len(arr)))
        return np.concatenate([rng.permutation(arr) for _ in range(k)])[:L]
    L = reps * max(len(D_law), 2000)
    D = tile(np.asarray(D_law, dtype=np.int64), L)
    G = tile(np.asarray(G_law, dtype=np.int64), L)
    n = int(D.sum() + G.sum())
    X = np.zeros(n, dtype=np.int8)
    i = 0
    for g, d in zip(G, D):
        i += int(g)
        X[i:i + int(d)] = 1
        i += int(d)
    return X


def mc_mean_aoii(X, K):
    """Time-average symmetric AoII over all phases (uniform phase), d = 0."""
    tot_m = tot_f = 0.0; n = 0
    for phi in range(K):
        s = C.periodic(len(X), K, phi)
        xh = C.belief(s, X, 0)
        A = C.aoii_trace(xh, X)
        tot_m += A[(X == 1)].sum(); tot_f += A[(X == 0)].sum(); n += len(X)
    return dict(mean_miss=tot_m / n, mean_false=tot_f / n, mean_total=(tot_m + tot_f) / n)


def main():
    v = C.load_videos()
    D_full = np.array([d for x in v.values() for _, d in x["ev"]], dtype=np.int64)
    G_full = np.concatenate([np.array(x["gaps"], dtype=np.int64) for x in v.values() if x["gaps"]])
    sparse = [k for k, x in v.items() if x["g"].mean() < 0.3]
    D_sp = np.array([d for k in sparse for _, d in v[k]["ev"]], dtype=np.int64)
    G_sp = np.concatenate([np.array(v[k]["gaps"], dtype=np.int64) for k in sparse if v[k]["gaps"]])
    # geometric family matched on the full-corpus means
    pD, pG = 1.0 / D_full.mean(), 1.0 / G_full.mean()
    D_geo = RNG.geometric(pD, size=200_000).astype(np.int64)
    G_geo = RNG.geometric(pG, size=200_000).astype(np.int64)

    fams = dict(geometric=(D_geo, G_geo), measured_full=(D_full, G_full), measured_sparse=(D_sp, G_sp))
    out = {"seed": C.SEED, "n_slots": N_SLOTS, "families": {}}
    for name, (D, G) in fams.items():
        rec = {"D_mean": float(D.mean()), "D_cv": float(D.std() / D.mean()),
               "G_mean": float(G.mean()), "G_median": float(np.median(G)),
               "n_D": int(len(D)), "n_G": int(len(G)), "by_K": {}}
        for K in KS:
            Gc = G[G >= K]
            if len(Gc) < 50:
                Gc = np.maximum(G, K)          # fallback: shift, keeps R(K) true
            cf_c = TH.mean_aoii_closed(D, float(Gc.mean()), K)
            cf_u = TH.mean_aoii_closed(D, float(G.mean()), K)
            rel = lambda a, b: float(abs(a - b) / b) if b else 0.0
            # (i) low-variance check of the algebra
            Xp = build_sequence_perm(D, Gc, MC_REPS, np.random.RandomState(C.SEED + K))
            mc_c = mc_mean_aoii(Xp, K)
            # (ii) i.i.d. replicates -> standard error
            reps = [mc_mean_aoii(build_sequence(D, Gc, N_SLOTS, np.random.RandomState(C.SEED + 100 * j + K)), K)
                    ["mean_total"] for j in range(N_REP)]
            mc_u = mc_mean_aoii(build_sequence(D, G, N_SLOTS, np.random.RandomState(C.SEED + K)), K)
            rec["by_K"][K] = dict(
                closed_RK=cf_c, mc_RK=mc_c, rel_err_RK=rel(mc_c["mean_total"], cf_c["mean_total"]),
                mc_RK_iid_mean=float(np.mean(reps)),
                mc_RK_iid_se=float(np.std(reps, ddof=1) / np.sqrt(len(reps))),
                mc_RK_iid_z=float(abs(np.mean(reps) - cf_c["mean_total"]) /
                                  (np.std(reps, ddof=1) / np.sqrt(len(reps)) + 1e-12)),
                closed_uncond=cf_u, mc_uncond=mc_u,
                rel_err_uncond=rel(mc_u["mean_total"], cf_u["mean_total"]),
                miss_prob=TH.miss_probability(D, K))
        out["families"][name] = rec

    # ---- Lemma 1 worst case on simulated sources (certificate check, d = 0 and d >= 1)
    lem = {}
    for name, (D, G) in fams.items():
        X = build_sequence(D, G, 1_000_000, np.random.RandomState(C.SEED))
        for K in KS:
            for d in (0, 1, 2, 3):
                mx = 0
                for phi in range(K):
                    A = C.aoii_trace(C.belief(C.periodic(len(X), K, phi), X, d), X)
                    mx = max(mx, int(A.max()))
                sc = TH.sojourn_condition(D, G, K, d)
                lem[f"{name}|K{K}|d{d}"] = dict(max_aoii=mx, nominal_K_minus_1_plus_d=K - 1 + d,
                                                holds=bool(mx <= K - 1 + d),
                                                min_sojourn=sc["min_sojourn"],
                                                sojourn_required=sc["required"],
                                                sojourn_cond_holds=sc["holds"])
    out["lemma1_worstcase"] = lem
    worst = [k for k, r in lem.items() if not r["holds"]]
    out["summary"] = dict(
        max_rel_err_RK=max(r["rel_err_RK"] for f in out["families"].values() for r in f["by_K"].values()),
        max_rel_err_uncond=max(r["rel_err_uncond"] for f in out["families"].values() for r in f["by_K"].values()),
        max_iid_z=max(r["mc_RK_iid_z"] for f in out["families"].values() for r in f["by_K"].values()),
        lemma1_d0_all_hold=all(r["holds"] for k, r in lem.items() if k.endswith("|d0")),
        delayed_nominal_violations=worst)
    C.jdump(out, C.RESULTS / "sanity_lemma1.json")
    s = out["summary"]
    print(f"max |MC-closed|/closed under R(K)      : {s['max_rel_err_RK']*100:.3f} %")
    print(f"max |MC-closed|/closed unconditional   : {s['max_rel_err_uncond']*100:.3f} %  (assumption R(K) error)")
    print(f"max |MC_iid-closed|/SE                 : {s['max_iid_z']:.2f}")
    print(f"Lemma 1 (d=0, sup A <= K-1) holds      : {s['lemma1_d0_all_hold']}")
    print(f"nominal K-1+d violated at (d>=1)       : {len(worst)} / {len(lem)-len(KS)} configs")
    return out


if __name__ == "__main__":
    main()
