#!/usr/bin/env python3
"""B4 heavy-tail proposition: two-sided mean-AoII bounds under the measured duration
law, tail x K sensitivity, and the comparison against a memoryless source of the same
mean.  seed = 42.  Output: results/heavy_tail.json

Every duration-law number is printed with its scope (full corpus / sparse subset).
"""
from __future__ import annotations
import numpy as np
import common as C
import theory as TH

KS = (2, 4, 8, 16, 32)
RNG = np.random.RandomState(C.SEED)


def law_summary(D):
    D = np.asarray(D, dtype=float)
    return dict(n=int(len(D)), mean=float(D.mean()), median=float(np.median(D)),
                cv=float(D.std() / D.mean()),
                frac_single=float((D == 1).mean()), frac_lt5=float((D < 5).mean()),
                frac_lt8=float((D < 8).mean()),
                p90=float(np.percentile(D, 90)), p99=float(np.percentile(D, 99)),
                max=float(D.max()))


def main():
    vids = C.load_videos()
    sparse_keys = [k for k, x in vids.items() if x["g"].mean() < 0.3]
    scopes = {
        "full": dict(D=np.array([d for x in vids.values() for _, d in x["ev"]], dtype=np.int64),
                     G=np.concatenate([np.array(x["gaps"], dtype=np.int64)
                                       for x in vids.values() if x["gaps"]]),
                     n_videos=len(vids)),
        "sparse": dict(D=np.array([d for k in sparse_keys for _, d in vids[k]["ev"]], dtype=np.int64),
                       G=np.concatenate([np.array(vids[k]["gaps"], dtype=np.int64)
                                         for k in sparse_keys if vids[k]["gaps"]]),
                       n_videos=len(sparse_keys)),
    }
    out = {"seed": C.SEED, "sparse_definition": "videos with in-event fraction < 0.3",
           "scopes": {}}
    for name, sc in scopes.items():
        D, G = sc["D"], sc["G"]
        rec = dict(n_videos=sc["n_videos"], duration_law=law_summary(D), gap_law=law_summary(G),
                   by_K={})
        # memoryless reference with the same mean event duration and same mean gap
        D_geo = RNG.geometric(1.0 / D.mean(), size=400_000).astype(np.int64)
        for K in KS:
            b = TH.mean_aoii_bounds(D, float(G.mean()), K)
            bg = TH.mean_aoii_bounds(D_geo, float(G.mean()), K)
            rec["by_K"][K] = dict(
                lower=b["lower"], upper=b["upper"], exact=b["exact"],
                mean_miss=b["mean_miss"], mean_false=b["mean_false"],
                miss_probability=TH.miss_probability(D, K),
                geometric_exact=bg["exact"], geometric_miss_probability=TH.miss_probability(D_geo, K),
                ratio_measured_over_geometric=b["exact"] / bg["exact"] if bg["exact"] else None)
        # tail x K sensitivity: truncate the duration law at a quantile and refit
        sens = {}
        for q in (0.90, 0.95, 0.99, 0.999, 1.0):
            cap = float(np.quantile(D, q))
            Dt = np.minimum(D, cap)
            sens[str(q)] = dict(cap=cap, mean=float(Dt.mean()), cv=float(Dt.std() / Dt.mean()),
                                by_K={K: TH.mean_aoii_bounds(Dt, float(G.mean()), K)["exact"]
                                      for K in KS})
        rec["tail_sensitivity"] = sens
        out["scopes"][name] = rec

    f, s = out["scopes"]["full"], out["scopes"]["sparse"]
    print("scope   n_ev  mean  median   CV   single<1 <5     <8")
    for nm, r in (("full", f), ("sparse", s)):
        d = r["duration_law"]
        print(f"{nm:7s}{d['n']:6d}{d['mean']:7.2f}{d['median']:7.1f}{d['cv']:7.2f}"
              f"{d['frac_single']*100:8.1f}%{d['frac_lt5']*100:6.1f}%{d['frac_lt8']*100:6.1f}%")
    print("\nK   full: lower  exact  upper  missP | geom exact  missP")
    for K in KS:
        a = f["by_K"][K]
        print(f"{K:<4d}{a['lower']:8.4f}{a['exact']:8.4f}{a['upper']:8.4f}{a['miss_probability']*100:7.1f}%"
              f" |{a['geometric_exact']:9.4f}{a['geometric_miss_probability']*100:7.1f}%")
    C.jdump(out, C.RESULTS / "heavy_tail.json")
    return out


if __name__ == "__main__":
    main()
