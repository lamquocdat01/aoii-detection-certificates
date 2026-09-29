#!/usr/bin/env python3
"""Refined transition-count bound (conjecture T2*, from exhaustive search K<=4, d<=K):
a mismatch run with N>=1 source transitions in (s-d, t] has length <= N*K + d - 1
(N = 0: <= K-1). Checked on the corpus for periodic (every phase) and capped gates.
Output: results/t2_refined_corpus.json"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from spike_lagged import runs, transitions, DS, KS, KCAPS
vids = C.load_videos()
allS = np.concatenate([x["S"] for x in vids.values()])
tauW = json.load(open(HERE.parent / "results/aoii_scores.json", encoding="utf-8"))["iso_recall90"]["W"]["tau"]
tau125 = float(np.quantile(allS, 1 - 0.125))
pols = [(f"periodic_K{K}", K, (lambda x, K=K, p=p: C.periodic(len(x["g"]), K, p))) for K in KS for p in range(K)]
pols += [(f"capped_{n}_Kcap{Kc}", Kc, (lambda x, t=t, Kc=Kc: C.capped(x["S"] >= t, Kc)))
         for n, t in (("iso90W", tauW), ("a0.125", tau125)) for Kc in KCAPS]
agg = {}
for name, K, mk in pols:
    for d in DS:
        a = agg.setdefault(f"{name}_d{d}", dict(policy=name, K=K, d=d, regime_d_le_K=d <= K,
                                                runs=0, viol=0, tight=0, max_ratio=0.0))
        for x in vids.values():
            g = x["g"]; s = mk(x)
            st, L = runs(C.belief(s, g, d) != g)
            ctr = np.concatenate(([0], np.cumsum(transitions(g))))
            N = ctr[st + L] - ctr[np.maximum(st - d + 1, 0)]
            b = np.where(N == 0, K - 1, N * K + d - 1)
            a["runs"] += len(L); a["viol"] += int((L > b).sum()); a["tight"] += int((L == b).sum())
            if len(L):
                a["max_ratio"] = max(a["max_ratio"], float((L / b).max()))
rows = list(agg.values())
summ = dict(
    periodic_d_le_K=dict(runs=sum(r["runs"] for r in rows if r["policy"].startswith("periodic") and r["regime_d_le_K"]),
                         violations=sum(r["viol"] for r in rows if r["policy"].startswith("periodic") and r["regime_d_le_K"]),
                         tight_runs=sum(r["tight"] for r in rows if r["policy"].startswith("periodic") and r["regime_d_le_K"] and r["d"] >= 1)),
    periodic_d_gt_K_violations=sum(r["viol"] for r in rows if r["policy"].startswith("periodic") and not r["regime_d_le_K"]),
    capped_violations=sum(r["viol"] for r in rows if r["policy"].startswith("capped")))
C.jdump(dict(rows=rows, summary=summ), HERE.parent / "results" / "t2_refined_corpus.json")
print(json.dumps(summ, indent=1))
for r in rows:
    if r["d"] >= 1: print(r["policy"], r["d"], r["viol"], r["tight"], round(r["max_ratio"], 3))
