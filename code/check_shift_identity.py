#!/usr/bin/env python3
"""Identity behind Thm L (spike 3.2): Alag_t(d) == A_{t-d}(d=0) for every t >= d,
and Alag_t = 0 for t < d (padding X_u = 0, u < 0). Checked slot-by-slot on the corpus."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from spike_lagged import lagged_ref, DS, KS, KCAPS

vids = C.load_videos()
allS = np.concatenate([x["S"] for x in vids.values()])
tau = float(np.quantile(allS, 1 - 0.125))
n_cmp = n_bad = 0
for x in vids.values():
    g = x["g"]
    samplers = [C.periodic(len(g), K, p) for K in KS for p in range(K)]
    samplers += [C.capped(x["S"] >= tau, Kc) for Kc in KCAPS] + [x["S"] >= tau]
    for s in samplers:
        A0 = C.aoii_trace(C.belief(s, g, 0), g)
        for d in DS:
            Al = C.aoii_trace(C.belief(s, g, d), lagged_ref(g, d))
            ok = np.array_equal(Al[d:], A0[:len(g) - d]) and not Al[:d].any()
            n_cmp += 1; n_bad += (not ok)
res = dict(n_trace_comparisons=n_cmp, mismatches=n_bad, identity_holds=(n_bad == 0))
C.jdump(res, HERE.parent / "results" / "shift_identity.json")
print(res)
