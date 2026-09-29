#!/usr/bin/env python3
"""Đợt 1b: final tables with the nesting of feasible sets made explicit.  seed = 42.

Prop 5: a policy capped at A'+1 <= A+1 is feasible for A, so the optimum over the
union of all policies evaluated for A' <= A is a valid (and better) value for A:
    J*(A) := min_{A' <= A} J_found(A'),   J*(inf) := min over all A (incl. A = 127).
This removes the non-monotonicity produced by the local (coordinate-descent) search
for large A.  Also re-checks the Markov c-independence on HULL (randomised) values,
since deterministic best-at-budget values differ by activation granularity.
Output: results/summary_delta.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import optim_delta as O  # noqa: E402
import markov_baseline as M  # noqa: E402

R = HERE.parent / "results"


def main():
    r = json.load(open(R / "optim_delta.json", encoding="utf-8"))
    out = {"nested": [], "markov_hull_check": []}
    for d in (0, 2):
        unc = [u for u in r["uncapped_BEL"] if u["d"] == d][0]
        for ab in O.ABARS:
            rows = sorted([x for x in r["table"] if x["d"] == d and abs(x["abar"] - ab) < 1e-9],
                          key=lambda x: x["A"])
            best_r = best_d = np.inf
            inf_r = min([x["HIST"]["rnd_mean_aoii"] for x in rows if x["HIST"]] + [unc["at"][str(ab)]])
            for x in rows:
                if x["HIST"] is None:
                    out["nested"].append(dict(d=d, abar=ab, A=x["A"], feasible=False)); continue
                best_r = min(best_r, x["HIST"]["rnd_mean_aoii"])
                best_d = min(best_d, x["HIST"]["det_mean_aoii"])
                per = x["PER"]["det_mean_aoii"]
                gate = x["GATE"]["rnd_mean_aoii"] if x["GATE"] else None
                out["nested"].append(dict(
                    d=d, abar=ab, A=x["A"], feasible=True, J_opt_rnd=best_r, J_opt_det=best_d,
                    J_periodic=per, J_gate=gate, J_inf=inf_r,
                    price_vs_inf=best_r / inf_r - 1, gain_vs_periodic=best_r / per - 1,
                    policy_found=x["HIST"]["policy"], a_det=x["HIST"]["det_a"],
                    mJ_det=x["HIST"]["mJ_per_slot"], mJ_periodic=x["PER"]["mJ_per_slot"],
                    recall_strict_opt=x["HIST"]["recall_strict"], recall_strict_per=x["PER"]["recall_strict"],
                    recall_W_opt=x["HIST"]["recall_W"], recall_W_per=x["PER"]["recall_W"]))
    # Markov c-independence on hull values
    vids = C.load_videos()
    train, _ = M.split(vids)
    mD, mG = M.sojourn_means(vids, train)
    mk = M.markov_clips(mD, mG)
    for A in (7, 15):
        sp = O.Space(mk, list(mk), 2, A, {k: 0 for k in mk})
        pts = O.optimise(sp, [], M.LAMBDAS)
        for ab in O.ABARS:
            bh, bb = O.best_at(pts["HIST"], ab), O.best_at(pts["BEL"], ab)
            if bh is None:
                continue
            out["markov_hull_check"].append(dict(
                A=A, abar=ab, hull_HIST=bh["rnd"]["J"], hull_BEL=bb["rnd"]["J"],
                rel_gain=(bb["rnd"]["J"] - bh["rnd"]["J"]) / bb["rnd"]["J"],
                det_HIST=bh["det_J"], det_BEL=bb["det_J"]))
    C.jdump(out, R / "summary_delta.json")
    for x in out["nested"]:
        if x["feasible"]:
            print(x["d"], x["abar"], x["A"], "opt", round(x["J_opt_rnd"], 4), round(x["J_opt_det"], 4),
                  "per", round(x["J_periodic"], 4), "gate", x["J_gate"] and round(x["J_gate"], 4),
                  "inf", round(x["J_inf"], 4), "price", f"{x['price_vs_inf']:+.1%}",
                  "vsPer", f"{x['gain_vs_periodic']:+.1%}")
        else:
            print(x["d"], x["abar"], x["A"], "infeasible")
    for m in out["markov_hull_check"]:
        print("markov", m)


if __name__ == "__main__":
    main()
