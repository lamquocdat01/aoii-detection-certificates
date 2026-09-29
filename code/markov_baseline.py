#!/usr/bin/env python3
"""Đợt 1b / Việc 1: the precedent baseline — the policy that is optimal when the
source is ASSUMED Markov (geometric sojourns), evaluated on the measured traces.
seed = 42.

Markov fit: two-state discrete-time chain with exit probabilities p_D = 1/E[D],
p_G = 1/E[G] (mean-matched sojourns), fitted on TRAIN (60/40 split identical to
holdout_delta.py) and on the whole corpus.  The optimum is searched in the SAME
observation class (HIST(j,b,c) incl. BEL/PER), under the SAME constraints (cap A+1,
a <= abar) and the same delay, with Algorithm 1, but J and a are evaluated on a
simulated Markov source (40 clips x 50 000 slots = 2e6 slots).  For a memoryless
source c carries no information: we check that HIST-opt == BEL-opt on the Markov
source (c-independence).
Compared on TEST (and on the whole corpus): periodic (best K), capped content gate
(tau chosen on TRAIN), HIST-opt trained on TRAIN traces, Markov-opt.
Output: results/markov_baseline.json, figures/figD3_price_of_assumption.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import optim_delta as O  # noqa: E402

D = 2
AS = (7, 15, 31)
FIG_A = 15
FIG_ABARS = (0.07, 0.09, 0.125, 0.16, 0.2, 0.25, 0.33, 0.5)
LAMBDAS = np.geomspace(0.005, 5, 10)


def split(vids):
    keys = sorted(vids)
    rng = np.random.default_rng(C.SEED)                   # identical to holdout_delta.py
    perm = rng.permutation(len(keys))
    ntr = int(round(0.6 * len(keys)))
    return [keys[i] for i in perm[:ntr]], [keys[i] for i in perm[ntr:]]


def sojourn_means(vids, keys):
    Dl = np.array([dd for k in keys for _, dd in vids[k]["ev"]])
    Gl = np.concatenate([np.array(vids[k]["gaps"]) for k in keys if vids[k]["gaps"]])
    return float(Dl.mean()), float(Gl.mean())


def markov_clips(mD, mG, n_clips=40, T=50000, seed=C.SEED + 7):
    rng = np.random.default_rng(seed)
    pD, pG = 1.0 / mD, 1.0 / mG
    out = {}
    pi1 = mD / (mD + mG)
    for i in range(n_clips):
        x = int(rng.random() < pi1)
        u = rng.random(T)
        g = np.empty(T, np.int8)
        for t in range(T):
            g[t] = x
            if (x == 1 and u[t] < pD) or (x == 0 and u[t] < pG):
                x = 1 - x
        out[("markov", i)] = dict(g=g, S=np.zeros(T), ev=C.runs_of_ones(g))
    return out


_PTS = {}


def opt_det(sp, abar, cls="HIST"):
    if id(sp) not in _PTS:
        _PTS[id(sp)] = O.optimise(sp, taus=[], lambdas=LAMBDAS)
    b = O.best_at(_PTS[id(sp)][cls], abar)
    return None if b is None else b


def best_periodic(sp, A, abar):
    c = [sp.per(K) for K in range(1, A + 2) if sp.per(K)["activation"] <= abar + O.TOL]
    return min(c, key=lambda r: r["mean_aoii"]) if c else None


def gate_on_train(sp_tr, sp_te, taus, abar):
    c = [(t, sp_tr.gate(t)) for t in taus if sp_tr.gate(t)["activation"] <= abar + O.TOL]
    if not c:
        return None, None
    t, _ = min(c, key=lambda z: z[1]["mean_aoii"])
    return float(t), sp_te.gate(t)


def slim(r):
    if r is None:
        return None
    return {k: r[k] for k in ("activation", "mean_aoii", "sup_aoii", "sup_aoii_lag",
                              "recall_strict", "recall_W")} | {"mJ_per_slot": O.mJ(r["activation"])}


def main():
    vids = C.load_videos()
    train, test = split(vids)
    allk = sorted(vids)
    mD_tr, mG_tr = sojourn_means(vids, train)
    mD_all, mG_all = sojourn_means(vids, allk)
    mk_tr = markov_clips(mD_tr, mG_tr)
    mk_all = markov_clips(mD_all, mG_all, seed=C.SEED + 8)
    zero = {k: 0 for k in list(vids) + list(mk_tr) + list(mk_all)}
    allS = np.concatenate([vids[k]["S"] for k in train])
    taus = np.quantile(allS, 1 - np.linspace(0.01, 0.99, 50))
    out = dict(seed=C.SEED, d=D, fit=dict(train=dict(mean_D=mD_tr, mean_G=mG_tr, p_D=1 / mD_tr, p_G=1 / mG_tr),
                                          all=dict(mean_D=mD_all, mean_G=mG_all, p_D=1 / mD_all, p_G=1 / mG_all)),
               markov_slots=40 * 50000, rows=[], c_independence=[], fig=[])

    spaces = {}

    def run(A, abar, record=True):
        if A not in spaces:
            spaces[A] = (O.Space(mk_tr, list(mk_tr), D, A, zero), O.Space(mk_all, list(mk_all), D, A, zero),
                         O.Space(vids, train, D, A, zero), O.Space(vids, test, D, A, zero),
                         O.Space(vids, allk, D, A, zero))
        sp_mk, sp_mka, sp_tr, sp_te, sp_all = spaces[A]
        b_mk = opt_det(sp_mk, abar); b_mka = opt_det(sp_mka, abar)
        b_bel = opt_det(sp_mk, abar, "BEL")
        b_tr = opt_det(sp_tr, abar)
        if b_mk is None:
            return None
        tau, g_te = gate_on_train(sp_tr, sp_te, taus, abar)
        # (b) budget-corrected Markov-opt (decision 29-09): keep the Markov model's
        # ranking, but enforce the budget on the REAL traces, calibrated on TRAIN only:
        # the Markov-best policy whose TRAIN-trace activation is <= abar (+TOL)
        bc = None
        for p in sorted(_PTS[id(sp_mk)]["HIST"], key=lambda z: z[1]):
            if p[0] > abar + O.TOL + 0.25:          # far outside the budget on the model
                continue
            if O.describe(sp_tr, p[2])["activation"] <= abar + O.TOL:
                bc = p; break
        row = dict(A=A, abar=abar,
                   markov_opt=dict(policy=list(map(str, b_mk["det"])), J_on_markov=b_mk["det_J"],
                                   test=slim(O.describe(sp_te, b_mk["det"]))),
                   markov_opt_bc=(None if bc is None else dict(
                       policy=list(map(str, bc[2])), J_on_markov=bc[1], a_on_markov=bc[0],
                       a_train=O.describe(sp_tr, bc[2])["activation"],
                       test=slim(O.describe(sp_te, bc[2])))),
                   markov_opt_all=dict(policy=list(map(str, b_mka["det"])),
                                       corpus=slim(O.describe(sp_all, b_mka["det"]))),
                   hist_opt_train=dict(policy=list(map(str, b_tr["det"])),
                                       test=slim(O.describe(sp_te, b_tr["det"])),
                                       corpus=slim(O.describe(sp_all, b_tr["det"]))),
                   periodic_test=slim(best_periodic(sp_te, A, abar)),
                   periodic_corpus=slim(best_periodic(sp_all, A, abar)),
                   gate_test=dict(tau_from_train=tau, test=slim(g_te)))
        ci = dict(A=A, abar=abar, J_markov_HIST=b_mk["det_J"], J_markov_BEL=b_bel["det_J"],
                  HIST_policy=list(map(str, b_mk["det"])), BEL_policy=list(map(str, b_bel["det"])),
                  rel_gain_HIST_over_BEL=(b_bel["det_J"] - b_mk["det_J"]) / b_bel["det_J"])
        return row, ci

    for A in AS:
        for abar in O.ABARS:
            r = run(A, abar)
            if r is None:
                print(A, abar, "infeasible"); continue
            row, ci = r
            out["rows"].append(row); out["c_independence"].append(ci)
            f = lambda z: None if z is None else round(z["mean_aoii"], 4)
            print(A, abar, "markov->test", f(row["markov_opt"]["test"]), row["markov_opt"]["policy"],
                  "a", round(row["markov_opt"]["test"]["activation"], 3),
                  "| markov-bc", None if row["markov_opt_bc"] is None else (f(row["markov_opt_bc"]["test"]),
                  row["markov_opt_bc"]["policy"], round(row["markov_opt_bc"]["test"]["activation"], 3)),
                  "| hist->test", f(row["hist_opt_train"]["test"]), "| per", f(row["periodic_test"]),
                  "| gate", f(row["gate_test"]["test"]), "| c-indep gain", round(ci["rel_gain_HIST_over_BEL"], 4))
    for abar in FIG_ABARS:
        r = run(FIG_A, abar)
        if r is None:
            continue
        row, _ = r
        out["fig"].append(dict(abar=abar,
                               periodic=row["periodic_test"]["mean_aoii"] if row["periodic_test"] else None,
                               markov=row["markov_opt"]["test"]["mean_aoii"],
                               markov_a=row["markov_opt"]["test"]["activation"],
                               markov_bc=(row["markov_opt_bc"]["test"]["mean_aoii"] if row["markov_opt_bc"] else None),
                               gate=row["gate_test"]["test"]["mean_aoii"] if row["gate_test"]["test"] else None,
                               hist=row["hist_opt_train"]["test"]["mean_aoii"]))
    C.jdump(out, HERE.parent / "results" / "markov_baseline.json")

    F = out["fig"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = [f["abar"] for f in F]
    for key, lab, sty in (("periodic", "periodic (best K)", "s:"),
                          ("markov_bc", "Markov-optimal, budget-corrected", "d-."),
                          ("gate", "capped content gate", "^--"), ("hist", "history policy (this work)", "o-")):
        ax.plot(x, [f[key] if f[key] is not None else np.nan for f in F], sty, ms=3, lw=1, label=lab)
    ax.set_xlabel(r"activation budget $\bar a$"); ax.set_ylabel("mean AoII on TEST (slots)")
    ax.set_yscale("log")
    ax.set_title(f"price of the Markov assumption, A={FIG_A}, d={D}", fontsize=7)
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    (HERE.parent / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE.parent / "figures" / f"figD3_price_of_assumption.{ext}", dpi=300)


if __name__ == "__main__":
    main()
