#!/usr/bin/env python3
"""P2(d): hold-out protocol for the certificate-constrained optimum.  seed = 42.
Split the 121 videos 60/40 (seed 42).  For d = 2, A in {7, 15}, abar in {.125,.25,.5}:
  trace-opt : optimise HIST/BEL on TRAIN traces, evaluate the chosen deterministic
              policy on TEST traces
  model-opt : optimise on an alternating-renewal source resampled from the TRAIN
              duration/gap laws (40 synthetic clips x 20 000 slots), evaluate on TEST
  oracle    : optimise directly on TEST (in-sample reference)
  periodic  : best periodic K on TEST with a <= abar
Output: results/holdout_delta.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import optim_delta as O  # noqa: E402

D = 2
AS = (7, 15)


def synth(train_vids, n_clips=40, T=20000, seed=C.SEED):
    rng = np.random.default_rng(seed + 1)
    Dl = np.array([dd for x in train_vids.values() for _, dd in x["ev"]])
    Gl = np.concatenate([np.array(x["gaps"]) for x in train_vids.values() if x["gaps"]])
    out = {}
    for i in range(n_clips):
        n = int(T / (Dl.mean() + Gl.mean())) * 2 + 20
        L = np.empty(2 * n, int); L[0::2] = rng.choice(Gl, n); L[1::2] = rng.choice(Dl, n)
        start = int(rng.integers(0, 2))
        g = np.repeat((np.arange(2 * n) + start) % 2, L)[:T].astype(np.int8)
        out[("synth", i)] = dict(g=g, S=np.zeros(T), ev=C.runs_of_ones(g))
    return out


def best_payload(sp, abar, lambdas):
    pts = O.optimise(sp, taus=[], lambdas=lambdas)
    cand = pts["HIST"]
    b = O.best_at(cand, abar)
    return None if b is None else b["det"]


def main():
    vids = C.load_videos()
    keys = sorted(vids)
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(keys))
    ntr = int(round(0.6 * len(keys)))
    train = [keys[i] for i in perm[:ntr]]; test = [keys[i] for i in perm[ntr:]]
    offsets = {k: int(rng.integers(0, 10 ** 6)) for k in keys}
    lambdas = np.geomspace(0.005, 5, 10)
    sv = synth({k: vids[k] for k in train})
    soff = {k: int(rng.integers(0, 10 ** 6)) for k in sv}
    rows = []
    for A in AS:
        sp_tr = O.Space(vids, train, D, A, offsets)
        sp_te = O.Space(vids, test, D, A, offsets)
        sp_sy = O.Space(sv, list(sv), D, A, soff)
        for abar in O.ABARS:
            p_tr = best_payload(sp_tr, abar, lambdas)
            p_sy = best_payload(sp_sy, abar, lambdas)
            p_or = best_payload(sp_te, abar, lambdas)
            per = min((sp_te.per(K) for K in range(1, A + 2) if sp_te.per(K)["activation"] <= abar + O.TOL),
                      key=lambda r: r["mean_aoii"], default=None)
            ev = lambda p: O.describe(sp_te, p) if p is not None else None
            r_tr, r_sy, r_or = ev(p_tr), ev(p_sy), ev(p_or)
            row = dict(A=A, abar=abar, n_train=len(train), n_test=len(test),
                       trace_opt=dict(policy=str(p_tr), test=r_tr),
                       model_opt=dict(policy=str(p_sy), test=r_sy),
                       oracle=dict(policy=str(p_or), test=r_or),
                       periodic_test=per)
            rows.append(row)
            f = lambda r: None if r is None else (round(r["mean_aoii"], 4), round(r["activation"], 3), r["sup_aoii_lag"])
            print(A, abar, "trace", f(r_tr), "model", f(r_sy), "oracle", f(r_or), "per", f(per))
    C.jdump(dict(seed=C.SEED, d=D, split="60/40 videos, seed 42", rows=rows),
            HERE.parent / "results" / "holdout_delta.json")


if __name__ == "__main__":
    main()
