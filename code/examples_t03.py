#!/usr/bin/env python3
"""B1 worked examples for the T03 definitions, plus the explicit counterexample used
in Proposition 2(ii).  Every number quoted in theory_notes.md Sec. 1-2 is produced
here.  Output: results/examples_t03.json
"""
from __future__ import annotations
import numpy as np
import common as C


def trace(X, s, d):
    X = np.asarray(X, dtype=np.int8)
    s = np.asarray(s, dtype=bool)
    xh = C.belief(s, X, d)
    A = C.aoii_trace(xh, X)
    return dict(X=X.tolist(), sampled=np.flatnonzero(s).tolist(), belief=xh.tolist(),
                A=A.tolist(), max_A=int(A.max()),
                miss_slots=np.flatnonzero((X == 1) & (xh == 0)).tolist(),
                false_slots=np.flatnonzero((X == 0) & (xh == 1)).tolist())


def event_seq(T, onset, D):
    X = np.zeros(T, dtype=np.int8)
    X[onset:onset + D] = 1
    return X


def main():
    out = {"seed": C.SEED}
    K = 4
    s = C.periodic(20, K, 0)

    # Example 1: event shorter than K, sampling phase misses it entirely
    out["example1_short_event_missed"] = dict(
        K=K, d=0, onset=5, D=2,
        comment="D < K and no sample epoch falls inside the event: peak AoII = D = 2, "
                "the event is never detected, and the symmetric AoII never charges the miss.",
        **trace(event_seq(20, 5, 2), s, 0))

    # Example 2: event longer than K, miss side attains K-1, short false-side tail
    out["example2_long_event"] = dict(
        K=K, d=0, onset=5, D=10,
        comment="r = 3 slots to the next epoch: miss-side run 3 = K-1; after the offset "
                "the belief stays 1 for F = (K - ((D-r) mod K)) mod K = 1 slot.",
        **trace(event_seq(20, 5, 10), s, 0))

    # Example 3: false-side stretch after the offset
    out["example3_false_side"] = dict(
        K=K, d=0, onset=5, D=8,
        comment="r = 3, (D-r) mod K = 1, F = 3: the monitor believes the event is still "
                "running for three slots after it ended.",
        **trace(event_seq(20, 5, 8), s, 0))

    # Example 4: Proposition 2(ii) counterexample -- delay destroys the certificate
    ce = {}
    for K_, d_ in ((2, 2), (4, 2), (8, 2)):
        T = 40 * K_
        t = np.arange(T)
        X = ((t % (2 * K_)) >= d_) & ((t % (2 * K_)) <= K_ + d_ - 1)
        r = trace(X.astype(np.int8), C.periodic(T, K_, 0), d_)
        ce[f"K{K_}_d{d_}"] = dict(nominal_bound=K_ - 1 + d_, max_A=r["max_A"], horizon=T,
                                  activation=1.0 / K_,
                                  source="X_t = 1{ t mod 2K in [d, K+d-1] }",
                                  grows_without_bound=bool(r["max_A"] >= T - d_ - 1))
    out["example4_delay_counterexample"] = ce

    # horizon doubling to show the run really is unbounded
    K_, d_ = 2, 2
    grow = []
    for T in (40, 400, 4000, 40000):
        t = np.arange(T)
        X = (((t % (2 * K_)) >= d_) & ((t % (2 * K_)) <= K_ + d_ - 1)).astype(np.int8)
        A = C.aoii_trace(C.belief(C.periodic(T, K_, 0), X, d_), X)
        grow.append(dict(horizon=T, max_A=int(A.max())))
    out["example4_horizon_growth"] = grow

    C.jdump(out, C.RESULTS / "examples_t03.json")
    for k in ("example1_short_event_missed", "example2_long_event", "example3_false_side"):
        e = out[k]
        print(f"{k}: A = {e['A'][:18]} ... max {e['max_A']}, miss {e['miss_slots']}, false {e['false_slots']}")
    print("counterexample:", {k: (v["nominal_bound"], v["max_A"]) for k, v in ce.items()})
    print("horizon growth (K=2,d=2):", [(g["horizon"], g["max_A"]) for g in grow])
    return out


if __name__ == "__main__":
    main()
