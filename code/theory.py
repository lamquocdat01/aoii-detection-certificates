#!/usr/bin/env python3
"""Closed-form AoII expressions for periodic-K sampling (Part B, Topic 01).

Notation (theory_notes.md):
  alternating renewal binary source:  gap G (X=0) then event D (X=1), i.i.d.
  periodic-K sampling, uniformly random phase, zero delivery delay (d = 0)
  r = slots from an event onset to the next sample epoch, r ~ Uniform{0..K-1}
Assumption R(K):  every gap satisfies G >= K.  Under R(K) the belief is 0 at every
onset and the false-side stretch is never truncated by the next onset.
"""
from __future__ import annotations
import numpy as np

tri = lambda n: n * (n + 1) / 2.0          # 1+2+...+n  (AoII area of a run of length n)


def cycle_area(D: int, r: int, K: int):
    """(miss-side area, false-side area) accrued in one gap+event cycle, d = 0."""
    if r >= D:                              # event never sampled -> missed entirely
        return tri(D), 0.0
    F = (K - ((D - r) % K)) % K             # false-side stretch after the offset
    return tri(r), tri(F)


def mean_aoii_closed(D_law, G_mean, K):
    """Exact mean (time-average) symmetric AoII under R(K).  D_law = array of event
    durations (empirical law, equal weights).  Returns dict with the decomposition."""
    D = np.asarray(D_law, dtype=np.int64)
    Lam = D.mean() + G_mean
    miss = np.zeros(len(D)); fals = np.zeros(len(D))
    for r in range(K):
        m = np.minimum(r, D)
        a_m = tri(m)
        Dm = (D - r) % K
        F = np.where(r >= D, 0, (K - Dm) % K)
        miss += a_m * (r >= D) + tri(np.where(r >= D, 0, r)) * (r < D)
        fals += tri(F) * (r < D)
    miss /= K; fals /= K
    return dict(mean_miss=miss.mean() / Lam, mean_false=fals.mean() / Lam,
                mean_total=(miss.mean() + fals.mean()) / Lam, Lambda=Lam, K=K)


def mean_aoii_bounds(D_law, G_mean, K):
    """Two-sided bound of Proposition 4 (heavy tail): the miss-side term is exact and
    the false-side term is at most K(K-1)/2 per cycle."""
    c = mean_aoii_closed(D_law, G_mean, K)
    lo = c["mean_miss"]
    hi = c["mean_miss"] + K * (K - 1) / 2.0 / c["Lambda"]
    return dict(lower=lo, upper=hi, exact=c["mean_total"], **c)


def miss_probability(D_law, K):
    """P(event never sampled) = P(r >= D), r ~ U{0..K-1}."""
    D = np.asarray(D_law, dtype=np.int64)
    return float(np.mean([np.mean(r >= D) for r in range(K)]))


def sojourn_condition(D_law, G_law, K, d):
    """Lemma 1(ii) requires every sojourn (event or gap) to be >= K+d+1."""
    mn = min(int(np.min(D_law)), int(np.min(G_law))) if len(G_law) else int(np.min(D_law))
    return dict(min_sojourn=mn, required=K + d + 1, holds=bool(mn >= K + d + 1))
