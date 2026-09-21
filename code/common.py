#!/usr/bin/env python3
"""Common loader / AoII engine for Topic 01 (TNET), Part B. seed = 42.

Slot model (theory_notes.md Sec. 1):
  * slot t = one video frame; slot duration = 1/30 s = 33.3 ms
  * X_t in {0,1} = g_t (ground-truth event indicator of frame t)
  * a sampling policy chooses s_t in {0,1} (detector invoked on frame t)
  * constant detection delay d slots: an invocation at u delivers X_u at slot u+d
  * belief  Xhat_t = X_{sigma(t)},  sigma(t) = max{u <= t-d : s_u = 1}  (0 if none)
  * SYMMETRIC AoII: A_t = (A_{t-1}+1) * 1{Xhat_t != X_t},  A_{-1} = 0
Read-only on all source data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
FPS = 30.0
SLOT_MS = 1000.0 / FPS
W_WINDOW = 7                      # window-refresh recall protocol (A3)
HERE = Path(__file__).resolve().parent
PKG = HERE.parent                 # repository root
RESULTS = PKG / "results"
FIGURES = PKG / "figures"
DATA = PKG / "data"
CFS = DATA                        # derived event / frame-score traces
C1 = DATA / "power"               # raw board-power logs
C1_CSV = C1 / "orin_results.csv"
C1_LOGS = C1
DATASETS = ("CDnet2014", "LASIESTA", "BMC")   # the three source datasets

DATASET_DOI = {
    "CDnet2014": "10.1109/CVPRW.2014.126",
    "LASIESTA": "10.1016/j.cviu.2016.08.005",
    "BMC": "10.1007/978-3-642-37410-4_25",
}


# ----------------------------------------------------------------- loading
def load_frames() -> pd.DataFrame:
    fs = pd.read_parquet(CFS / "frame_scores.parquet")
    fs = fs[fs.dataset.isin(DATASETS)].sort_values(["dataset", "video", "frame"])
    return fs.reset_index(drop=True)


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(CFS / "events_gated.csv")
    return ev[ev.dataset.isin(DATASETS)].reset_index(drop=True)


def runs_of_ones(g: np.ndarray):
    """Maximal runs of g_t == 1 -> list of (onset_index, duration)."""
    g = np.asarray(g).astype(bool)
    out, on = [], None
    for i, x in enumerate(g):
        if x and on is None:
            on = i
        elif (not x) and on is not None:
            out.append((on, i - on)); on = None
    if on is not None:
        out.append((on, len(g) - on))
    return out


def gaps_of_zeros(g: np.ndarray):
    """Maximal interior runs of g_t == 0 (silence law). Leading/trailing runs are
    censored by the clip boundary and are excluded."""
    g = np.asarray(g).astype(bool)
    out, on = [], None
    for i, x in enumerate(g):
        if (not x) and on is None:
            on = i
        elif x and on is not None:
            if on > 0:
                out.append(i - on)
            on = None
    return out


def load_videos(cache={}):
    """dict (dataset, video) -> {g, S, ev, gaps}. Events are re-derived from the
    g_t runs, so the replay never depends on a hand-made event table."""
    if cache:
        return cache["v"]
    fs = load_frames()
    vids = {}
    for (ds, v), dd in fs.groupby(["dataset", "video"], sort=True):
        g = dd.g_t.values.astype(np.int8)
        vids[(ds, v)] = dict(g=g, S=dd.score.values.astype(np.float64),
                             frame0=int(dd.frame.values[0]),
                             ev=runs_of_ones(g), gaps=gaps_of_zeros(g))
    cache["v"] = vids
    return vids


# ------------------------------------------------------------- AoII engine
def belief(sampled: np.ndarray, g: np.ndarray, d: int) -> np.ndarray:
    """Xhat_t = X_{sigma(t)}, sigma(t) = last invocation whose result is delivered by t."""
    T = len(g)
    src = np.full(T, -1, dtype=np.int64)          # slot -> source frame of the delivery
    idx = np.flatnonzero(sampled)
    tgt = idx + d
    ok = tgt < T
    src[tgt[ok]] = idx[ok]
    pos = np.where(src >= 0, np.arange(T), -1)
    pos = np.maximum.accumulate(pos)              # last delivery slot at or before t
    out = np.zeros(T, dtype=np.int8)
    has = pos >= 0
    out[has] = g[src[pos[has]]]
    return out


def aoii_trace(xh: np.ndarray, g: np.ndarray) -> np.ndarray:
    """A_t = (A_{t-1}+1) * 1{Xhat_t != X_t}; vectorised consecutive-run counter."""
    m = (xh != g).astype(np.int32)
    if m.size == 0:
        return m
    c = np.cumsum(m)
    reset = np.where(m == 0, c, 0)
    reset = np.maximum.accumulate(reset)
    return (c - reset) * m


def run_lengths_by_side(xh: np.ndarray, g: np.ndarray):
    """Maximal mismatch runs -> (length, side) with side in {'miss','false','mixed'}.
    'miss' = belief 0 while X = 1; 'false' = belief 1 while X = 0; a run that
    contains both kinds of slots is 'mixed' (it straddles an event boundary)."""
    mism = (xh != g)
    out, on = [], None
    for i, m in enumerate(mism):
        if m and on is None:
            on = i
        elif (not m) and on is not None:
            out.append((on, i - on)); on = None
    if on is not None:
        out.append((on, len(mism) - on))
    res = []
    for s, L in out:
        gm = g[s:s + L]
        side = "miss" if gm.all() else ("false" if (gm == 0).all() else "mixed")
        res.append((s, L, side))
    return res


def per_event_metrics(ev, g, sampled, d, W=W_WINDOW):
    """For each event (onset, D): detection latency, strict recall, window recall.

    strict recall  : some invocation u falls inside the event (onset <= u < onset+D),
                     so the detector actually observes X_u = 1.
    window recall  : some invocation u satisfies u < onset + max(D, W)  -- the
                     window-refresh protocol of the gated-skipping literature; it
                     does NOT require the invocation to land inside the event.
                     Reported only together with strict recall and miss probability.
    """
    idx = np.flatnonzero(sampled)
    T = len(g)
    out = []
    for on, D in ev:
        j = np.searchsorted(idx, on)
        nxt = idx[j] if j < len(idx) else T + 10 ** 9
        strict = nxt < on + D
        lat = (nxt + d - on) if strict else np.inf       # slots from onset to belief flip
        win = nxt < on + max(D, W)
        peak_miss = min(D, nxt + d - on) if strict else D
        out.append((D, strict, win, lat, peak_miss))
    return out


# ---------------------------------------------------------------- policies
def periodic(T: int, K: int, phase: int = 0) -> np.ndarray:
    s = np.zeros(T, dtype=bool)
    s[phase % K::K] = True
    return s


def s_gate(S: np.ndarray, tau: float) -> np.ndarray:
    return S >= tau


def capped(sampled: np.ndarray, K_cap: int) -> np.ndarray:
    """Capped gate: force a refresh whenever the gate has stayed closed for K_cap slots
    (a virtual invocation at t = -1 seeds the timer)."""
    out = sampled.copy()
    idx = np.flatnonzero(sampled)
    T = len(sampled)
    bounds = np.concatenate(([-1], idx, [T]))
    extra = []
    for p, q in zip(bounds[:-1], bounds[1:]):
        if q - p > K_cap:
            extra.append(np.arange(p + K_cap, q, K_cap))
    if extra:
        e = np.concatenate(extra)
        out[e[e < T]] = True
    return out


def closed_intervals(sampled: np.ndarray):
    """Lengths of the closed-gate stretches (slots between consecutive invocations)."""
    idx = np.flatnonzero(sampled)
    if len(idx) < 2:
        return np.array([], dtype=int)
    return np.diff(idx)


# ------------------------------------------------------------------- misc
def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    return path
