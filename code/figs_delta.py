#!/usr/bin/env python3
"""CP-delta1 figures (pdf + png, no jpeg).  seed = 42.
 figD1_n_vs_N : every mismatch run of periodic K=8 on the corpus at d = 2 (all phases):
                run length n vs source transitions N in (s-d, t], with Theorem 2's
                line N*K + d - 1 and the general-sampler bound (K-1) + N(K-1+d).
 figD2_price  : price of the certificate - randomised optimum mean AoII vs A for
                abar in {0.125, 0.25, 0.5} (HIST class), with periodic K* and the
                capped gate, d = 2; dashed = uncapped BEL reference.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from spike_lagged import runs, transitions  # noqa: E402

FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "font.family": "serif"})
K, D = 8, 2


def fig_n_vs_N():
    vids = C.load_videos()
    pairs = []
    for x in vids.values():
        g = x["g"]
        ctr = np.concatenate(([0], np.cumsum(transitions(g))))
        for phi in range(K):
            s = C.periodic(len(g), K, phi)
            st, L = runs(C.belief(s, g, D) != g)
            N = ctr[st + L] - ctr[np.maximum(st - D + 1, 0)]
            pairs.append(np.stack([N, L], 1))
    P = np.concatenate(pairs)
    Ns, ns = P[:, 0], P[:, 1]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    u, cnt = np.unique(P, axis=0, return_counts=True)
    ax.scatter(u[:, 0], u[:, 1], s=4 + 10 * np.log10(cnt), c="#3b6ea5", alpha=0.7, lw=0,
               label="measured runs (size ~ log count)")
    xx = np.arange(0, Ns.max() + 1)
    ax.plot(xx[1:], xx[1:] * K + D - 1, "k-", lw=0.9, label=r"Thm 2: $NK+d-1$")
    ax.plot(xx, (K - 1) + xx * (K - 1 + D), "k--", lw=0.7, label=r"any gap$\leq K$: $(K-1)+N(K-1+d)$")
    ax.set_xlabel(r"source transitions $N$ in $(s-d,t]$")
    ax.set_ylabel(r"mismatch run length $n$")
    ax.set_xlim(-0.3, min(Ns.max(), 8) + 0.3); ax.set_ylim(0, max(ns.max() + 3, 40))
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title(f"periodic K={K}, d={D}: {len(ns):,} runs, 0 above the line", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"figD1_n_vs_N.{ext}", dpi=300)
    tight = int((ns == Ns * K + D - 1).sum())
    viol = int((ns > np.where(Ns == 0, K - 1, Ns * K + D - 1)).sum())
    return dict(n_runs=int(len(ns)), violations=viol, tight_runs=tight, max_N=int(Ns.max()),
                max_n=int(ns.max()))


def fig_price():
    """Nested optimum J*(A) (summary_delta.json), breakpoint marked."""
    r = json.load(open(HERE.parent / "results" / "summary_delta.json", encoding="utf-8"))
    rows = [x for x in r["nested"] if x["d"] == D and x["feasible"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cols = {0.125: "#1b9e77", 0.25: "#d95f02", 0.5: "#7570b3"}
    for ab in (0.125, 0.25, 0.5):
        sel = sorted([x for x in rows if abs(x["abar"] - ab) < 1e-9], key=lambda x: x["A"])
        As = [x["A"] for x in sel]
        ax.plot(As, [x["J_opt_rnd"] for x in sel], "o-", c=cols[ab], ms=3, lw=1, label=rf"optimal, $\bar a$={ab}")
        ax.plot(As, [x["J_periodic"] for x in sel], "s:", c=cols[ab], ms=2.5, lw=0.7)
        ax.plot(As, [x["J_gate"] if x["J_gate"] else np.nan for x in sel], "^--", c=cols[ab], ms=2.5, lw=0.7)
        ax.axhline(sel[0]["J_inf"], c=cols[ab], lw=0.5, alpha=0.5)
        bp = [x for x in sel if x["price_vs_inf"] > 1e-9]
        for x in bp:
            ax.annotate(f"+{x['price_vs_inf']:.0%}", (x["A"], x["J_opt_rnd"]), fontsize=5.5,
                        xytext=(3, 3), textcoords="offset points", color=cols[ab])
    ax.set_xscale("log", base=2); ax.set_xticks([3, 7, 15, 31]); ax.set_xticklabels(["3", "7", "15", "31"])
    ax.set_yscale("log")
    ax.set_xlabel(r"certificate $A$ (sup AoII$_{\rm lag}\leq A$)")
    ax.set_ylabel("mean AoII (slots)")
    ax.plot([], [], "ks:", ms=2.5, lw=0.7, label="periodic (best K)")
    ax.plot([], [], "k^--", ms=2.5, lw=0.7, label="capped gate")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    ax.set_title(f"price of the certificate, d={D}; thin = uncapped", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"figD2_price_of_certificate.{ext}", dpi=300)


def fig_assumption():
    """Re-draw figD3 from markov_baseline.json in the paper's font (no recomputation)."""
    import json as _j
    F = _j.load(open(HERE.parent / "results" / "markov_baseline.json", encoding="utf-8"))["fig"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = [f["abar"] for f in F]
    for key, lab, sty in (("periodic", "periodic (best K)", "s:"),
                          ("markov_bc", "Markov-optimal (budget-corrected)", "d-."),
                          ("gate", "capped content gate", "^--"), ("hist", "history-threshold", "o-")):
        ax.plot(x, [f.get(key) if f.get(key) is not None else np.nan for f in F], sty, ms=3, lw=1, label=lab)
    ax.set_xlabel(r"activation budget $\bar a$"); ax.set_ylabel("mean AoII, held-out (slots)")
    ax.set_yscale("log")
    ax.set_title(f"price of the Markov assumption, A=15, d={D}", fontsize=7)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(top=0.32)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"figD3_price_of_assumption.{ext}", dpi=300)


def fig_sojourn():
    """Measured sojourn laws (events and silences) against geometric laws with the same
    means: empirical CCDFs on log-log axes (existing data, no new experiment)."""
    vids = C.load_videos()
    Dl = np.array([dd for x in vids.values() for _, dd in x["ev"]])
    Gl = np.concatenate([np.array(x["gaps"]) for x in vids.values() if x["gaps"]])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for arr, lab, col in ((Dl, "event durations", "#d95f02"), (Gl, "silences", "#1b9e77")):
        v = np.sort(arr)
        cc = 1.0 - np.arange(len(v)) / len(v)
        ax.step(v, cc, where="post", c=col, lw=1.1, label=f"measured {lab} (n={len(v)})")
        m = arr.mean(); xs = np.unique(np.geomspace(1, v.max(), 200).astype(int))
        ax.plot(xs, (1 - 1 / m) ** (xs - 1), ls=":", c=col, lw=0.9, label=f"geometric, same mean ({m:.1f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("sojourn length (slots)"); ax.set_ylabel(r"$\Pr[\mathrm{sojourn}\geq x]$")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax.set_title("measured sojourn laws are far from geometric", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"figD0_sojourn.{ext}", dpi=300)


def graphical_abstract():
    """Two panels: the certificate (n vs N) and the out-of-sample optimisation (TEST)."""
    import json as _j
    mk = _j.load(open(HERE.parent / "results" / "markov_baseline.json", encoding="utf-8"))["fig"]
    vids = C.load_videos()
    P = []
    for x in vids.values():
        g = x["g"]
        ctr = np.concatenate(([0], np.cumsum(transitions(g))))
        for phi in range(K):
            s = C.periodic(len(g), K, phi)
            st, L = runs(C.belief(s, g, D) != g)
            N = ctr[st + L] - ctr[np.maximum(st - D + 1, 0)]
            P.append(np.stack([N, L], 1))
    P = np.concatenate(P)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.3))
    u, cnt = np.unique(P, axis=0, return_counts=True)
    a1.scatter(u[:, 0], u[:, 1], s=4 + 10 * np.log10(cnt), c="#3b6ea5", alpha=0.7, lw=0)
    xx = np.arange(1, P[:, 0].max() + 1)
    a1.plot(xx, xx * K + D - 1, "k-", lw=1)
    a1.set_xlabel("source transitions N in the run"); a1.set_ylabel("run of incorrect belief")
    a1.set_title(r"certificate: $n\leq NK+d-1$ (tight)", fontsize=8)
    x = [f["abar"] for f in mk]
    for key, lab, sty in (("periodic", "periodic", "s:"), ("markov_bc", "Markov-optimal", "d-."),
                          ("gate", "capped gate", "^--"), ("hist", "history policy", "o-")):
        a2.plot(x, [f.get(key) if f.get(key) is not None else np.nan for f in mk], sty, ms=3, lw=1, label=lab)
    a2.set_yscale("log"); a2.set_xlabel(r"activation budget $\bar a$"); a2.set_ylabel("mean AoII, held-out")
    a2.set_title("optimal sampling under the certificate", fontsize=8)
    a2.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"graphical_abstract_delta.{ext}", dpi=300)


if __name__ == "__main__":
    info = fig_n_vs_N()
    print(info)
    C.jdump(info, HERE.parent / "results" / "figD1_n_vs_N.json")
    if (HERE.parent / "results" / "summary_delta.json").exists():
        fig_price()
    if (HERE.parent / "results" / "markov_baseline.json").exists():
        fig_assumption()
    fig_sojourn()
