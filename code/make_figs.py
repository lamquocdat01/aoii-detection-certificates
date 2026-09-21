#!/usr/bin/env python3
"""B7 figures (PDF + PNG, never JPEG).  seed = 42.
Fig 1  symmetric AoII sawtooth on one real trace (dataset/video/frames recorded)
  Fig 2  energy-AoII frontier with bootstrap CI, extrapolation and feasibility marks
  Fig 3  certificate vs realised worst case (tightness)
  Fig 4  heavy-tail sensitivity, full corpus vs sparse subset
  graphical_abstract.(pdf|png)
Also writes results/fig_provenance.json.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as C

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", grey="#666666", sky="#56B4E9")
W1, W2 = 3.45, 7.16          # IEEE single / double column width (inches)

# Fig 1 trace: the only window in the corpus containing exactly one event of
# 12..40 slots with >= 40 slots of silence before and >= 60 after.
TRACE = ("CDnet2014", "intermittentPan")
TRACE_ONSET_IDX, TRACE_D, TRACE_PRE, TRACE_POST = 1257, 35, 40, 60


def save(fig, name, prov, meta):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIGURES / f"{name}.{ext}")
    plt.close(fig)
    prov[name] = meta
    print(f"  wrote figures/{name}.pdf + .png")


def fig1(vids, prov):
    x = vids[TRACE]
    lo, hi = TRACE_ONSET_IDX - TRACE_PRE, TRACE_ONSET_IDX + TRACE_D + TRACE_POST
    g = x["g"][lo:hi]
    t = np.arange(lo, hi) + x["frame0"]
    fig, axes = plt.subplots(3, 1, figsize=(W1, 3.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.6, 1, 1], hspace=0.18))
    axes[0].step(t, g, where="post", color=CB["grey"], lw=1.0)
    axes[0].fill_between(t, 0, g, step="post", color=CB["grey"], alpha=0.18)
    axes[0].set_ylabel(r"$X_t$"); axes[0].set_yticks([0, 1]); axes[0].set_ylim(-0.15, 1.3)
    for ax, d, col in ((axes[1], 0, CB["blue"]), (axes[2], 2, CB["red"])):
        s = C.periodic(len(x["g"]), 8, 0)
        xh = C.belief(s, x["g"], d)[lo:hi]
        A = C.aoii_trace(C.belief(s, x["g"], d), x["g"])[lo:hi]
        miss = (g == 1) & (xh == 0)
        fals = (g == 0) & (xh == 1)
        ax.fill_between(t, 0, A, where=miss, step="post", color=CB["orange"], alpha=0.55, lw=0,
                        label="miss side")
        ax.fill_between(t, 0, A, where=fals, step="post", color=CB["sky"], alpha=0.75, lw=0,
                        label="false side")
        ax.step(t, A, where="post", color=col, lw=1.0)
        bound = 7 if d == 0 else None
        if bound:
            ax.axhline(bound, color=CB["green"], ls="--", lw=0.8)
            ax.text(t[2], bound + 0.4, r"certificate $K-1=7$", color=CB["green"], fontsize=6.5)
        ax.set_ylabel(r"$A_t$ (slots)")
        ax.text(0.985, 0.88, rf"$d={d}$", transform=ax.transAxes, ha="right", fontsize=7)
    axes[2].set_xlabel(f"frame index ({TRACE[0]}/{TRACE[1]})")
    axes[1].legend(loc="upper left", frameon=False, ncol=2, handlelength=1.2,
                   bbox_to_anchor=(-0.02, 0.80))
    save(fig, "fig1_sawtooth", prov,
         dict(dataset=TRACE[0], video=TRACE[1], frames=[int(t[0]), int(t[-1])],
              policy="periodic K=8, phase 0", delays=[0, 2],
              event_onset_frame=int(x["frame0"] + TRACE_ONSET_IDX), event_duration=TRACE_D))


def fig2(scores, energy, prov):
    """Energy--AoII frontier: measured device energy per slot against the realised
worst-case symmetric AoII of each policy (d = 0)."""
    P = scores["policies"]
    fit = energy["fit"]
    a_sweep = energy["sweep_a_max"]
    a_ort = energy["throughput"]["a_max_ort"]
    E = lambda a: (fit["W0"] + fit["slope_W"] * a) / C.FPS * 1000.0

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.axvspan(E(a_sweep), E(1.0), color=CB["grey"], alpha=0.10, lw=0)
    ax.text(E(1.0) - 3, 2600, "extrapolated ($a>0.5$)", fontsize=6, color=CB["grey"],
            ha="right", va="top")
    ax.axvline(E(a_ort), color=CB["purple"], ls=":", lw=0.9)
    ax.text(E(a_ort) - 3, 900, "not real-time\nfeasible (ORT)", rotation=90, fontsize=6,
            color=CB["purple"], ha="right", va="top")

    per = [(P[f"periodic_K{K}_d0"]["activation_sup"], P[f"periodic_K{K}_d0"]["max_aoii_sup"], K)
           for K in (16, 8, 4, 2)]
    ax.plot([E(a) for a, _, _ in per], [y for _, y, _ in per], color=CB["blue"], lw=1.0,
            marker="o", ms=4, label="periodic $K$ (certified)")
    for a, y, K in per:
        ax.annotate(f"$K$={K}", (E(a), y), textcoords="offset points", xytext=(3, 4),
                    fontsize=6, color=CB["blue"])

    cap = [(P[f"cappedsgate_iso90W_Kcap{Kc}_d0"]["activation"],
            P[f"cappedsgate_iso90W_Kcap{Kc}_d0"]["max_aoii"], Kc) for Kc in (32, 16, 8)]
    ax.plot([E(a) for a, _, _ in cap], [y for _, y, _ in cap], color=CB["green"], lw=1.0,
            marker="^", ms=4, ls="-", label="capped S-gate (certified)")
    for a, y, Kc in cap:
        ax.annotate(rf"$K_{{cap}}$={Kc}", (E(a), y), textcoords="offset points", xytext=(4, -1),
                    fontsize=6, color=CB["green"])

    for nm, lab, off in (("sgate_a0.125_d0", "$a$=0.125", (7, 0)),
                         ("sgate_iso90W_d0", "iso-recall 90%", (7, 0))):
        p = P[nm]
        ax.scatter([E(p["activation"])], [p["max_aoii"]], color=CB["red"], marker="s", s=26,
                   zorder=6, edgecolor="white", linewidth=0.5,
                   label="uncapped S-gate (no certificate)" if nm.endswith("0.125_d0") else None)
        ax.annotate(lab + "\n" + f"{p['max_aoii']:g}", (E(p["activation"]), p["max_aoii"]),
                    textcoords="offset points", xytext=off, fontsize=6, color=CB["red"],
                    ha="left", va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 4000)
    ax.set_xlabel("measured device energy per slot (mJ)")
    ax.set_ylabel(r"realised $\sup_t A_t$ (slots)")
    ax.legend(loc="center left", frameon=False, handlelength=1.4,
              bbox_to_anchor=(0.01, 0.62))
    save(fig, "fig2_frontier", prov,
         dict(model=f"P(a) = {fit['W0']:.4f} + {fit['slope_W']:.4f} a  W", r2=fit["r2"],
              sweep_a_max=a_sweep, a_max_ort=a_ort, delay=0,
              periodic=[dict(K=K, a=a, sup_aoii=y) for a, y, K in per],
              capped=[dict(K_cap=Kc, a=a, sup_aoii=y) for a, y, Kc in cap],
              uncapped=[dict(name=n, a=P[n]["activation"], sup_aoii=P[n]["max_aoii"])
                        for n in ("sgate_a0.125_d0", "sgate_iso90W_d0")]))


def fig3(scores, prov):
    rows = [r for r in scores["b6_certificate_vs_empirical"] if r["d"] == 0]
    rows.sort(key=lambda r: r["bound"])
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    b = np.array([r["bound"] for r in rows], dtype=float)
    e = np.array([r["empirical"] for r in rows], dtype=float)
    lab = [r["policy"] for r in rows]
    ax.plot([1, 40], [1, 40], color=CB["grey"], lw=0.8, ls="--", label="certificate = realised")
    per = [i for i, l in enumerate(lab) if l.startswith("periodic")]
    cap = [i for i, l in enumerate(lab) if l.startswith("capped")]
    ax.scatter(b[per], e[per], color=CB["blue"], marker="o", s=22, label="periodic $K$", zorder=4)
    ax.scatter(b[cap], e[cap], color=CB["green"], marker="^", s=22, label="capped S-gate", zorder=4)
    for nm, col in (("sgate_iso90W_d0", CB["red"]), ("sgate_a0.125_d0", CB["orange"])):
        p = scores["policies"][nm]
        ax.axhline(p["max_aoii"], color=col, lw=0.8, ls=":")
        ax.text(1.1, p["max_aoii"] * 1.05,
                ("uncapped S-gate, iso-recall 90%: " if "iso" in nm else "uncapped S-gate, $a$=0.125: ")
                + f"{p['max_aoii']:g}", color=col, fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"certificate $K-1$ (resp. $K_{cap}-1$), slots")
    ax.set_ylabel(r"realised $\sup_t A_t$, slots")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.02, 0.62))
    save(fig, "fig3_bound_vs_empirical", prov,
         dict(n_points=len(rows), tightness_min=scores["b6_summary"]["min_tightness_d0"],
              tightness_max=scores["b6_summary"]["max_tightness_d0"], delay=0))


def fig4(ht, prov):
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.80, 2.3))
    fig.subplots_adjust(wspace=0.55)
    Ks = [2, 4, 8, 16, 32]
    for ax, scope in zip(axes, ("full", "sparse")):
        r = ht["scopes"][scope]
        lo = [r["by_K"][str(K)]["lower"] for K in Ks]
        ex = [r["by_K"][str(K)]["exact"] for K in Ks]
        up = [r["by_K"][str(K)]["upper"] for K in Ks]
        ge = [r["by_K"][str(K)]["geometric_exact"] for K in Ks]
        ax.fill_between(Ks, lo, up, color=CB["blue"], alpha=0.18, lw=0, label="two-sided bound")
        ax.plot(Ks, ex, color=CB["blue"], marker="o", ms=3, lw=1.0, label="measured law")
        ax.plot(Ks, ge, color=CB["orange"], marker="s", ms=3, lw=1.0, ls="--",
                label="memoryless, same mean")
        ax2 = ax.twinx()
        mp = [r["by_K"][str(K)]["miss_probability"] for K in Ks]
        ax2.plot(Ks, mp, color=CB["purple"], marker="^", ms=3, lw=0.9, ls=":",
                 label="miss probability")
        ax2.set_ylim(0, 1); ax2.grid(False)
        ax2.set_ylabel("miss probability", color=CB["purple"], fontsize=7)
        ax2.tick_params(axis="y", colors=CB["purple"], labelsize=6.5)
        if scope == "full":
            ax2.set_ylabel("")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("sampling period $K$ (slots)")
        d = r["duration_law"]
        ax.set_title(f"{scope} corpus  (CV {d['cv']:.2f}, median {d['median']:.0f})", fontsize=7)
    axes[0].set_ylabel("mean symmetric AoII (slots)")
    axes[0].legend(loc="upper left", frameon=False)
    save(fig, "fig4_heavy_tail", prov,
         dict(scopes=["full", "sparse"], K=Ks,
              note="left axis: mean AoII (log); right axis: P(event never sampled)"))


def graphical_abstract(scores, energy, prov):
    P = scores["policies"]
    fit = energy["fit"]
    E = lambda a: (fit["W0"] + fit["slope_W"] * a) / C.FPS * 1000.0
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    names = [("periodic_K8_d0", "periodic $K$=8", CB["blue"], "max_aoii_sup", "activation_sup"),
             ("cappedsgate_iso90W_Kcap8_d0", "capped S-gate ($K_{cap}$=8)", CB["green"], "max_aoii", "activation"),
             ("sgate_iso90W_d0", "content-driven S-gate", CB["red"], "max_aoii", "activation")]
    xs, ys, cs, ls = [], [], [], []
    recs = []
    for n, lab, col, kA, kact in names:
        p = P[n]
        xs.append(E(p[kact])); ys.append(max(p[kA], 0.5)); cs.append(col); ls.append(lab)
        rw = p.get("recall_W", p.get("recall_W_mean_phase"))
        rs = p.get("recall_strict", p.get("recall_strict_mean_phase"))
        recs.append((rw, rs))
    ax.scatter(xs, ys, c=cs, s=70, zorder=5, edgecolor="white", linewidth=0.8)
    for x, y, l, c in zip(xs, ys, ls, cs):
        ax.annotate(f"{l}\n" + r"$\sup_t A_t$ = " + f"{y:g} slots", (x, y),
                    textcoords="offset points", xytext=(8, 0), fontsize=7.5, color=c, va="center")
    ax.set_yscale("log")
    ax.set_xlim(150, 470); ax.set_ylim(3, 4000)
    ax.set_xlabel("measured device energy per slot (mJ), Jetson Orin Nano")
    ax.set_ylabel(r"worst-case symmetric AoII (slots)")
    ax.set_title("A worst-case AoII certificate exists only when the closed-gate interval is bounded",
                 fontsize=8)
    save(fig, "graphical_abstract", prov, dict(points=list(zip(ls, xs, ys))))


def main():
    vids = C.load_videos()
    scores = json.load(open(C.RESULTS / "aoii_scores.json", encoding="utf-8"))
    energy = json.load(open(C.RESULTS / "energy_frontier.json", encoding="utf-8"))
    ht = json.load(open(C.RESULTS / "heavy_tail.json", encoding="utf-8"))
    prov = {}
    fig1(vids, prov)
    fig2(scores, energy, prov)
    fig3(scores, prov)
    fig4(ht, prov)
    graphical_abstract(scores, energy, prov)
    C.jdump(prov, C.RESULTS / "fig_provenance.json")
    return prov


if __name__ == "__main__":
    main()
