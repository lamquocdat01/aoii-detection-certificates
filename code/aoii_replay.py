#!/usr/bin/env python3
"""B5/B6 trace-driven replay of symmetric AoII over the 121-video corpus. seed = 42.

Policy zoo (decision Q-A):  periodic K in {2,4,8,16} (every phase),
S-gate (global score threshold), capped S-gate (K_cap in {8,16,32}).
Delays d in {0,1,2,3}; d = 2 is the nominal operating point (Q-B).
Output: results/aoii_scores.json
"""
from __future__ import annotations
import numpy as np
import common as C
import theory as TH

KS = (2, 4, 8, 16)
DS = (0, 1, 2, 3)
D_NOM = 2
KCAPS = (8, 16, 32)
A_GRID = (0.05, 0.125, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)


# ------------------------------------------------------------------ scoring
def score_policy(vids, make_sampler, d, W=C.W_WINDOW):
    """make_sampler: (key, video_dict) -> bool array. Returns pooled AoII statistics."""
    tot_slots = 0
    sum_all = sum_miss = sum_false = 0.0
    mx = mx_miss = mx_false = mx_mixed = 0
    n_act = 0
    allA = []
    ev_rows = []
    runs_by_side = {"miss": 0, "false": 0, "mixed": 0}
    for k, x in vids.items():
        g = x["g"]
        s = make_sampler(k, x)
        xh = C.belief(s, g, d)
        A = C.aoii_trace(xh, g)
        tot_slots += len(g)
        n_act += int(s.sum())
        sum_all += float(A.sum())
        sum_miss += float(A[g == 1].sum())
        sum_false += float(A[g == 0].sum())
        mx = max(mx, int(A.max()) if len(A) else 0)
        for st, L, side in C.run_lengths_by_side(xh, g):
            runs_by_side[side] += 1
            if side == "miss":
                mx_miss = max(mx_miss, L)
            elif side == "false":
                mx_false = max(mx_false, L)
            else:
                mx_mixed = max(mx_mixed, L)
        allA.append(A)
        ev_rows += C.per_event_metrics(x["ev"], g, s, d, W)
    A = np.concatenate(allA)
    cols = [np.array([r[i] for r in ev_rows], dtype=float) for i in range(5)]
    D_, strict, win, lat, peak = cols
    det = np.isfinite(lat)
    return dict(
        activation=n_act / tot_slots,
        max_aoii=mx, mean_aoii=sum_all / tot_slots,
        p99_aoii=float(np.percentile(A, 99)), p999_aoii=float(np.percentile(A, 99.9)),
        mean_miss_side=sum_miss / tot_slots, mean_false_side=sum_false / tot_slots,
        max_miss_run=mx_miss, max_false_run=mx_false, max_mixed_run=mx_mixed,
        n_runs=runs_by_side,
        recall_strict=float(strict.mean()), recall_W=float(win.mean()),
        miss_probability=float(1 - strict.mean()),
        mean_detection_latency_slots=(float(lat[det].mean()) if det.any() else None),
        max_peak_miss_aoii=float(peak.max()), mean_peak_miss_aoii=float(peak.mean()),
        n_events=int(len(D_)), n_slots=int(tot_slots))


def iso_recall_tau(vids, allS, target, key, lo=0.005, hi=1.0, iters=18):
    """Smallest activation whose recall (`key`) reaches `target`; bisection on a."""
    def rec(a):
        tau = np.quantile(allS, 1 - a)
        r = score_policy(vids, lambda k, x: x["S"] >= tau, 0)
        return r[key], float(tau), r["activation"]
    if rec(hi)[0] < target:
        return None
    a_lo, a_hi = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a_lo + a_hi)
        if rec(mid)[0] >= target:
            a_hi = mid
        else:
            a_lo = mid
    r, tau, act = rec(a_hi)
    return dict(a_target=a_hi, tau=tau, activation=act, recall=r)


def main():
    vids = C.load_videos()
    allS = np.concatenate([x["S"] for x in vids.values()])
    out = {"seed": C.SEED, "W_window": C.W_WINDOW, "slot_ms": C.SLOT_MS,
           "delays": list(DS), "d_nominal": D_NOM, "policies": {}}

    # -------- periodic, every phase
    keys = ("max_aoii", "mean_aoii", "p99_aoii", "mean_miss_side", "mean_false_side",
            "max_miss_run", "max_false_run", "max_mixed_run", "activation",
            "recall_strict", "recall_W", "miss_probability", "max_peak_miss_aoii",
            "mean_peak_miss_aoii")
    for K in KS:
        for d in DS:
            per_phase = {}
            for phi in range(K):
                per_phase[phi] = score_policy(
                    vids, lambda k, x, K=K, phi=phi: C.periodic(len(x["g"]), K, phi), d)
            agg = {f"{k}_sup": max(per_phase[p][k] for p in per_phase) for k in keys}
            agg.update({f"{k}_mean_phase": float(np.mean([per_phase[p][k] for p in per_phase]))
                        for k in keys})
            agg["bound_d0_K_minus_1"] = K - 1
            agg["nominal_K_minus_1_plus_d"] = K - 1 + d
            agg["per_phase"] = {str(p): per_phase[p] for p in per_phase}
            out["policies"][f"periodic_K{K}_d{d}"] = agg

    # -------- S-gate on a fixed activation grid
    for a in A_GRID:
        tau = float(np.quantile(allS, 1 - a))
        for d in DS:
            r = score_policy(vids, lambda k, x, tau=tau: x["S"] >= tau, d)
            r["tau"] = tau
            r["a_target"] = a
            out["policies"][f"sgate_a{a}_d{d}"] = r

    # -------- S-gate iso-recall 90 %
    iso = {"strict": iso_recall_tau(vids, allS, 0.90, "recall_strict"),
           "W": iso_recall_tau(vids, allS, 0.90, "recall_W")}
    out["iso_recall90"] = iso
    for name, rec in iso.items():
        if rec is None:
            continue
        tau = rec["tau"]
        for d in DS:
            r = score_policy(vids, lambda k, x, tau=tau: x["S"] >= tau, d)
            r["tau"] = tau
            out["policies"][f"sgate_iso90{name}_d{d}"] = r
        for Kc in KCAPS:
            for d in DS:
                r = score_policy(vids, lambda k, x, tau=tau, Kc=Kc: C.capped(x["S"] >= tau, Kc), d)
                r["tau"] = tau
                r["K_cap"] = Kc
                r["bound_d0_Kcap_minus_1"] = Kc - 1
                r["nominal_Kcap_minus_1_plus_d"] = Kc - 1 + d
                out["policies"][f"cappedsgate_iso90{name}_Kcap{Kc}_d{d}"] = r

    # -------- capped gate at the low-activation operating point a = 0.125
    tau125 = float(np.quantile(allS, 1 - 0.125))
    for Kc in KCAPS:
        for d in DS:
            r = score_policy(vids, lambda k, x, tau=tau125, Kc=Kc: C.capped(x["S"] >= tau, Kc), d)
            r["tau"] = tau125
            r["K_cap"] = Kc
            r["bound_d0_Kcap_minus_1"] = Kc - 1
            r["nominal_Kcap_minus_1_plus_d"] = Kc - 1 + d
            out["policies"][f"cappedsgate_a0.125_Kcap{Kc}_d{d}"] = r

    # -------- closed-gate interval distribution of the S-gate (Theorem 2 evidence)
    gapdist = {}
    cand = [(n, r) for n, r in iso.items() if r] + [("a0.125", {"tau": tau125})]
    for name, rec in cand:
        tau = rec["tau"]
        parts = [C.closed_intervals(x["S"] >= tau) for x in vids.values()]
        ci = np.concatenate([p for p in parts if p.size])
        gapdist[name] = dict(tau=float(tau), n=int(len(ci)), mean=float(ci.mean()),
                             median=float(np.median(ci)), p99=float(np.percentile(ci, 99)),
                             max=int(ci.max()), cv=float(ci.std() / ci.mean()),
                             frac_gt_8=float((ci > 8).mean()), frac_gt_32=float((ci > 32).mean()))
    out["closed_gate_intervals"] = gapdist

    # -------- B6: certificate vs empirical
    b6 = []
    for K in KS:
        for d in DS:
            p = out["policies"][f"periodic_K{K}_d{d}"]
            bd = K - 1 if d == 0 else K - 1 + d
            b6.append(dict(
                policy=f"periodic_K{K}", d=d,
                certificate=("K-1 (unconditional)" if d == 0
                             else "K-1+d (conditional on min sojourn >= K+d+1)"),
                bound=bd, empirical=p["max_aoii_sup"], holds=bool(p["max_aoii_sup"] <= bd),
                tightness=(p["max_aoii_sup"] / bd if bd else None), conditional=bool(d > 0)))
    for Kc in KCAPS:
        for d in DS:
            for stem in [f"cappedsgate_iso90W_Kcap{Kc}_d{d}", f"cappedsgate_a0.125_Kcap{Kc}_d{d}"]:
                if stem not in out["policies"]:
                    continue
                p = out["policies"][stem]
                bd = Kc - 1 if d == 0 else Kc - 1 + d
                b6.append(dict(
                    policy=stem.rsplit("_d", 1)[0], d=d,
                    certificate=("K_cap-1 (unconditional)" if d == 0 else "K_cap-1+d (conditional)"),
                    bound=bd, empirical=p["max_aoii"], holds=bool(p["max_aoii"] <= bd),
                    tightness=p["max_aoii"] / bd, conditional=bool(d > 0)))
    out["b6_certificate_vs_empirical"] = b6
    t0 = [r["tightness"] for r in b6 if r["d"] == 0 and r["tightness"]]
    out["b6_summary"] = dict(
        n_checks=len(b6),
        unconditional_all_hold=all(r["holds"] for r in b6 if not r["conditional"]),
        unconditional_violations=[r for r in b6 if not r["conditional"] and not r["holds"]],
        conditional_violations=[r for r in b6 if r["conditional"] and not r["holds"]],
        min_tightness_d0=min(t0), max_tightness_d0=max(t0))

    # -------- source-regularity diagnostics
    D_all = np.array([dd for x in vids.values() for _, dd in x["ev"]], dtype=np.int64)
    G_all = np.concatenate([np.array(x["gaps"], dtype=np.int64) for x in vids.values() if x["gaps"]])
    out["sojourn_condition"] = {f"K{K}_d{d}": TH.sojourn_condition(D_all, G_all, K, d)
                                for K in KS for d in DS}
    C.jdump(out, C.RESULTS / "aoii_scores.json")

    s = out["b6_summary"]
    print(f"B6 checks: {s['n_checks']}  unconditional all hold: {s['unconditional_all_hold']}  "
          f"tightness(d=0) in [{s['min_tightness_d0']:.3f}, {s['max_tightness_d0']:.3f}]")
    print(f"conditional (d>=1) nominal-bound violations: {len(s['conditional_violations'])}")
    return out


if __name__ == "__main__":
    main()
