#!/usr/bin/env python3
"""B7 energy--AoII frontier.  Refits the affine device power model from the raw 10 Hz
INA logs of the Jetson Orin Nano runs and attaches a moving-block bootstrap CI.
seed = 42.   Output: results/energy_frontier.json

Measurement protocol as reported in the paper:
  onboard INA3221 rails read through tegrastats at 10 Hz, 300 s per run after a 60 s
  warm-up, n = 3 runs for the idle and always-on points, n = 1 run per duty-cycle
  sweep point (k in {2,5,10} -> a = 1/k).  Slot = 1/30 s.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import common as C

RNG = np.random.RandomState(C.SEED)
B_BOOT = 10000
BLOCK = 50            # 5 s moving blocks (INA logs are strongly autocorrelated)
SWEEP = {"sweep_yolo26s_k2": 1 / 2, "sweep_yolo26s_k5": 1 / 5, "sweep_yolo26s_k10": 1 / 10}
IDLE = "idle_yolo26s"


def measure_samples(tag, runs=(1, 2, 3)):
    out = []
    for r in runs:
        f = C.C1_LOGS / f"{tag}_run{r}.csv"
        if f.exists():
            d = pd.read_csv(f)
            out.append(d[d.phase == "measure"].watt.values.astype(float))
    return out


def block_boot_mean(runs, rng):
    """One moving-block bootstrap replicate of the pooled mean of a design point."""
    means = []
    for x in runs:
        n = len(x)
        nb = int(np.ceil(n / BLOCK))
        starts = rng.randint(0, n - BLOCK + 1, size=nb)
        samp = np.concatenate([x[s:s + BLOCK] for s in starts])[:n]
        means.append(samp.mean())
    return float(np.mean(means))


def fit(a, w):
    s, w0 = np.polyfit(a, w, 1)
    pred = w0 + s * a
    r2 = 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum()
    return float(w0), float(s), float(r2)


def main():
    csv = pd.read_csv(C.C1_CSV)
    pts = {0.0: measure_samples(IDLE)}
    for tag, a in SWEEP.items():
        pts[a] = measure_samples(tag, runs=(1,))
    avals = np.array(sorted(pts))
    wvals = np.array([np.mean([x.mean() for x in pts[a]]) for a in avals])
    n_runs = {float(a): len(pts[a]) for a in avals}
    w0, s, r2 = fit(avals, wvals)

    boots = np.empty((B_BOOT, 2))
    for b in range(B_BOOT):
        wb = np.array([block_boot_mean(pts[a], RNG) for a in avals])
        boots[b, 0], boots[b, 1], _ = fit(avals, wb)[:2] + (0,)
    ci = lambda col: (float(np.percentile(boots[:, col], 2.5)), float(np.percentile(boots[:, col], 97.5)))
    u_w0, u_s = float(boots[:, 0].std(ddof=1)), float(boots[:, 1].std(ddof=1))

    slot_s = 1.0 / C.FPS
    E = lambda a, W0=w0, S=s: (W0 + S * a) * slot_s * 1000.0        # mJ per slot

    def E_ci(a):
        v = (boots[:, 0] + boots[:, 1] * a) * slot_s * 1000.0
        return dict(mJ_per_slot=E(a), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    u=float(v.std(ddof=1)), U_k2=2 * float(v.std(ddof=1)))

    ort = csv[csv.tag == "active_yolo26s"]
    trt = csv[csv.tag == "active_trt_yolo26s"]
    thr = dict(ort_cuda_infer_fps=float(ort.infer_fps.mean()),
               trt_fp16_infer_fps=float(trt.infer_fps.mean()),
               a_max_ort=float(ort.infer_fps.mean() / C.FPS),
               a_max_trt=float(trt.infer_fps.mean() / C.FPS),
               latency_ms_ort=float(1000.0 / ort.infer_fps.mean()),
               latency_ms_trt=float(1000.0 / trt.infer_fps.mean()),
               slot_ms=C.SLOT_MS)
    thr["d_slots_ort"] = int(np.ceil(thr["latency_ms_ort"] / C.SLOT_MS))
    thr["d_slots_trt"] = int(np.ceil(thr["latency_ms_trt"] / C.SLOT_MS))

    # run-to-run spread where n = 3 (the bootstrap only sees WITHIN-run variability;
    # at the n = 1 sweep points the run-to-run component is unquantified)
    rr = {}
    for tag in (IDLE, "active_yolo26s", "active_trt_yolo26s"):
        w = csv[csv.tag == tag].watts_mean.values.astype(float)
        if len(w) > 1:
            rr[tag] = dict(n=len(w), mean=float(w.mean()), sd=float(w.std(ddof=1)),
                           cv=float(w.std(ddof=1) / w.mean()))
    a_sweep_max = float(max(SWEEP.values()))
    out = dict(
        seed=C.SEED, n_bootstrap=B_BOOT, block_len_samples=BLOCK, sampling_hz=10,
        design_points=[dict(a=float(a), watts=float(w), n_runs=n_runs[float(a)],
                            n_samples=int(sum(len(x) for x in pts[a]))) for a, w in zip(avals, wvals)],
        fit=dict(W0=w0, slope_W=s, r2=r2, n_points=len(avals),
                 W0_ci95=ci(0), slope_ci95=ci(1), u_W0=u_w0, u_slope=u_s,
                 U_k2_W0=2 * u_w0, U_k2_slope=2 * u_s),
        sweep_a_max=a_sweep_max, throughput=thr,
        run_to_run=dict(points=rr,
                        caveat=("The bootstrap CI covers within-run 10 Hz variability only. "
                                "Where n = 3 runs exist, the run-to-run CV of mean power is "
                                "up to %.2f %%, an order of magnitude above the bootstrap CI; "
                                "at the n = 1 duty-cycle sweep points this component is "
                                "unquantified, so the reported CI understates the true "
                                "uncertainty." % (100 * max(v["cv"] for v in rr.values())))),
        energy=dict((f"a={a}", dict(extrapolated=bool(a > a_sweep_max),
                                    realtime_feasible_ort=bool(a <= thr["a_max_ort"]),
                                    realtime_feasible_trt=bool(a <= thr["a_max_trt"]),
                                    **E_ci(a)))
                    for a in (0.0, 0.0628, 0.125, 0.25, 0.5, 0.77, 0.78, 0.8067, 0.8852, 1.0)))

    # headline ratios (total device energy, and the part above idle)
    def ratio(a_hi, a_lo):
        tot = (w0 + s * a_hi) / (w0 + s * a_lo)
        marg = (s * a_hi) / (s * a_lo)
        v = ((boots[:, 0] + boots[:, 1] * a_hi) / (boots[:, 0] + boots[:, 1] * a_lo))
        return dict(a_hi=a_hi, a_lo=a_lo, ratio_total=float(tot),
                    ratio_total_ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    ratio_above_idle=float(marg))
    # operating points come from the replay when it has been run
    import json as _json
    sp = C.RESULTS / "aoii_scores.json"
    if sp.exists():
        _P = _json.load(open(sp, encoding="utf-8"))["policies"]
        a_per = _P["periodic_K8_d0"]["activation_sup"]
        a_isoW = _P["sgate_iso90W_d0"]["activation"]
        a_isoS = _P["sgate_iso90strict_d0"]["activation"]
        a_cap = _P["cappedsgate_iso90W_Kcap8_d0"]["activation"]
    else:
        a_per, a_isoW, a_isoS, a_cap = 0.1253, 0.77, 0.78, 0.7886
    out["headline_ratio"] = dict(
        gate_iso90W_vs_periodicK8=ratio(a_isoW, a_per),
        gate_iso90strict_vs_periodicK8=ratio(a_isoS, a_per),
        cappedgate_Kcap8_vs_periodicK8=ratio(a_cap, a_per),
        operating_points=dict(periodic_K8=a_per, sgate_iso90W=a_isoW,
                              sgate_iso90strict=a_isoS, capped_Kcap8=a_cap))
    C.jdump(out, C.RESULTS / "energy_frontier.json")
    f = out["fit"]
    print(f"P(a) = {f['W0']:.4f} [{f['W0_ci95'][0]:.4f},{f['W0_ci95'][1]:.4f}] "
          f"+ {f['slope_W']:.4f} [{f['slope_ci95'][0]:.4f},{f['slope_ci95'][1]:.4f}] * a  W, R2={f['r2']:.5f}")
    print(f"E(a=0.125) = {out['energy']['a=0.125']['mJ_per_slot']:.1f} mJ/slot "
          f"(U_k2 = {out['energy']['a=0.125']['U_k2']:.1f})")
    print(f"headline total-energy ratio gate/periodic = "
          f"{out['headline_ratio']['gate_iso90W_vs_periodicK8']['ratio_total']:.3f}")
    print("run-to-run CV (n=3 points): " +
          ", ".join(f"{k}:{v['cv']*100:.2f}%" for k, v in out["run_to_run"]["points"].items()))
    print(f"throughput ceiling: a_max ORT = {thr['a_max_ort']:.3f}, TRT = {thr['a_max_trt']:.3f}; "
          f"d = {thr['d_slots_ort']} (ORT) / {thr['d_slots_trt']} (TRT) slots")
    return out


if __name__ == "__main__":
    main()
