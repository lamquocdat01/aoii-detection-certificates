#!/usr/bin/env python3
"""B5 self-contained corpus description: one row per source dataset (DOI, videos,
events, frames, in-event fraction, duration law) plus the measurement protocol of the
Jetson Orin Nano runs used in Sections VI-VII.  seed = 42.
Output: results/corpus_table.json
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import common as C


def law(D):
    D = np.asarray(D, dtype=float)
    return dict(n_events=int(len(D)), mean=float(D.mean()), median=float(np.median(D)),
                cv=float(D.std() / D.mean()), frac_single=float((D == 1).mean()),
                frac_lt5=float((D < 5).mean()), frac_lt8=float((D < 8).mean()),
                p99=float(np.percentile(D, 99)), max=float(D.max()))


def main():
    vids = C.load_videos()
    rows = []
    for ds in C.DATASETS:
        ks = [k for k in vids if k[0] == ds]
        D = np.array([d for k in ks for _, d in vids[k]["ev"]], dtype=np.int64)
        n_fr = sum(len(vids[k]["g"]) for k in ks)
        in_ev = float(np.concatenate([vids[k]["g"] for k in ks]).mean())
        rows.append(dict(dataset=ds, doi=C.DATASET_DOI[ds], n_videos=len(ks),
                         n_frames=int(n_fr), in_event_fraction=in_ev, **law(D)))
    D_all = np.array([d for x in vids.values() for _, d in x["ev"]], dtype=np.int64)
    total = dict(dataset="Total", doi="", n_videos=len(vids),
                 n_frames=int(sum(len(x["g"]) for x in vids.values())),
                 in_event_fraction=float(np.concatenate([x["g"] for x in vids.values()]).mean()),
                 **law(D_all))
    sparse = [k for k, x in vids.items() if x["g"].mean() < 0.3]
    D_sp = np.array([d for k in sparse for _, d in vids[k]["ev"]], dtype=np.int64)
    sp = dict(dataset="Sparse subset (in-event < 0.3)", doi="", n_videos=len(sparse),
              n_frames=int(sum(len(vids[k]["g"]) for k in sparse)),
              in_event_fraction=float(np.concatenate([vids[k]["g"] for k in sparse]).mean()),
              **law(D_sp))

    csv = pd.read_csv(C.C1_CSV)
    prot = dict(
        device="NVIDIA Jetson Orin Nano 8 GB, JetPack default power mode",
        sensor="onboard INA3221 rails read through tegrastats",
        sampling_rate_hz=10,
        warmup_s=60, measure_s=float(csv.measure_s.max()),
        samples_per_run=int(csv.n_samples.max()),
        detector="YOLO26-s, 640 px",
        runtimes=sorted(str(r) for r in set(csv.runtime) if str(r) in ("ort-cuda", "trt-fp16")),
        n_runs=dict(idle=int((csv.tag == "idle_yolo26s").sum()),
                    always_on_ort=int((csv.tag == "active_yolo26s").sum()),
                    always_on_trt=int((csv.tag == "active_trt_yolo26s").sum()),
                    duty_cycle_sweep_per_point=int(csv[csv["mode"] == "sweep"].run.max())),
        sweep_points_a=[float(1 / k) for k in sorted(csv[csv["mode"] == "sweep"].k.unique())],
        slot_s=1.0 / C.FPS, frame_rate_fps=C.FPS,
        note=("Duty-cycle sweep points are single runs (n = 1); idle and always-on "
              "points are n = 3. The affine model is therefore extrapolated above "
              "a = 0.5, and activations above the measured inference throughput are "
              "not real-time feasible on this device."))

    out = dict(seed=C.SEED, rows=rows, total=total, sparse_subset=sp,
               sparse_video_ids=[f"{a}/{b}" for a, b in sorted(sparse)],
               measurement_protocol=prot,
               note="Corpus = CDnet2014 + LASIESTA + BMC.")
    C.jdump(out, C.RESULTS / "corpus_table.json")
    hdr = f"{'dataset':<12}{'videos':>7}{'events':>8}{'frames':>9}{'in-ev':>8}{'mean':>8}{'med':>6}{'CV':>7}"
    print(hdr)
    for r in rows + [total, sp]:
        print(f"{r['dataset'][:12]:<12}{r['n_videos']:>7}{r['n_events']:>8}{r['n_frames']:>9}"
              f"{r['in_event_fraction']*100:>7.1f}%{r['mean']:>8.2f}{r['median']:>6.0f}{r['cv']:>7.2f}")
    return out


if __name__ == "__main__":
    main()
