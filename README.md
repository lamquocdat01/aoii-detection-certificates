# Worst-case AoII certificates for networked monitoring over delayed links: traces, power logs and code

Everything needed to reproduce the numbers, tables and figures of the paper *"Worst-Case
Certificates for the Age of Incorrect Information in Networked Monitoring over Delayed
Links"*. Seed 42 throughout; all results are deterministic.

## What the code reproduces

| Result in the paper | Script | Output |
|---|---|---|
| Theorem 1 (transition-count certificate): exhaustive check, attainment, Monte Carlo, Corollary 1 | `t2star_verify.py` | `results/t2star_verify.json` |
| Theorem 1 on the measured workload | `t2_refined_corpus.py` | `results/t2_refined_corpus.json` |
| Lemma 1 and Proposition 2 (certificate ⇔ cap), both directions | `sanity_delta.py` | `results/sanity_delta.json` |
| Algorithm 1: certified schedules, price of the certificate | `optim_delta.py`, `summary_delta.py` | `results/optim_delta.json`, `results/summary_delta.json` |
| Hold-out (60/40 by video), semi-Markov model, oracle | `holdout_delta.py` | `results/holdout_delta.json` |
| Markov-optimal baseline (raw and budget-corrected) | `markov_baseline.py` | `results/markov_baseline.json` |
| Remark 1 (certificate in distribution) | `cor22_tail.py` | `results/cor22_tail.json` |
| Lemma 2 / supplementary monotonicity checks | `prop34_check.py` | `results/prop34_check.json` |
| Trace replay of the policy zoo (gate thresholds) | `aoii_replay.py` | `results/aoii_scores.json` |
| Measured power model | `energy_frontier.py` | `results/energy_frontier.json` |
| Figures | `figs_delta.py` | `figures/` |
| Every number printed in the paper | `make_numbers_delta.py` | `results/numbers_delta.json` |

## Contents

```
data/events.csv           one row per event: dataset, video, event_id, onset, duration
data/frame_scores.parquet one row per frame: dataset, video, frame, score, g_t
data/frame_scores.csv     the same table as CSV
data/power/*.csv          raw 10 Hz board-power logs, one file per run
data/power/orin_results.csv  run-level summary (mean watts, fps, inference fps, metadata)
code/                     analysis scripts
expected/numbers_delta.json  the values printed in the paper (reference for reproduction)
```

An *event* is a maximal run of frames with `g_t == 1`; `events.csv` is re-derived from
`frame_scores` by the code, so the two files cannot drift apart. In the paper the binary
state `g_t` is the remote state of the monitoring loop and one frame is one slot.

## Workload

| Source dataset | DOI | Videos | Events | Frames |
|---|---|---|---|---|
| CDnet2014 | 10.1109/CVPRW.2014.126 | 53 | 2516 | 118173 |
| LASIESTA | 10.1016/j.cviu.2016.08.005 | 48 | 936 | 18425 |
| BMC | 10.1007/978-3-642-37410-4_25 | 20 | 261 | 29991 |
| **Total** | | **121** | **3713** | **166589** |

The source videos are not redistributed. `frame_scores` carries only a binary
ground-truth event indicator and a scalar content score per frame; obtain the imagery and
the original annotations from the distributors listed above. See `LICENSE-DATA`.

## Measurement protocol (update cost and delay)

* **Device** NVIDIA Jetson Orin Nano 8 GB, JetPack default power mode.
* **Update** one YOLO-class detector inference at 640 px, under two runtimes: ort-cuda, trt-fp16.
* **Instrument** onboard INA3221 rails read through tegrastats at 10 Hz; internal Jetson
  power sensors need care in interpretation (DOI 10.1109/EDGE60047.2023.00034).
* **Run structure** 60 s warm-up, then 300 s of logging (2997 samples per run). Idle and
  always-on points use n = 3 runs; each duty-cycle sweep point (a in {0.5, 0.2, 0.1}) is a
  single run (n = 1).
* **Slot** 33.3 ms (30 frames/s).
* **Affine power model** `P(a) = 4.9381 + 6.9155 a` W, R^2 = 0.99901; the moving-block
  bootstrap interval covers within-run variability only.
* **Delay** mean inference latency is 56.8 ms (ort-cuda) and 37.4 ms (trt-fp16), both above
  one slot, so the constant end-to-end delay is d = 2 slots. Only the mean latency was
  measured; there is no per-inference latency distribution in this release.

## Reproduction

```bash
pip install -r requirements.txt
python code/reproduce_delta.py
```

`reproduce_delta.py` runs the scripts above in order and compares every regenerated value
with `expected/numbers_delta.json`, key by key (exit code 0 only if all match). The full
run takes 1–3 hours on a laptop CPU. Results land in `results/` and figures in `figures/`.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`), with the provenance and scope
limits stated there.

## Citation

See `CITATION.cff`.
