# Detection-grounded symmetric AoII: traces, power logs and analysis code

Everything needed to reproduce the numbers and figures of the accompanying paper on
worst-case symmetric age-of-incorrect-information (AoII) certificates for sampling
policies in edge perception. Seed 42 throughout; all results are deterministic.

## Contents

```
data/events.csv           one row per event: dataset, video, event_id, onset, duration
data/frame_scores.parquet one row per frame: dataset, video, frame, score, g_t
data/frame_scores.csv     the same table as CSV
data/power/*.csv          raw 10 Hz board-power logs, one file per run
data/power/orin_results.csv  run-level summary (mean watts, fps, inference fps, metadata)
code/                     analysis scripts (see "Reproduction" below)
```

An *event* is a maximal run of frames with `g_t == 1`; `events.csv` is re-derived from
`frame_scores` by the code, so the two files cannot drift apart.

## Corpus

| Source dataset | DOI | Videos | Events | Frames |
|---|---|---|---|---|
| CDnet2014 | 10.1109/CVPRW.2014.126 | 53 | 2516 | 118173 |
| LASIESTA | 10.1016/j.cviu.2016.08.005 | 48 | 936 | 18425 |
| BMC | 10.1007/978-3-642-37410-4_25 | 20 | 261 | 29991 |
| **Total** | | **121** | **3713** | **166589** |

The source videos are not redistributed. `frame_scores` carries only a binary
ground-truth event indicator and a scalar content score per frame; obtain the imagery and
the original annotations from the distributors listed above. See `LICENSE-DATA`.

## Measurement protocol

The power and latency figures used in the paper were measured as follows.

* **Device** NVIDIA Jetson Orin Nano 8 GB, JetPack default power mode.
* **Detector** YOLO26-s, 640 px, under two runtimes: ort-cuda, trt-fp16.
* **Instrument** onboard INA3221 rails read through tegrastats, sampled at 10 Hz. Internal
  Jetson power sensors need care in interpretation; see DOI 10.1109/EDGE60047.2023.00034.
* **Run structure** 60 s warm-up, then 300 s of
  logging (2997 samples per run). Idle and always-on points use
  n = 3 runs; each duty-cycle sweep point
  (a in {0.5, 0.2, 0.1}) is a **single run
  (n = 1)**.
* **Slot** one video frame at 30 frames/s, i.e.
  33.3 ms.
* **Affine power model** fitted over the four design points:
  `P(a) = 4.9381 + 6.9155 a` W, R^2 = 0.99901. The 95%
  interval quoted in the paper comes from a moving-block bootstrap over the raw 10 Hz
  samples and therefore covers **within-run** variability only; the run-to-run component
  is unquantified at the n = 1 sweep points.
* **Throughput ceiling** measured inference throughput caps the achievable activation at
  a <= 0.587 (ONNX Runtime/CUDA) and a <= 0.891
  (TensorRT FP16). Mean inference latency is 56.8 ms and
  37.4 ms respectively, so the constant verdict delay used in the
  paper is d = 2 slots. There is **no** per-inference latency
  distribution in this release; only the mean was measured.

## Reproduction

```bash
pip install -r requirements.txt
cd code
python corpus_table.py       # corpus description and measurement protocol
python examples_t03.py       # worked examples + the delay counterexample
python sanity_lemma1.py      # Monte-Carlo check of the closed-form mean AoII
python aoii_replay.py        # the trace-driven replay (policies x delays x phases)
python sanity_thm.py         # impossibility construction, capped-gate check, gate gaps
python heavy_tail.py         # two-sided bounds and tail sensitivity
python energy_frontier.py    # affine power fit with moving-block bootstrap
python make_figs.py          # figures
```

Results land in `results/` and figures in `figures/`.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`), with the provenance and scope
limits stated there.

## Citation

See `CITATION.cff`.
