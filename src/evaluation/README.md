# Evaluation protocol (Person 4)

Owner: Person 4. This file fixes **what** we measure, **on which data**, and **how**, so that every
number in the report can be reproduced with one command. Status labels: **DONE** (measured, in
`results/metrics/`), **PARTIAL**, **PLANNED** (not done yet, must be written as planned in the report).

## 1. Questions and metrics

The course brief asks the system to "report detection/classification performance and speed" and to
handle distance/scale, temporal consistency, and small or partially occluded objects.

| # | Question | Metric(s) | Needs | Status |
|---|---|---|---|---|
| 1 | Are objects found? | Precision, Recall (class-agnostic), mAP50 | GT boxes + predictions | DONE (mAP50 from Ultralytics) |
| 2 | Is the light state / sign type correct? | Class-aware P/R, confusion matrix, classification accuracy, per-class recall | GT class of matched boxes | DONE |
| 3 | How fast is it? | Latency (ms/frame) and FPS, model-only and end-to-end | Timestamps on one machine | PLANNED |
| 4 | Small / far / occluded / dark / blurred objects? | Recall per size bin; recall occluded vs. not; recall under synthetic degradations | Box sizes, occlusion flag, images | PARTIAL (lights size bins only) |
| 5 | Is tracking temporally consistent? | Proxy measures (see 7) | Tracker output on video | PLANNED |

## 2. Data used for evaluation

| Model | Dataset | Split used | Images / boxes | Note |
|---|---|---|---|---|
| Traffic lights (YOLO11n, imgsz 1280) | Bosch Small Traffic Lights | val (sequence-based, other dates/routes than train) | 865 / 1756 | `best.pt` was **selected on this val set** (epoch 23) and there is no separate test set, so results may be optimistic |
| Traffic signs (YOLO11n, imgsz 1280) | Mapillary Traffic Sign Dataset, 6 classes | **test** (random image split, seed 42) | 218 / 323 | Fair number; val (217 / 313) was used for model selection |
| Video pipeline | 20 s Kraków driving video (CC BY) | - | - | No ground truth: speed and qualitative results only |

Small classes give unstable numbers and are always reported with their counts: yellow (56) and off (70)
lights; stop (21) and priority_road (27) signs in test.

## 3. Matching rule (fixed for every model)

* Keep predictions with confidence >= **0.25**.
* Per image, visit predictions from highest to lowest confidence (greedy).
* A prediction is a **TP** if it overlaps a not-yet-matched GT box with **IoU >= 0.5**; otherwise **FP**.
  Unmatched GT boxes are **FN**.
* **Class-agnostic**: class ignored ("was the object found?").
  **Class-aware**: GT class must also be equal ("found and correctly classified?").

This is the same rule Person 1 and Person 2 used; `evaluation.py` reproduces their numbers exactly
(lights 1201 TP / 151 FP, signs test 203 TP / 101 FP).

## 4. Metric definitions

* Precision = TP / (TP + FP); Recall = TP / (TP + FN).
* **mAP50**: taken from Ultralytics `model.val()` (confidence threshold 0.001). It is **not** computed
  from our prediction files, because those are cut at conf 0.25 and the precision-recall curve would be
  truncated (AP would be underestimated).
* **Confusion matrix**: rows = true class (+ `FP` row for predictions on no object),
  columns = predicted class (+ `missed`). Built from class-agnostic matches. Its diagonal can differ by
  a few boxes from class-aware TP (lights: 1170 vs. 1173) because the two use separate greedy matchings.
* **Classification accuracy** = correct class / all class-agnostic matches. It isolates
  "wrong colour/type" from "not found".
* **Size bins** (<10, 10-20, >=20 px at model input). Currently lights use box **width** (Person 1) and
  signs use box **height** (Person 2) - not comparable yet. Proposal: use the **shorter side**
  `min(w, h)` for both (team decision pending). Sign bins also need each image's original size.

## 5. Current results (conf 0.25, IoU 0.5)

| Model / split | Matching | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| Lights / val | class-agnostic | 1201 | 151 | 555 | 0.888 | 0.684 |
| Lights / val | class-aware | 1173 | 179 | 583 | 0.868 | 0.668 |
| Signs / test | class-agnostic | 210 | 94 | 113 | 0.691 | 0.650 |
| Signs / test | class-aware | 203 | 101 | 120 | 0.668 | 0.628 |

Classification accuracy on detected objects: lights 0.974 (1170/1201; off only 9/20),
signs test 0.967 (203/210). mAP50 (Ultralytics): lights val 0.533, signs test 0.643.
Lights recall by width: <10 px 0.523 (530/1013), 10-20 px 0.894, >=20 px 0.919.

## 6. Speed protocol (PLANNED)

All timings on **one machine**, reported with CPU/GPU, OS, Python, torch and ultralytics versions.
Ignore the first 20 frames (warm-up), then time >= 300 frames at imgsz 1280:
(a) light model only, (b) sign model only, (c) full pipeline: read frame -> both models -> tracker ->
draw/write. Report mean and median ms/frame and FPS = 1000 / mean ms. The numbers reported so far
(Apple M2 5.7 ms, Tesla T4 15.9 ms, RTX 4060 19.8 ms model-only; video pipeline 19.7 FPS on unknown
hardware) come from different machines and must not be compared directly.

## 7. Robustness and tracking (PLANNED)

* **Synthetic degradations** on the same evaluation images (methods from the CENG391 lab notebooks),
  each at increasing severity: brightness/contrast `I' = clip(alpha*I + c)`, gamma (power-law),
  Gaussian blur (kernel 3/9/15), Gaussian noise, salt-and-pepper noise. Report recall and precision
  vs. the clean images. Optional: does a pre-processing step (histogram equalization, median filter)
  recover the loss? These are simulated conditions, not real night or motion blur, and will be
  labelled so. Needs the dataset images, so it runs on a machine that has them.
* **Occlusion**: Bosch provides an `occluded` flag per box; it was dropped during YOLO conversion.
  With a per-box list from Person 1 we report recall for occluded vs. non-occluded lights.
* **Tracking / temporal consistency**: no tracking ground truth exists for our video, so identity
  metrics (IDF1, ID switches) cannot be computed. Clearly-limited proxies: number of tracks, mean track
  length, short gaps (object lost for 1-2 frames then found again), and for lights the share of tracks
  whose colour flips, before vs. after majority voting over the last N frames.

## 8. Reproducibility record (fill for every experiment)

Data subset and split file, model name and weight file (md5), thresholds (conf, IoU, imgsz), exact
command, hardware, library versions, date. Example run:

```bash
python src/evaluation/evaluation.py signs \
  --pred results/traffic_signs/train_y11n_1280/predictions_test.csv \
  --gt results/traffic_signs/handover/gt_test.csv \
  --out results/metrics/signs_finetuned_test --size-dim none
```

The script only reads the input files and writes into `--out` (`results/metrics/...`).

## 9. Known limitations (threats to validity)

* Lights are evaluated on the split used for model selection (no independent test set).
* Different split methods: lights sequence-based, signs random per image (near-duplicate images across
  splits not checked).
* Lights and signs come from different datasets and regions (USA vs. worldwide); their numbers describe
  each model on its own data and are not a head-to-head comparison.
* Sign false positives are partly unlabeled real signs: on val, 36 of 57 FPs lie on MTSD sign types we
  did not train on, so precision is underestimated.
* Few examples for some classes (see section 2).
* The video has no ground truth, so on video only speed and qualitative behaviour are reported.
