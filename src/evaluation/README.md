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
| 3 | How fast is it? | Latency (ms/frame) and FPS, model-only and end-to-end | Timestamps on one machine | DONE (Apple M3 Pro, see 6) |
| 4 | Small / far / occluded / dark / blurred objects? | Recall per size bin; recall occluded vs. not; recall under synthetic degradations | Box sizes, occlusion flag, images | PARTIAL (size bins DONE for lights and signs; occlusion DONE for lights on the Bosch test sequence; degradations PLANNED) |
| 5 | Is tracking temporally consistent? | Effect of tracking on P/R against labels; proxy measures (see 7) | Tracker output on labelled video | DONE for lights (tracking vs. no tracking on the Bosch test sequence; track lengths and colour changes from Person 3, recounted by Person 4). Identity metrics impossible: no track ids in the labels |

## 2. Data used for evaluation

| Model | Dataset | Split used | Images / boxes | Note |
|---|---|---|---|---|
| Traffic lights (YOLO11n, imgsz 1280) | Bosch Small Traffic Lights | val (sequence-based, other dates/routes than train) | 865 / 1756 | `best.pt` was **selected on this val set** (epoch 23) and there is no separate test set, so results may be optimistic |
| Traffic signs (YOLO11n, imgsz 1280) | Mapillary Traffic Sign Dataset, 6 classes | **test** (random image split, seed 42) | 218 / 323 | Fair number; val (217 / 313) was used for model selection |
| Traffic lights on video (video pipeline output) | Bosch Small Traffic Lights, **official test sequence** (`test.yaml`), made into a video by Person 3 | test (never used for training, model selection or thresholds) | 8334 frames / 13486 boxes (2088 occluded) | Independent test set for the light model. Predictions come from the H.264 video and a CSV rounded to 0.1 px, so this measures the system on video. Consecutive 15 fps frames show the same lights, so boxes are not independent samples |
| Video pipeline | 20 s Kraków driving video (CC BY 3.0), 1280x720, 30 fps, 600 frames | - | - | No ground truth: speed and qualitative results only. Rebuilt independently by Person 4 from the recipe in `src/video_pipeline/README.md` (Person 3); frames match Person 3's sample frames (PSNR 35-40 dB at the same frame index) |

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
(lights 1201 TP / 151 FP, signs test 203 TP / 101 FP). The team agreed on conf 0.25 / IoU 0.5 as the
common protocol (25 Nov 2026); other IoU values appear only as sensitivity checks.

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
* **Size bins** (<10, [10, 20), >=20 px at model input; a box of exactly 10 px is in 10-20).
  Lights use box **width** (Person 1, images are already 1280x720, so original px = model px).
  Signs use box **height** at model input (Person 2): `height * 1280 / max(W, H)` of the original image,
  with image sizes from `results/traffic_signs/handover/image_sizes.csv`.
  Decision for the progress report: each model keeps its owner's definition, every table/figure states
  it, and light and sign size results are **not compared** with each other. A common definition
  (shorter side `min(w, h)`) can be adopted for the final report.
* Size bins and per-class recall are recall only: a false positive has no GT box, so it has no size or
  true class.

## 5. Current results (conf 0.25, IoU 0.5)

| Model / split | Matching | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| Lights / val | class-agnostic | 1201 | 151 | 555 | 0.888 | 0.684 |
| Lights / val | class-aware | 1173 | 179 | 583 | 0.868 | 0.668 |
| Signs / test | class-agnostic | 210 | 94 | 113 | 0.691 | 0.650 |
| Signs / test | class-aware | 203 | 101 | 120 | 0.668 | 0.628 |

Classification accuracy on detected objects: lights 0.974 (1170/1201; off only 9/20),
signs test 0.967 (203/210), signs val 0.966 (196/203). mAP50 (Ultralytics): lights val 0.533,
signs test 0.643.

Recall by size (all numbers reproduce Person 1's / Person 2's own results exactly):

| Model / split | Size definition | Matching | <10 px | 10-20 px | >=20 px |
|---|---|---|---|---|---|
| Lights / val | box width | class-agnostic | 0.523 (530/1013) | 0.894 (420/470) | 0.919 (251/273) |
| Signs / test | box height at model input | class-aware | 0.189 (10/53) | 0.551 (70/127) | 0.860 (123/143) |
| Signs / val | box height at model input | class-aware | 0.143 (10/70) | 0.655 (72/110) | 0.857 (114/133) |

Figures (from `plot_metrics.py`): `results/metrics/figures/` - confusion matrices for lights val,
signs val and signs test (cell text = count, colour = share of row) and recall by size (both
matchings, one panel per model).

**IoU sensitivity check (not the protocol, IoU 0.3 instead of 0.5)**, outputs in `*_iou03/`:

| Model / split | TP (agnostic) 0.5 -> 0.3 | FP 0.5 -> 0.3 | <10 px recall 0.5 -> 0.3 |
|---|---|---|---|
| Lights / val | 1201 -> 1273 | 151 -> 79 | 530 -> 586 of 1013 |
| Signs / test | 210 -> 212 | 94 -> 92 | 12 -> 13 of 53 |
| Signs / val | 203 -> 205 | 57 -> 55 | 14 -> 14 of 70 |

For signs, missed small signs are missing detections, not boxes that are slightly off. For lights,
about half of the false positives lie on a real light with IoU 0.3-0.5, mostly in the smallest bin
(poor box fit on tiny objects, or imprecise labels on boxes of a few pixels - the data cannot tell
which). Even at IoU 0.3, 427 of 1013 lights below 10 px are missed.

**Lights on the Bosch test sequence (independent test set, video pipeline output, conf 0.25, IoU 0.5)**,
outputs in `lights_video_bosch_test_{detect,track}/`:

| Run | Matching | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| detect (no tracking) | class-agnostic | 9846 | 1681 | 3640 | 0.854 | 0.730 |
| detect (no tracking) | class-aware | 9825 | 1702 | 3661 | 0.852 | 0.729 |
| track (confirmed tracks only) | class-agnostic | 8604 | 757 | 4882 | 0.919 | 0.638 |

| Run | <10 px | 10-20 px | >=20 px | not occluded | occluded |
|---|---|---|---|---|---|
| detect | 0.622 (5361/8624) | 0.923 (4094/4436) | 0.918 (391/426) | 0.805 (9176/11398) | 0.321 (670/2088) |
| track | 0.501 | 0.886 | 0.831 | 0.715 | 0.220 |

(recall, class-agnostic, size = box width)

* The light model reaches a similar level on the independent test set as on val (recall 0.730 vs.
  0.684, precision 0.854 vs. 0.888), so the val numbers are not clearly optimistic. The two sets come
  from different drives, so this is not a strict comparison.
* Occluded lights are found far less often (0.32 vs. 0.81).
* `off` lights: none of the 442 is detected with the right state (426 missed, 16 found with a colour).
* Tracking keeps only confirmed tracks, so it works as a temporal filter: fewer false positives
  (1681 -> 757) but also fewer found lights, mostly small ones. It does not recover missed lights.

## 6. Speed (DONE on one machine)

Script: `speed.py`. Method: batch 1, imgsz 1280, conf 0.25, the first 20 frames are run but not
timed (warm-up), the next 300 frames are timed with `time.perf_counter()`, and the GPU is synchronised
before each timer stops. Per frame we time video decoding, the light model and the sign model
(each = pre-processing + inference + post-processing, wall clock).
**End-to-end = decode + both models (+ tracker).** Drawing boxes and writing the output video are
**not** included: they are visualisation, not part of the system, and their cost depends on the codec.
(The earlier plan included them; this was changed when the script was written.)

Machine: Apple M3 Pro, 18 GB, macOS 15.6.1, Python 3.12.7, torch 2.15.0, ultralytics 8.4.211,
OpenCV 5.0.0. Input: the Kraków clip, 1280x720. Results in `results/metrics/speed/<run>/summary.txt`
(with per-frame times in `per_frame.csv`).

| Run | Device | Tracking | Both models, ms/frame (mean / p95) | End-to-end FPS |
|---|---|---|---|---|
| `m3pro_mps` | MPS (GPU) | no | 22.6 / 23.0 | 43.5 |
| `m3pro_mps_track` | MPS (GPU) | ByteTrack | 23.9 / 25.6 | 41.2 |
| `m3pro_cpu` | CPU | no | 157.6 / 170.1 | 6.3 |

The clip is 30 fps, so on the GPU the system runs faster than real time (also at p95); on the CPU it
does not. Tracking adds about 1.3 ms per frame.

Notes:
* Report the wall-clock numbers. Ultralytics' own "inference" time on MPS (about 3.9 ms per model) is
  much lower than the wall clock (about 11 ms); on the CPU the two differ by only about 1.7 ms, so the
  gap on MPS is not resizing or NMS. Most likely the Ultralytics timer stops before the GPU has finished
  (the wait happens when the boxes are copied back to the CPU). We do not use it as a result.
* Other numbers in the team come from other machines or definitions and must not be compared directly:
  Tesla T4 15.9 ms (Person 1, fine-tuned light model, model only), RTX 4060 19.8 ms (Person 2),
  5.7 ms (Person 1, COCO baseline model, Ultralytics inference on MPS, chip not recorded); video pipeline
  19.7 FPS on Apple M4 (Person 3, commit 103efb4, includes drawing and video writing; its 3.7 / 3.5 ms
  are Ultralytics inference times).

What each speed number covers (state this next to every speed number in the report):

| Number | Machine | Decode | Light + sign model | Tracking | Drawing + video writing | Source |
|---|---|---|---|---|---|---|
| 43.5 FPS | Apple M3 Pro, MPS | yes | yes | no | no | Person 4, `speed/m3pro_mps` |
| 41.2 FPS | Apple M3 Pro, MPS | yes | yes | yes | no | Person 4, `speed/m3pro_mps_track` |
| 6.3 FPS | Apple M3 Pro, CPU | yes | yes | no | no | Person 4, `speed/m3pro_cpu` |
| 19.7 FPS (Kraków) | Apple M4, MPS | yes | yes | yes | yes | Person 3, `results/video_pipeline/finetuned_track/run_info.txt` |
| 20.2 / 18.1 FPS (Bosch, detect / track) | Apple M4, MPS | yes | yes | no / yes | yes | Person 3, `results/video_pipeline/bosch_test_{detect,track}/run_info.txt` |

## 7. Robustness and tracking

* **Synthetic degradations** (PLANNED, final, if time allows). Agreed plan with Person 3: apply the
  degradations to `datasets/videos/bosch_test.mp4` with ffmpeg, re-run the video pipeline (Person 3)
  and measure each run with `evaluation.py lights_video` against the same `test.yaml` (Person 4); no
  dataset zip needed. Methods follow the CENG391 lab notebooks, each at increasing severity:
  brightness/contrast `I' = clip(alpha*I + c)` and gamma (ffmpeg `eq`), Gaussian blur (`gblur`),
  noise (`noise`). Report precision and recall vs. the clean run. These are simulated conditions on an
  already compressed video, not real night or motion blur, and will be labelled so.
* **Occlusion**: DONE for lights on the Bosch test sequence (`occluded` flag in `test.yaml`, see 5).
  On val the flag was dropped during YOLO conversion; a per-box list from Person 1 would add val.
* **Tracking / temporal consistency** (DONE for lights, by Person 3): track lengths and colour changes
  for both videos are in `src/video_pipeline/README.md`. Person 4 recounted them from the committed
  `detections.csv` files and got the same numbers: Bosch test 147 light tracks, 68 with >= 15 boxes,
  longest 697 boxes; 19 tracks change colour (22 changes: 17 red->green, 2 green->yellow,
  2 yellow->red, 1 red->yellow); no single-frame colour flicker. Kraków: one light track of 2 frames.
  Track gaps are reported for Kraków by Person 3 (Bosch can be added). Identity metrics (IDF1, ID
  switches) are not possible: no labelled track ids. Precision rising and recall falling with tracking
  (see 5) is the expected effect of keeping only confirmed tracks (Person 3's design choice).
* **Labelled video**: DONE for lights on the Bosch test sequence (detection, state, size, occlusion,
  tracking vs. no tracking; see 5). Bosch labels have no track ids, so ID switches / IDF1 are still not
  possible. There is no labelled video for signs.

## 8. Reproducibility record

Every result folder is made by one command from the repo root (activate `.venv` first).
The scripts only read their inputs and write into `results/metrics/...`.

Inputs that are not in git:
* `datasets/traffic_lights_eval/{val_gt_labels,pred_labels}.zip` - Person 1's Drive, `tl_runs/finetuned_eval/`
  (use `pred_labels.zip`, 6 decimals; `val_predictions.csv` is rounded to 0.1 px and changes one match).
* `models/traffic_lights_y11n_1280_best.pt` (Person 1's Drive link) and `models/traffic_signs_y11n_1280.pt`
  (GitHub Release `signs-y11n-1280-v1`); both trained with ultralytics 8.4.208. SHA-256:
  lights `30554dc9...d6f60408`, signs `e64c1ac0...ab09d14c`.
* `datasets/videos/sample_drive.mp4` - rebuild with the commands in `src/video_pipeline/README.md`
  (needs ffmpeg; Person 4 used ffmpeg 9.0.2).
* `datasets/bosch_test/test.yaml` - inside `dataset_test_rgb.zip` (12.2 GB) of Zenodo record 12706046
  (Bosch licence for non-commercial use; not in git). The file is 2.4 MB; Person 4 fetched only this
  member with HTTP range requests instead of the whole zip.

```bash
# metrics (add --iou 0.3 and an *_iou03 output folder for the sensitivity check)
python src/evaluation/evaluation.py lights --pred datasets/traffic_lights_eval/pred_labels.zip \
  --gt datasets/traffic_lights_eval/val_gt_labels.zip --out results/metrics/lights_finetuned_val --size-dim w
python src/evaluation/evaluation.py signs --pred results/traffic_signs/train_y11n_1280/predictions_test.csv \
  --gt results/traffic_signs/handover/gt_test.csv --out results/metrics/signs_finetuned_test \
  --size-dim h --sizes results/traffic_signs/handover/image_sizes.csv
# (signs val: replace "test" with "val" in the three paths)
python src/evaluation/evaluation.py lights_video --pred results/video_pipeline/bosch_test_detect/detections.csv \
  --gt datasets/bosch_test/test.yaml --out results/metrics/lights_video_bosch_test_detect --size-dim w
# (tracking run: replace "detect" with "track" in both paths)

# figures (reads only the CSVs above)
python src/evaluation/plot_metrics.py

# speed
python src/evaluation/speed.py --video datasets/videos/sample_drive.mp4 --device mps --name m3pro_mps
python src/evaluation/speed.py --video datasets/videos/sample_drive.mp4 --device mps --name m3pro_mps_track --track
python src/evaluation/speed.py --video datasets/videos/sample_drive.mp4 --device cpu --name m3pro_cpu
```

## 9. Known limitations (threats to validity)

* The main light numbers are on the split used for model selection (val). The Bosch test sequence now
  gives an independent check, but through the video pipeline (H.264 video, 0.1 px CSV), not on the PNGs.
* Different split methods: lights sequence-based, signs random per image (near-duplicate images across
  splits not checked).
* Lights and signs come from different datasets and regions (USA vs. worldwide); their numbers describe
  each model on its own data and are not a head-to-head comparison.
* Sign false positives are partly unlabeled real signs: on val, 36 of 57 FPs lie on MTSD sign types we
  did not train on, so precision is underestimated.
* Few examples for some classes (see section 2).
* `off` lights are not recognised: on the Bosch test sequence none of 442 is found with the right state
  (426 missed, 16 found with a colour); on val only 9 of 20 detected `off` lights get the right state.
* The video has no ground truth, so on video only speed and qualitative behaviour are reported.
* Light and sign size bins use different definitions (width vs. height) and must not be compared.
* Speed is measured on one laptop GPU (Apple M3 Pro); other hardware will give other numbers.
* Our ultralytics (8.4.211) is newer than the one used for training and for the teammates' runs
  (8.4.208); all metric numbers still reproduce exactly, and speed was only measured with 8.4.211.
