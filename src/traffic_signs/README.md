# Traffic sign detection and type classification

Owner: Person 2. A YOLO11n detector fine-tuned on the Mapillary Traffic Sign Dataset (MTSD) for 6 sign classes. The detector predicts the box and the sign type in one step.

## Classes

The IDs are final and shared with the team (`class_map.yaml`).

| ID | Class | MTSD labels (prefix) |
|---|---|---|
| 0 | stop | `regulatory--stop--` |
| 1 | yield | `regulatory--yield--` |
| 2 | speed_limit | `regulatory--maximum-speed-limit-` (all values merged) |
| 3 | no_entry | `regulatory--no-entry--` |
| 4 | priority_road | `regulatory--priority-road--` |
| 5 | no_parking | `regulatory--no-parking--` |

## Model

- File: `traffic_signs_y11n_1280.pt` (5.7 MB). Download it from the GitHub Release [`signs-y11n-1280-v1`](https://github.com/berksahiinn/CENG391-Traffic-Understanding/releases/tag/signs-y11n-1280-v1). The repo is private, so you need to be logged in to GitHub. Put the file in `models/`. `*.pt` files are ignored by git.
- Settings: **imgsz=1280, conf=0.25**. Signs are very small, so a lower imgsz will miss many of them.
- Training: 100 epochs, batch 8, seed 42, COCO-pretrained `yolo11n.pt`, RTX 4060 Laptop GPU, ultralytics 8.4.208, torch 2.14.1+cu126. Full details are in `results/traffic_signs/train_y11n_1280/run_info.txt`.

## For Person 3 (video pipeline)

Minimal example that writes the common detection format, `frame_index, bbox_xyxy, class_name, confidence, track_id`:

```python
from ultralytics import YOLO

model = YOLO("models/traffic_signs_y11n_1280.pt")
# model.track(...) adds track IDs (ByteTrack). Use model.predict(...) for detection only.
for frame_index, r in enumerate(model.track(source="video.mp4", imgsz=1280, conf=0.25,
                                            tracker="bytetrack.yaml", stream=True, verbose=False)):
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
    for box, cls, conf, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                                   r.boxes.conf.tolist(), ids):
        print(frame_index, box, model.names[cls], round(conf, 4), tid)
```

- Tracking needs the `lap` package (`pip install lap`). Ultralytics installs it automatically the first time you use it, and then asks you to rerun the command. The snippet was tested on 5 test images (2026-11-10) and printed rows in the format above.
- On the RTX 4060 the model takes about 6-9 ms per image at imgsz 1280 (inference only). Decoding and resizing the image comes on top of that.
- Example outputs: `results/traffic_signs/handover/sample_predictions/` (6 random test images).
- This model has not been tested on video yet, only on single MTSD images. Video frames may be blurrier than the training photos.

## For Person 4 (evaluation)

All files are in the repo. Coordinates are in original image pixels. `frame_index` is the position of the image in the sorted image list of that split. Use the `image` column to join files.

| File | Content |
|---|---|
| `results/traffic_signs/train_y11n_1280/predictions_val.csv`, `predictions_test.csv` | Predictions at conf 0.25: `frame_index,image,x1,y1,x2,y2,class_name,confidence` |
| `results/traffic_signs/handover/gt_val.csv`, `gt_test.csv` | Ground truth: `frame_index,image,x1,y1,x2,y2,class_name` |
| `results/traffic_signs/handover/splits/{train,val,test}.txt` | Image names in each split |
| `results/traffic_signs/handover/image_sizes.csv` | `image,width,height` of every val and test image (original pixels). Sizes vary, so use them to scale boxes to the model input for recall by size: `scale = 1280 / max(width, height)`, size = box height × scale. This is the rule `06_evaluate.py` uses for its bins (<10, 10-20, >=20 px). |
| `results/traffic_signs/train_y11n_1280/metrics_*.txt` | Metrics computed by `06_evaluate.py` |

Split: random image-level split with seed 42 over the 1449 MTSD images that contain at least one target sign. Train has 1014 images, val 217 (313 boxes), test 218 (323 boxes). The model was selected on val, so **test is the fair split for reporting**.

Matching rule used in `06_evaluate.py` and `07_error_analysis.py`: conf ≥ 0.25 and IoU ≥ 0.5. A prediction counts as correct only if its class is also correct. Predictions are matched in order of confidence, highest first.

## Results so far

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.809 | 0.607 | 0.689 | 0.507 |
| test | 0.722 | 0.631 | 0.643 | 0.457 |

- **Stop class compared with the COCO-pretrained baseline** (val, conf 0.25, IoU 0.5). P went from 0.302 to 0.815 and R from 0.432 to 0.595.
- **Recall by box height at model input** (test). Below 10 px: 0.189. 10-20 px: 0.551. 20 px and above: 0.860.
- **Error analysis on val** (`results/traffic_signs/error_analysis_y11n_1280/`):
  - Wrong-class predictions are rare (7).
  - Most misses are tiny signs (56 of 110 misses are under 10 px).
  - 36 of 57 false positives fall on real MTSD signs that are not one of our classes, such as other-sign or no-stopping.

**Known limitations**
- About 66% of MTSD objects (`other-sign` and the classes we dropped) are left unlabeled. The model can still fire on them, and such detections count as false positives.
- Some classes have very few val/test samples (val priority_road 15, test stop 21).
- The random split may be optimistic: images from the same area can end up in different splits.

## Scripts

Run them from the repo root with the venv active. Data lives outside the repo in `C:\data`.

| Script | Purpose |
|---|---|
| `01_count_classes.py`, `01b_check_classes.py` | Count MTSD labels and candidate classes |
| `02_convert_to_yolo.py` | Convert MTSD to YOLO format and make the seed-42 split |
| `03_visualize_and_stats.py` | Draw GT boxes, plot class distribution, compute box sizes |
| `04_baseline_coco.py` | COCO-pretrained yolo11n baseline (stop class only) |
| `05_train_yolo.py --name y11n_1280 --epochs 100` | Fine-tune |
| `06_evaluate.py --run y11n_1280` | Metrics on val and test, prediction CSVs |
| `07_error_analysis.py --run y11n_1280` | Confusion matrix at conf 0.25, false positive and miss analysis |
| `08_export_handover.py --run y11n_1280` | Model copy for the release, split lists, GT CSVs, sample predictions |
