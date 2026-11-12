"""Baseline: pretrained YOLO11n (COCO) on the traffic sign val split.

COCO has only one class related to our task: class 11 'stop sign'.
So this baseline can only be evaluated on our class 0 (stop).
The model, imgsz, conf, IoU thresholds and bin limits (<10, 10-20, >=20 px)
follow the traffic light baseline in src/traffic_lights/baseline_pretrained.py.

Size bins use the box HEIGHT in pixels AFTER resizing the image to the
model input (longest side = IMGSZ), because that is what the model sees.
Note: the traffic light baseline bins by box WIDTH (gb[3]-gb[1] = x2-x1),
so the size-bin recalls of the two baselines are NOT directly comparable.

Outputs (results/traffic_signs/baseline_coco/):
  results.txt      metrics summary
  predictions.csv  all 'stop sign' predictions (common detection format)
  samples/         a few val images with GT (green) and predictions (red)
"""
import csv
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

DATA_DIR = Path(r"C:\data\signs_yolo")
OUT_DIR = Path("results") / "traffic_signs" / "baseline_coco"
MODEL, IMGSZ, CONF = "yolo11n.pt", 1280, 0.25
COCO_STOP = 11           # COCO class id of 'stop sign'
OUR_STOP = 0             # our class id of 'stop'
OUR_NAMES = ["stop", "yield", "speed_limit", "no_entry", "priority_road", "no_parking"]
IOU_THRESHOLDS = [0.5, 0.3]
N_SAMPLES = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def read_gt(label_path, W, H):
    """Return array of [class_id, x1, y1, x2, y2] in original image pixels."""
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            c, xc, yc, w, h = map(float, line.split())
            rows.append([int(c), (xc - w / 2) * W, (yc - h / 2) * H,
                         (xc + w / 2) * W, (yc + h / 2) * H])
    return np.array(rows).reshape(-1, 5)


def iou(box, boxes):
    """IoU between one box [x1,y1,x2,y2] and an array of boxes (N,4)."""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def size_bin(h_px):
    return "<10px" if h_px < 10 else "10-20px" if h_px < 20 else ">=20px"


OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "samples").mkdir(exist_ok=True)
img_paths = sorted((DATA_DIR / "images" / "val").glob("*.jpg"))

# Per IoU threshold: TP, FP, recall per size bin, and what the false positives hit
stats = {t: {"TP": 0, "FP": 0, "size": {}, "fp_on_class": {}} for t in IOU_THRESHOLDS}
speeds, pred_rows, n_samples_saved = [], [], 0
model = YOLO(MODEL)

# Pass the folder (not a Python list): a list is processed as ONE batch and runs out of GPU memory.
for frame_index, r in enumerate(model.predict(source=str(DATA_DIR / "images" / "val"), batch=1,
                                              imgsz=IMGSZ, conf=CONF, classes=[COCO_STOP],
                                              device=DEVICE, stream=True, verbose=False)):
    speeds.append(r.speed["inference"])
    H, W = r.orig_shape
    scale = IMGSZ / max(W, H)                      # original px -> model input px
    gt = read_gt(DATA_DIR / "labels" / "val" / (Path(r.path).stem + ".txt"), W, H)
    gt_stop = gt[gt[:, 0] == OUR_STOP][:, 1:]
    gt_other = gt[gt[:, 0] != OUR_STOP]

    conf = r.boxes.conf.cpu().numpy()
    order = np.argsort(-conf)                      # match highest confidence first
    preds, conf = r.boxes.xyxy.cpu().numpy()[order], conf[order]

    for b, c in zip(preds, conf):
        pred_rows.append([frame_index, Path(r.path).name,
                          f"{b[0]:.1f}", f"{b[1]:.1f}", f"{b[2]:.1f}", f"{b[3]:.1f}",
                          "stop", f"{c:.4f}"])

    for t, S in stats.items():
        used = np.zeros(len(gt_stop), bool)
        for b in preds:
            if len(gt_stop):
                ious = iou(b, gt_stop); ious[used] = 0; k = ious.argmax()
                if ious[k] >= t:
                    used[k] = True; S["TP"] += 1
                    continue
            S["FP"] += 1
            # Did this false positive land on one of our other sign classes?
            hit = "background"
            if len(gt_other):
                ious_o = iou(b, gt_other[:, 1:]); k = ious_o.argmax()
                if ious_o[k] >= t:
                    hit = OUR_NAMES[int(gt_other[k, 0])]
            S["fp_on_class"][hit] = S["fp_on_class"].get(hit, 0) + 1
        for k, g in enumerate(gt_stop):
            key = size_bin((g[3] - g[1]) * scale)
            S["size"].setdefault(key, [0, 0])
            S["size"][key][0] += int(used[k]); S["size"][key][1] += 1

    # Save a few example images that contain a stop sign
    if len(gt_stop) and n_samples_saved < N_SAMPLES:
        im = r.orig_img.copy()
        th = max(3, W // 500)
        for g in gt_stop:
            cv2.rectangle(im, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), th)
        for b, c in zip(preds, conf):
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), th)
            cv2.putText(im, f"stop {c:.2f}", (int(b[0]), max(int(b[1]) - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, W / 1500, (0, 0, 255), th)
        im = cv2.resize(im, (1600, int(H * 1600 / W)))
        cv2.imwrite(str(OUT_DIR / "samples" / (Path(r.path).stem + ".jpg")), im)
        n_samples_saved += 1

with open(OUT_DIR / "predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_index", "image", "x1", "y1", "x2", "y2", "class_name", "confidence"])
    w.writerows(pred_rows)

dev_name = torch.cuda.get_device_name(0) if DEVICE == 0 else platform.processor()
lines = [f"Model: {MODEL} (COCO, class {COCO_STOP} 'stop sign') | ultralytics {ultralytics.__version__} | torch {torch.__version__}",
         f"Device: {dev_name} | imgsz={IMGSZ} conf={CONF} | val images: {len(img_paths)}",
         f"Mean inference time: {np.mean(speeds):.1f} ms/image (~{1000 / np.mean(speeds):.1f} FPS, model only)",
         "Evaluated class: stop only (COCO has no other of our classes)",
         "Size bins: GT box height in px at model input (longest side = imgsz)"]
for t, S in stats.items():
    n_gt = sum(v[1] for v in S["size"].values())
    lines.append(f"\n--- IoU >= {t} (stop only) ---")
    lines.append(f"Precision {S['TP'] / max(S['TP'] + S['FP'], 1):.3f} | Recall {S['TP'] / max(n_gt, 1):.3f}"
                 f" | TP {S['TP']} FP {S['FP']} GT {n_gt}")
    lines.append("Recall by size:")
    for k in ["<10px", "10-20px", ">=20px"]:
        if k in S["size"]:
            a, b = S["size"][k]
            lines.append(f"   {k:8s} {a / max(b, 1):.3f}  ({a}/{b})")
    lines.append("False positives located on:")
    for k, v in sorted(S["fp_on_class"].items(), key=lambda kv: -kv[1]):
        lines.append(f"   {k:14s} {v}")
txt = "\n".join(lines)
print(txt)
(OUT_DIR / "results.txt").write_text(txt)
print(f"\nSaved results, {len(pred_rows)} predictions and {n_samples_saved} sample images to {OUT_DIR}")
