"""Error analysis of the fine-tuned sign detector on the val split (conf=0.25, IoU>=0.5).

Why not use the ultralytics confusion matrix? It is computed at a very low confidence
threshold, so its 'background' column is huge and not representative of real use.

What this script does:
  1) Confusion matrix at conf=0.25. Each prediction is matched to the best unused GT box
     of ANY class (IoU>=0.5). Matched -> cell [true class, predicted class].
     Unmatched prediction -> false positive (true = background).
     Unmatched GT -> missed (predicted = background).
  2) For every false positive, look at the ORIGINAL MTSD annotation of that image:
     does the box overlap (IoU>=0.5) a sign that we did not use as a class
     (e.g. other-sign, no-stopping)? Then it is a real sign, just not one of our 6 classes.
  3) Missed signs are listed with their size at model input.
  4) Saves example crops of each error type.

Usage (repo root, venv active):
  python src/traffic_signs/07_error_analysis.py --run y11n_1280

Outputs in results/traffic_signs/error_analysis_<run>/
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

DATA_DIR = Path(r"C:\data\signs_yolo")
ANN_DIR = Path(r"C:\data\mtsd_ann\mtsd_v2_fully_annotated\annotations")
NAMES = ["stop", "yield", "speed_limit", "no_entry", "priority_road", "no_parking"]
BG = len(NAMES)                      # index used for 'background'
IMGSZ, CONF, IOU_T = 1280, 0.25, 0.5
N_EXAMPLES = 3
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


def read_mtsd_other(key):
    """All MTSD objects of this image as (label, [x1,y1,x2,y2]); label without '--gN' suffix."""
    data = json.loads((ANN_DIR / f"{key}.json").read_text(encoding="utf-8"))
    out = []
    for o in data["objects"]:
        b = o["bbox"]
        out.append((re.sub(r"--g\d+$", "", o["label"]), [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]))
    return out


def iou(box, boxes):
    """IoU between one box [x1,y1,x2,y2] and an array of boxes (N,4)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def size_bin(h_px):
    return "<10px" if h_px < 10 else "10-20px" if h_px < 20 else ">=20px"


def save_crop(img, box, text, path):
    """Save a zoomed crop around a box (box drawn in red) with a caption."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    side = max(x2 - x1, y2 - y1) * 4 + 200          # context around the box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    a, b = max(cx - side // 2, 0), max(cy - side // 2, 0)
    c, d = min(cx + side // 2, W), min(cy + side // 2, H)
    crop = img[b:d, a:c].copy()
    cv2.rectangle(crop, (x1 - a, y1 - b), (x2 - a, y2 - b), (0, 0, 255), max(2, side // 150))
    crop = cv2.resize(crop, (400, int(400 * crop.shape[0] / crop.shape[1])))
    cv2.rectangle(crop, (0, 0), (400, 28), (0, 0, 0), -1)
    cv2.putText(crop, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(path), crop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run name, e.g. y11n_1280")
    args = p.parse_args()

    out_dir = Path("results") / "traffic_signs" / f"error_analysis_{args.run}"
    ex_dir = out_dir / "examples"
    for sub in ["confusion", "fp_on_unused_sign", "fp_on_nothing", "missed_small"]:
        (ex_dir / sub).mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path("runs") / "traffic_signs" / args.run / "weights" / "best.pt"))

    cm = np.zeros((BG + 1, BG + 1), int)             # rows: true, cols: predicted
    fp_rows, missed_rows = [], []
    examples = Counter()

    for r in model.predict(source=str(DATA_DIR / "images" / "val"), batch=1, imgsz=IMGSZ,
                           conf=CONF, device=DEVICE, stream=True, verbose=False):
        H, W = r.orig_shape
        scale = IMGSZ / max(W, H)
        key = Path(r.path).stem
        gt = read_gt(DATA_DIR / "labels" / "val" / f"{key}.txt", W, H)
        conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-conf)
        boxes = r.boxes.xyxy.cpu().numpy()[order]
        classes = r.boxes.cls.cpu().numpy().astype(int)[order]
        conf = conf[order]
        mtsd = None

        used = np.zeros(len(gt), bool)
        for b, c, s in zip(boxes, classes, conf):
            if len(gt):
                ious = iou(b, gt[:, 1:]); ious[used] = 0; k = ious.argmax()
                if ious[k] >= IOU_T:
                    used[k] = True
                    t = int(gt[k, 0]); cm[t, c] += 1
                    if t != c and examples["confusion"] < N_EXAMPLES:
                        save_crop(r.orig_img, b, f"true {NAMES[t]} -> pred {NAMES[c]} {s:.2f}",
                                  ex_dir / "confusion" / f"{key}_{NAMES[t]}_as_{NAMES[c]}.jpg")
                        examples["confusion"] += 1
                    continue
            # False positive: does it overlap a sign we did not use as a class?
            cm[BG, c] += 1
            if mtsd is None:
                mtsd = read_mtsd_other(key)
            hit = "none"
            if mtsd:
                ious_m = iou(b, [m[1] for m in mtsd]); k = ious_m.argmax()
                if ious_m[k] >= IOU_T:
                    hit = mtsd[k][0]
            fp_rows.append([key, NAMES[c], f"{s:.3f}", f"{(b[3] - b[1]) * scale:.1f}", hit])
            sub = "fp_on_nothing" if hit == "none" else "fp_on_unused_sign"
            if examples[sub] < N_EXAMPLES:
                save_crop(r.orig_img, b, f"pred {NAMES[c]} {s:.2f} | MTSD: {hit}"[:60],
                          ex_dir / sub / f"{key}_{NAMES[c]}.jpg")
                examples[sub] += 1

        for k, g in enumerate(gt):
            if used[k]:
                continue
            t = int(g[0]); h_px = (g[4] - g[2]) * scale
            cm[t, BG] += 1
            missed_rows.append([key, NAMES[t], f"{h_px:.1f}", size_bin(h_px)])
            if h_px < 10 and examples["missed_small"] < N_EXAMPLES:
                save_crop(r.orig_img, g[1:], f"missed {NAMES[t]} ({h_px:.1f} px at input)",
                          ex_dir / "missed_small" / f"{key}_{NAMES[t]}.jpg")
                examples["missed_small"] += 1

    # Save tables
    labels = NAMES + ["background"]
    with open(out_dir / "confusion_matrix_conf025.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true \\ predicted"] + labels)
        for i, name in enumerate(labels):
            w.writerow([name] + list(cm[i]))
    with open(out_dir / "false_positives.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "pred_class", "confidence", "height_px_at_input", "mtsd_label_overlap"])
        w.writerows(fp_rows)
    with open(out_dir / "missed.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "true_class", "height_px_at_input", "size_bin"])
        w.writerows(missed_rows)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(BG + 1):
        for j in range(BG + 1):
            if i == BG and j == BG:
                continue
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(BG + 1), labels, rotation=35, ha="right")
    ax.set_yticks(range(BG + 1), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Val confusion matrix (conf={CONF}, IoU>={IOU_T})")
    plt.tight_layout(); plt.savefig(out_dir / "confusion_matrix_conf025.png", dpi=150)

    # Summary
    n_fp = len(fp_rows)
    on_sign = Counter(row[4] for row in fp_rows if row[4] != "none")
    lines = [f"Val split, conf={CONF}, IoU>={IOU_T}",
             "Confusion matrix (rows true, cols predicted):",
             "               " + " ".join(f"{n[:8]:>8s}" for n in labels)]
    for i, name in enumerate(labels):
        lines.append(f"{name:14s} " + " ".join(f"{v:8d}" for v in cm[i]))
    off_diag = sum(cm[i, j] for i in range(BG) for j in range(BG) if i != j)
    lines += [f"\nCorrect: {np.trace(cm[:BG, :BG])} | wrong class: {off_diag} | "
              f"false positives: {n_fp} | missed: {len(missed_rows)}",
              f"\nFalse positives on an unused MTSD sign: {sum(on_sign.values())} / {n_fp}",
              f"False positives on no annotated sign: {n_fp - sum(on_sign.values())} / {n_fp}",
              "Most frequent unused MTSD labels under false positives:"]
    lines += [f"   {n:3d}  {lab}" for lab, n in on_sign.most_common(10)]
    lines.append("\nMissed signs by size (height at model input):")
    for sb, n in sorted(Counter(row[3] for row in missed_rows).items()):
        lines.append(f"   {sb:8s} {n}")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    print(f"\nSaved tables, figure and examples to {out_dir}")


# The guard is required on Windows: data loader workers re-import this file.
if __name__ == "__main__":
    main()
