"""Evaluate the fine-tuned sign detector on val and test.

Two kinds of evaluation:
  1) Standard ultralytics metrics (P, R, mAP50, mAP50-95 per class) on val and test.
  2) Our own matching at a FIXED conf=0.25 and IoU>=0.5, the same rule as
     04_baseline_coco.py, so the stop class can be compared with the COCO baseline.
     A prediction is correct only if its class equals the GT class.
     Recall is also split by GT box height at model input (<10, 10-20, >=20 px).

Usage (repo root, venv active):
  python src/traffic_signs/06_evaluate.py --run y11n_1280

Outputs in results/traffic_signs/train_<run>/:
  metrics_ultralytics.txt   standard metrics on val and test
  metrics_conf025.txt       our fixed-threshold metrics on val and test
  predictions_val.csv, predictions_test.csv   (common detection format)
  copies of training plots (results.png, confusion matrix, PR curve, ...)
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

DATA_DIR = Path(r"C:\data\signs_yolo")
DATA_YAML = str(DATA_DIR / "data.yaml")
NAMES = ["stop", "yield", "speed_limit", "no_entry", "priority_road", "no_parking"]
IMGSZ, CONF, IOU_T = 1280, 0.25, 0.5
SIZE_BINS = ["<10px", "10-20px", ">=20px"]
PLOTS = ["results.png", "results.csv", "args.yaml", "confusion_matrix.png",
         "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]
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


def ultralytics_metrics(model, split, project, run):
    """Run ultralytics validation and return report lines."""
    m = model.val(data=DATA_YAML, split=split, imgsz=IMGSZ, batch=8, device=DEVICE,
                  project=project, name=f"{run}_eval_{split}", exist_ok=True, verbose=False)
    lines = [f"\n=== {split} (ultralytics val, imgsz={IMGSZ}) ===",
             f"{'class':14s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>8s}",
             f"{'all':14s} {m.box.mp:6.3f} {m.box.mr:6.3f} {m.box.map50:6.3f} {m.box.map:8.3f}"]
    for i, cid in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        lines.append(f"{NAMES[cid]:14s} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:8.3f}")
    return lines


def fixed_conf_metrics(model, split, out_dir):
    """Our own class-aware matching at conf=CONF, IoU>=IOU_T. Also saves predictions."""
    tp = {c: 0 for c in range(len(NAMES))}
    fp = {c: 0 for c in range(len(NAMES))}
    n_gt = {c: 0 for c in range(len(NAMES))}
    size = {b: [0, 0] for b in SIZE_BINS}           # [found, total] over all classes
    pred_rows = []

    for frame_index, r in enumerate(model.predict(source=str(DATA_DIR / "images" / split), batch=1,
                                                  imgsz=IMGSZ, conf=CONF, device=DEVICE,
                                                  stream=True, verbose=False)):
        H, W = r.orig_shape
        scale = IMGSZ / max(W, H)
        gt = read_gt(DATA_DIR / "labels" / split / (Path(r.path).stem + ".txt"), W, H)
        conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-conf)
        boxes = r.boxes.xyxy.cpu().numpy()[order]
        classes = r.boxes.cls.cpu().numpy().astype(int)[order]
        conf = conf[order]

        used = np.zeros(len(gt), bool)
        for b, c, s in zip(boxes, classes, conf):
            pred_rows.append([frame_index, Path(r.path).name, f"{b[0]:.1f}", f"{b[1]:.1f}",
                              f"{b[2]:.1f}", f"{b[3]:.1f}", NAMES[c], f"{s:.4f}"])
            same = (gt[:, 0] == c) & ~used if len(gt) else np.zeros(0, bool)
            if same.any():
                ious = iou(b, gt[:, 1:]); ious[~same] = 0; k = ious.argmax()
                if ious[k] >= IOU_T:
                    used[k] = True; tp[c] += 1
                    continue
            fp[c] += 1
        for k, g in enumerate(gt):
            n_gt[int(g[0])] += 1
            sb = size_bin((g[4] - g[2]) * scale)
            size[sb][0] += int(used[k]); size[sb][1] += 1

    with open(out_dir / f"predictions_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "image", "x1", "y1", "x2", "y2", "class_name", "confidence"])
        w.writerows(pred_rows)

    lines = [f"\n=== {split} (conf={CONF}, IoU>={IOU_T}, class-aware) ===",
             f"{'class':14s} {'P':>6s} {'R':>6s} {'TP':>4s} {'FP':>4s} {'GT':>4s}"]
    for c in list(range(len(NAMES))) + ["all"]:
        t = sum(tp.values()) if c == "all" else tp[c]
        f_ = sum(fp.values()) if c == "all" else fp[c]
        g = sum(n_gt.values()) if c == "all" else n_gt[c]
        name = "all" if c == "all" else NAMES[c]
        lines.append(f"{name:14s} {t / max(t + f_, 1):6.3f} {t / max(g, 1):6.3f} {t:4d} {f_:4d} {g:4d}")
    lines.append("Recall by GT height at model input (all classes):")
    for b in SIZE_BINS:
        a, n = size[b]
        lines.append(f"   {b:8s} {a / max(n, 1):.3f}  ({a}/{n})")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run name, e.g. y11n_1280")
    args = p.parse_args()

    run_dir = Path("runs") / "traffic_signs" / args.run
    out_dir = Path("results") / "traffic_signs" / f"train_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    project = str((Path("runs") / "traffic_signs").resolve())
    model = YOLO(str(run_dir / "weights" / "best.pt"))

    # Copy training plots so they can be committed (runs/ is ignored by git)
    for name in PLOTS:
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, out_dir / name)

    header = [f"Model: {run_dir / 'weights' / 'best.pt'}",
              "Note: best.pt was selected on val, so val numbers are slightly optimistic; test is the fair number."]
    ul = header + ultralytics_metrics(model, "val", project, args.run) \
        + ultralytics_metrics(model, "test", project, args.run)
    fc = header + fixed_conf_metrics(model, "val", out_dir) + fixed_conf_metrics(model, "test", out_dir)

    (out_dir / "metrics_ultralytics.txt").write_text("\n".join(ul) + "\n")
    (out_dir / "metrics_conf025.txt").write_text("\n".join(fc) + "\n")
    print("\n".join(ul)); print("\n".join(fc[len(header):]))
    print(f"\nSaved metrics, predictions and plots to {out_dir}")


# The guard is required on Windows: data loader workers re-import this file.
if __name__ == "__main__":
    main()
