"""
Evaluation module (Person 4): one set of matching rules and metrics for every detector.

Common box format (pixels): image, x1, y1, x2, y2, class_name  (+ confidence for predictions)

Matching rule (same as Person 1 and Person 2 used, so numbers can be reproduced):
  * predictions with confidence >= CONF are kept
  * per image, predictions are visited from highest to lowest confidence (greedy)
  * a prediction is a TP if it overlaps a not-yet-used GT box with IoU >= IOU_T
      - class-aware:    the GT box must also have the same class
      - class-agnostic: the class is ignored (measures "was an object found?")
  * every other prediction is an FP; every GT box left unused is an FN

Usage (full commands: src/evaluation/README.md, section 8):
  python evaluation.py lights       --pred pred_labels.zip      --gt val_gt_labels.zip --out ... --size-dim w
  python evaluation.py signs        --pred predictions_test.csv --gt gt_test.csv       --out ... --size-dim h --sizes image_sizes.csv
  python evaluation.py lights_video --pred detections.csv       --gt test.yaml         --out ... --size-dim w

Size bins: --size-dim w = box width, h = box height (pixels at model input), none = skip.
  Lights are binned by WIDTH (Person 1), signs by HEIGHT rescaled to imgsz (Person 2, needs --sizes).
  The two definitions differ, so light and sign size results are not compared.

lights_video: the video pipeline's detections.csv (Person 3) on the Bosch TEST sequence, matched
  against Bosch's test.yaml. Frame i of the video is image 24068 + 2*i. Bosch labels are mapped to
  4 states like Person 1 did (Red* -> red, Green* -> green, Yellow* -> yellow, off -> off).
  Bosch labels have no track ids, so identity metrics (IDF1, ID switches) cannot be computed.

Note: prediction files are cut at conf >= 0.25, so mAP50 must NOT be computed from them
(the PR curve is truncated); use Ultralytics val (conf=0.001) for mAP50.
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

IOU_T = 0.5
CONF = 0.25
LIGHT_CLASSES = {0: "red", 1: "yellow", 2: "green", 3: "off"}   # label_mapping of Person 1
LIGHT_W, LIGHT_H = 1280, 720                                    # Bosch Small Traffic Lights images
SIZE_BINS = [("<10px", 0, 10), ("10-20px", 10, 20), (">=20px", 20, 1e9)]


# ----------------------------------------------------------------------------- loaders
def load_yolo_txt(path, W=LIGHT_W, H=LIGHT_H):
    """YOLO txt files in a folder or zip -> pixel boxes.
    GT lines: 'cls cx cy w h'; prediction lines: 'cls cx cy w h conf' (normalized coordinates).
    Prediction txt files keep 6 decimals, so they are preferred over the 1-decimal CSV
    (the rounding in the CSV changed one borderline match: 1200 vs 1201 TP)."""
    rows = []
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            items = [(Path(n).stem, z.read(n).decode()) for n in z.namelist() if n.endswith(".txt")]
    else:
        items = [(f.stem, f.read_text()) for f in path.rglob("*.txt")]
    for stem, text in items:
        for line in text.splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            row = [stem, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H,
                   LIGHT_CLASSES[c]]
            if len(p) >= 6:
                row.append(float(p[5]))
            rows.append(row)
    cols = ["image", "x1", "y1", "x2", "y2", "class_name"]
    df = pd.DataFrame(rows)
    df.columns = cols + (["confidence"] if df.shape[1] == 7 else [])
    return df


def load_csv(path):
    df = pd.read_csv(path)
    df["image"] = df["image"].astype(str).str.replace(r"\.(jpg|png|jpeg)$", "", regex=True)
    return df


def bosch_state(label):
    """Bosch label -> 4 states, the same rule as Person 1's map_state (src/traffic_lights/inspect_labels.py)."""
    l = label.lower()
    for s in ("red", "yellow", "green"):
        if l.startswith(s):
            return s
    return "off" if l == "off" else None


def load_bosch_yaml(path, first=24068, step=2, W=LIGHT_W, H=LIGHT_H):
    """Bosch test.yaml -> GT boxes keyed by video frame index (image number = first + step * frame).
    Boxes are sorted and clipped to the image and boxes under 1 px are dropped, like Person 1's
    convert_to_yolo.py. Keeps Bosch's 'occluded' flag."""
    import yaml
    rows = []
    for e in yaml.safe_load(open(path)):
        num = int(Path(e["path"]).stem)
        frame = (num - first) // step
        for b in e["boxes"]:
            s = bosch_state(b["label"])
            x1, x2 = sorted([b["x_min"], b["x_max"]])
            y1, y2 = sorted([b["y_min"], b["y_max"]])
            x1, x2, y1, y2 = max(0, x1), min(W, x2), max(0, y1), min(H, y2)
            if s is None or x2 - x1 < 1 or y2 - y1 < 1:
                continue
            rows.append([str(frame), x1, y1, x2, y2, s, bool(b["occluded"])])
    return pd.DataFrame(rows, columns=["image", "x1", "y1", "x2", "y2", "class_name", "occluded"])


def load_video_csv(path, source="light"):
    """detections.csv of the video pipeline (Person 3) -> rows of one model, keyed by frame index."""
    df = pd.read_csv(path)
    df = df[df["source"] == source].copy()
    df["image"] = df["frame_index"].astype(str)
    return df[["image", "x1", "y1", "x2", "y2", "class_name", "confidence"]]


# ----------------------------------------------------------------------------- matching
def _iou(b, B):
    ix = np.maximum(0, np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]))
    iy = np.maximum(0, np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]))
    inter = ix * iy
    union = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def match(pred, gt, class_aware, conf=CONF, iou_t=IOU_T):
    """Greedy matching. Returns (pred with 'gt_idx' column (-1 = FP), gt with 'matched' column)."""
    pred = pred[pred["confidence"] >= conf].copy()
    pred["gt_idx"] = -1
    gt = gt.reset_index(drop=True).copy()
    gt["matched"] = False
    gt_by_img = {k: v.index.to_numpy() for k, v in gt.groupby("image")}
    gxy = gt[["x1", "y1", "x2", "y2"]].to_numpy()
    gcls = gt["class_name"].to_numpy()
    used = np.zeros(len(gt), bool)
    for img, pg in pred.groupby("image"):
        idx = gt_by_img.get(img)
        if idx is None:
            continue
        for pi, r in pg.sort_values("confidence", ascending=False).iterrows():
            io_ = _iou([r.x1, r.y1, r.x2, r.y2], gxy[idx])
            io_[used[idx]] = 0
            if class_aware:
                io_[gcls[idx] != r.class_name] = 0
            k = io_.argmax()
            if io_[k] >= iou_t:
                used[idx[k]] = True
                pred.at[pi, "gt_idx"] = idx[k]
    gt["matched"] = used
    return pred, gt


# ----------------------------------------------------------------------------- metrics
def add_size(gt, dim="h", scale=None):
    """Box size in pixels at model input. scale = per-row factor (1.0 if image is not resized)."""
    s = 1.0 if scale is None else scale
    gt = gt.copy()
    if dim is None:                      # image sizes unknown -> no size bins
        gt["size_px"] = np.nan
        return gt
    gt["size_px"] = ((gt["y2"] - gt["y1"]) if dim == "h" else (gt["x2"] - gt["x1"])) * s
    return gt


def summary(pred_m, gt_m):
    tp = int((pred_m["gt_idx"] >= 0).sum())
    fp = len(pred_m) - tp
    fn = int((~gt_m["matched"]).sum())
    return dict(TP=tp, FP=fp, FN=fn, GT=len(gt_m),
                precision=tp / max(tp + fp, 1), recall=tp / max(len(gt_m), 1))


def recall_by_size(gt_m):
    out = []
    for name, lo, hi in SIZE_BINS:
        s = gt_m[(gt_m["size_px"] >= lo) & (gt_m["size_px"] < hi)]
        out.append(dict(size_bin=name, found=int(s["matched"].sum()), total=len(s),
                        recall=s["matched"].mean() if len(s) else np.nan))
    return pd.DataFrame(out)


def recall_by_occlusion(gt_m):
    """Recall for occluded vs. not occluded GT boxes (only when the GT has an 'occluded' flag)."""
    g = gt_m.groupby("occluded")["matched"].agg(found="sum", total="count")
    g["recall"] = g["found"] / g["total"]
    return g.reset_index()


def recall_by_class(gt_m):
    g = gt_m.groupby("class_name")["matched"].agg(found="sum", total="count")
    g["recall"] = g["found"] / g["total"]
    return g.reset_index()


def confusion(pred_m, gt_m, classes):
    """Rows = true class (+ 'FP' for predictions on no object); cols = predicted class (+ 'missed').
    Built from class-agnostic matches, so a wrong colour/type shows up off the diagonal."""
    cm = pd.DataFrame(0, index=classes + ["FP"], columns=classes + ["missed"])
    for _, r in pred_m.iterrows():
        t = gt_m.at[r.gt_idx, "class_name"] if r.gt_idx >= 0 else "FP"
        cm.at[t, r.class_name] += 1
    for _, r in gt_m[~gt_m["matched"]].iterrows():
        cm.at[r.class_name, "missed"] += 1
    return cm


def evaluate(pred, gt, classes, dim="h", scale=None, iou_t=IOU_T):
    res = {}
    for aware in (False, True):
        p, g = match(pred, gt, class_aware=aware, iou_t=iou_t)
        g = add_size(g, dim, scale)
        key = "class_aware" if aware else "class_agnostic"
        res[key] = dict(summary=summary(p, g), by_size=recall_by_size(g), by_class=recall_by_class(g))
        if "occluded" in g.columns:
            res[key]["by_occlusion"] = recall_by_occlusion(g)
        if not aware:
            res["confusion"] = confusion(p, g, classes)
            res["accuracy"] = classification_accuracy(res["confusion"], classes)
    return res


def classification_accuracy(cm, classes):
    """Among detected objects (class-agnostic matches), share whose class/state was correct.
    Separates 'wrong colour/type' errors from detection errors (missed / FP)."""
    m = cm.loc[classes, classes]
    correct, matched = int(np.trace(m.to_numpy())), int(m.to_numpy().sum())
    per = {c: (m.at[c, c] / m.loc[c].sum()) if m.loc[c].sum() else np.nan for c in classes}
    return dict(correct=correct, matched=matched, accuracy=correct / max(matched, 1), **per)


def write(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(matching=k, **res[k]["summary"]) for k in ("class_agnostic", "class_aware")]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame([res["accuracy"]]).to_csv(out / "classification_accuracy.csv", index=False)
    for k in ("class_agnostic", "class_aware"):
        if res[k]["by_size"]["total"].sum() > 0:          # skipped when image sizes are unknown
            res[k]["by_size"].to_csv(out / f"recall_by_size_{k}.csv", index=False)
        res[k]["by_class"].to_csv(out / f"recall_by_class_{k}.csv", index=False)
        if "by_occlusion" in res[k]:
            res[k]["by_occlusion"].to_csv(out / f"recall_by_occlusion_{k}.csv", index=False)
    res["confusion"].to_csv(out / "confusion_matrix.csv")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lights", "signs", "lights_video"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-dim", default="h", choices=["h", "w", "none"])
    ap.add_argument("--sizes", default=None,
                    help="CSV image,width,height (original px). Box sizes are scaled to model input.")
    ap.add_argument("--imgsz", type=int, default=1280, help="model input size (longest side)")
    ap.add_argument("--iou", type=float, default=IOU_T,
                    help="IoU threshold for a match (protocol: 0.5; other values are sensitivity checks only)")
    a = ap.parse_args()
    if a.kind == "lights_video":
        pred, gt = load_video_csv(a.pred), load_bosch_yaml(a.gt)
        classes = list(LIGHT_CLASSES.values())
    elif a.kind == "lights":
        pred = load_csv(a.pred) if a.pred.endswith(".csv") else load_yolo_txt(a.pred)
        gt, classes = load_yolo_txt(a.gt), list(LIGHT_CLASSES.values())
    else:
        pred, gt = load_csv(a.pred), load_csv(a.gt)
        classes = ["stop", "yield", "speed_limit", "no_entry", "priority_road", "no_parking"]
    scale = None
    if a.sizes:
        sz = load_csv(a.sizes)                                  # same image-name cleaning as gt
        g = gt[["image"]].merge(sz, on="image", how="left")     # keeps gt row order
        if g["width"].isna().any():
            raise SystemExit(f"{g['width'].isna().sum()} gt boxes have no image size in {a.sizes}")
        scale = (a.imgsz / g[["width", "height"]].max(axis=1)).to_numpy()
    res = evaluate(pred, gt, classes, dim=None if a.size_dim == "none" else a.size_dim, scale=scale,
                   iou_t=a.iou)
    print(write(res, a.out).to_string(index=False))
    print(res["class_agnostic"]["by_size"].to_string(index=False))
    if "by_occlusion" in res["class_agnostic"]:
        print(res["class_agnostic"]["by_occlusion"].to_string(index=False))
    print(res["confusion"].to_string())


if __name__ == "__main__":
    main()
