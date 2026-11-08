import csv
import glob
import os
import random
import re
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\data\signs_yolo"
CLASS_MAP = os.path.join("src", "traffic_signs", "class_map.yaml")
OUT_SAMPLES = os.path.join("results", "traffic_signs", "gt_samples")
OUT_CHART = os.path.join("results", "traffic_signs", "class_distribution.png")
OUT_SIZES = os.path.join("results", "traffic_signs", "object_sizes.csv")
N_SAMPLES = 12
SEED = 42

# Read class names from class_map.yaml
names = {}
for line in open(CLASS_MAP, encoding="utf-8"):
    m = re.match(r"^\s+(\d+):\s*(\w+)", line)
    if m and '"' not in line:
        names[int(m.group(1))] = m.group(2)
class_ids = sorted(names)


def read_labels(path):
    boxes = []
    for line in open(path):
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


# 1) Draw ground-truth boxes on random training images
os.makedirs(OUT_SAMPLES, exist_ok=True)
random.seed(SEED)
img_paths = glob.glob(os.path.join(DATA_DIR, "images", "train", "*.jpg"))
for p in random.sample(img_paths, N_SAMPLES):
    img = cv2.imread(p)
    H, W = img.shape[:2]
    key = os.path.splitext(os.path.basename(p))[0]
    label_path = os.path.join(DATA_DIR, "labels", "train", key + ".txt")
    thick = max(3, W // 500)
    for cid, xc, yc, w, h in read_labels(label_path):
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)
        cv2.putText(img, names[cid], (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, W / 1500, (0, 255, 0), thick)
    scale = 1600 / W
    img = cv2.resize(img, (1600, int(H * scale)))
    cv2.imwrite(os.path.join(OUT_SAMPLES, key + ".jpg"), img)
print("Saved", N_SAMPLES, "sample images to", OUT_SAMPLES)

# 2) Class distribution per split
counts = {}
sizes = []
for split in ["train", "val", "test"]:
    c = Counter()
    for lp in glob.glob(os.path.join(DATA_DIR, "labels", split, "*.txt")):
        for cid, xc, yc, w, h in read_labels(lp):
            c[cid] += 1
            sizes.append((split, cid, w, h))
    counts[split] = c

fig, ax = plt.subplots(figsize=(9, 5))
bottom = [0] * len(class_ids)
for split in ["train", "val", "test"]:
    vals = [counts[split][c] for c in class_ids]
    ax.bar([names[c] for c in class_ids], vals, bottom=bottom, label=split)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel("Number of annotated signs")
ax.set_title("Class distribution per split")
ax.legend()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_CHART, dpi=150)
print("Saved chart to", OUT_CHART)

# 3) Object size analysis (normalized box width, and approximate pixels at imgsz=960)
widths = sorted(w for _, _, w, _ in sizes)
n = len(widths)
median_w = widths[n // 2]
print()
print("Total boxes:", n)
print(f"Median box width: {median_w:.4f} of image width (~{median_w * 960:.0f} px at 960 px input)")
for thr_px in (16, 32):
    frac = sum(1 for w in widths if w * 960 < thr_px) / n
    print(f"Boxes narrower than {thr_px} px at 960 px input: {frac * 100:.1f}%")

with open(OUT_SIZES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "class", "w_norm", "h_norm"])
    for split, cid, w, h in sizes:
        writer.writerow([split, names[cid], f"{w:.6f}", f"{h:.6f}"])
print("Saved box sizes to", OUT_SIZES)