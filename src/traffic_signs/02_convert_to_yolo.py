import csv
import json
import os
import random
import re
import shutil
from collections import Counter

IMG_DIR = r"C:\data\mtsd\images"
ANN_DIR = r"C:\data\mtsd_ann\mtsd_v2_fully_annotated\annotations"
OUT_DIR = r"C:\data\signs_yolo"
CLASS_MAP = os.path.join("src", "traffic_signs", "class_map.yaml")
STATS_CSV = os.path.join("results", "traffic_signs", "split_class_counts.csv")
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Read class_map.yaml without extra dependencies
names, rules = {}, {}
for line in open(CLASS_MAP, encoding="utf-8"):
    m_rule = re.match(r'^\s*"([^"]+)":\s*(\d+)', line)
    m_name = re.match(r"^\s+(\d+):\s*(\w+)", line)
    if m_rule:
        rules[m_rule.group(1)] = int(m_rule.group(2))
    elif m_name:
        names[int(m_name.group(1))] = m_name.group(2)

print("Classes:", names)
print("Rules:", len(rules))


def map_label(label):
    for prefix, cid in rules.items():
        if label.startswith(prefix):
            return cid
    return None


# Convert annotations of every image that contains at least one target sign
usable = []
skipped_boxes = 0
for img_name in os.listdir(IMG_DIR):
    key, ext = os.path.splitext(img_name)
    if ext.lower() != ".jpg":
        continue
    ann_path = os.path.join(ANN_DIR, key + ".json")
    if not os.path.exists(ann_path):
        continue
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    W, H = data["width"], data["height"]
    lines = []
    for obj in data["objects"]:
        cid = map_label(obj["label"])
        if cid is None:
            continue
        b = obj["bbox"]
        x1, y1 = max(0.0, b["xmin"]), max(0.0, b["ymin"])
        x2, y2 = min(float(W), b["xmax"]), min(float(H), b["ymax"])
        if x2 <= x1 or y2 <= y1:
            skipped_boxes += 1
            continue
        xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    if lines:
        usable.append((img_name, key, lines))

print("Images with at least one target sign:", len(usable))
print("Skipped invalid boxes:", skipped_boxes)

# Random split by image
random.seed(SEED)
random.shuffle(usable)
n = len(usable)
n_train = int(n * SPLIT_RATIOS["train"])
n_val = int(n * SPLIT_RATIOS["val"])
splits = {
    "train": usable[:n_train],
    "val": usable[n_train:n_train + n_val],
    "test": usable[n_train + n_val:],
}

# Write images and label files
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
stats = {}
for split, items in splits.items():
    os.makedirs(os.path.join(OUT_DIR, "images", split))
    os.makedirs(os.path.join(OUT_DIR, "labels", split))
    counts = Counter()
    for img_name, key, lines in items:
        shutil.copy(os.path.join(IMG_DIR, img_name),
                    os.path.join(OUT_DIR, "images", split, img_name))
        with open(os.path.join(OUT_DIR, "labels", split, key + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        for ln in lines:
            counts[int(ln.split()[0])] += 1
    stats[split] = (len(items), counts)

# data.yaml for YOLO training
with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
    f.write("path: " + OUT_DIR.replace("\\", "/") + "\n")
    f.write("train: images/train\nval: images/val\ntest: images/test\n")
    f.write("names:\n")
    for cid in sorted(names):
        f.write(f"  {cid}: {names[cid]}\n")

# Print and save split statistics
os.makedirs(os.path.dirname(STATS_CSV), exist_ok=True)
with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "n_images"] + [names[c] for c in sorted(names)])
    print()
    print(f"{'split':8s} {'images':>7s} " + " ".join(f"{names[c]:>13s}" for c in sorted(names)))
    for split, (n_img, counts) in stats.items():
        row = [counts[c] for c in sorted(names)]
        writer.writerow([split, n_img] + row)
        print(f"{split:8s} {n_img:7d} " + " ".join(f"{v:13d}" for v in row))

print()
print("Done. Output folder:", OUT_DIR)