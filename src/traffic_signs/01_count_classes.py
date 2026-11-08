import csv
import json
import os
from collections import Counter, defaultdict

IMG_DIR = r"C:\data\mtsd\images"
ANN_DIR = r"C:\data\mtsd_ann\mtsd_v2_fully_annotated\annotations"
OUT_CSV = os.path.join("results", "traffic_signs", "class_counts_all.csv")

object_counts = Counter()
image_sets = defaultdict(set)
n_images = 0
n_objects = 0

for img_name in os.listdir(IMG_DIR):
    key, ext = os.path.splitext(img_name)
    if ext.lower() != ".jpg":
        continue
    ann_path = os.path.join(ANN_DIR, key + ".json")
    if not os.path.exists(ann_path):
        continue
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_images += 1
    for obj in data["objects"]:
        object_counts[obj["label"]] += 1
        image_sets[obj["label"]].add(key)
        n_objects += 1

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
rows = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "n_objects", "n_images"])
    for label, n in rows:
        writer.writerow([label, n, len(image_sets[label])])

print("Images:", n_images)
print("Objects:", n_objects)
print("Distinct labels:", len(object_counts))
print()
print("Top 40 labels (label, objects, images):")
for label, n in rows[:40]:
    print(f"{label:60s} {n:6d} {len(image_sets[label]):6d}")

print()
print("Candidate classes (keyword, objects):")
for kw in ["stop", "yield", "maximum-speed-limit", "go-straight", "no-u-turn", "warning--"]:
    total = sum(n for label, n in object_counts.items() if kw in label)
    print(f"{kw:25s} {total}")