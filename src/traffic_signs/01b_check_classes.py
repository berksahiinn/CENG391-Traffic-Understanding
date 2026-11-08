import json
import os
from collections import Counter, defaultdict

IMG_DIR = r"C:\data\mtsd\images"
ANN_DIR = r"C:\data\mtsd_ann\mtsd_v2_fully_annotated\annotations"

RULES = {
    "stop": "regulatory--stop--",
    "yield": "regulatory--yield--",
    "speed_limit": "regulatory--maximum-speed-limit-",
    "no_entry": "regulatory--no-entry--",
    "priority_road": "regulatory--priority-road--",
    "no_parking": "regulatory--no-parking--",
    "go_straight": "regulatory--go-straight",
    "no_u_turn": "regulatory--no-u-turn",
}

obj_counts = Counter()
img_sets = defaultdict(set)
matched_labels = defaultdict(Counter)

for img_name in os.listdir(IMG_DIR):
    key, ext = os.path.splitext(img_name)
    if ext.lower() != ".jpg":
        continue
    ann_path = os.path.join(ANN_DIR, key + ".json")
    if not os.path.exists(ann_path):
        continue
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in data["objects"]:
        for cls, prefix in RULES.items():
            if obj["label"].startswith(prefix):
                obj_counts[cls] += 1
                img_sets[cls].add(key)
                matched_labels[cls][obj["label"]] += 1

print(f"{'class':15s} {'objects':>8s} {'images':>8s}")
for cls in RULES:
    print(f"{cls:15s} {obj_counts[cls]:8d} {len(img_sets[cls]):8d}")

print()
for cls in ["speed_limit", "go_straight"]:
    print(f"Labels matched by '{cls}':")
    for label, n in matched_labels[cls].most_common():
        print(f"  {label:55s} {n}")