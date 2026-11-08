import yaml, csv
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/zumraaynurlu/Desktop/CENG391-Traffic-Understanding/bosch")
OUT = Path("results/traffic_lights"); OUT.mkdir(parents=True, exist_ok=True)

def map_state(label):
    l = label.lower()
    if l.startswith("red"): return "red"
    if l.startswith("yellow"): return "yellow"
    if l.startswith("green"): return "green"
    if l == "off": return "off"
    return None

if __name__ == "__main__":
    with open(ROOT / "train.yaml") as f:
        data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    print("Number of images:", len(data))
    orig = Counter(b["label"] for img in data for b in img["boxes"])
    with open(OUT / "label_mapping.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["original_label", "mapped_state", "count"])
        for lab, n in orig.most_common():
            w.writerow([lab, map_state(lab), n])
            print(f"{lab:22s} -> {map_state(lab)!s:7s} {n}")
    print("Total boxes:", sum(orig.values()))
