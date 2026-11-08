import yaml, csv, statistics as st
from collections import Counter
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_labels import ROOT, OUT, map_state

data = yaml.load(open(ROOT/"train.yaml"), Loader=getattr(yaml,"CSafeLoader",yaml.SafeLoader))
sizes, cls, occ = Counter(), Counter(), Counter()
widths, heights, n_empty, missing = [], [], 0, 0
bags = set()
for img in data:
    p = (ROOT / img["path"]).resolve()
    if not p.exists(): missing += 1; continue
    bags.add(p.parent.name)
    sizes[Image.open(p).size] += 1
    if not img["boxes"]: n_empty += 1
    for b in img["boxes"]:
        s = map_state(b["label"]); cls[s] += 1; occ[s] += int(b["occluded"])
        widths.append(abs(b["x_max"]-b["x_min"])); heights.append(abs(b["y_max"]-b["y_min"]))

print("Images:", len(data), "| missing files:", missing, "| images without lights:", n_empty)
print("Number of sequences (bags):", len(bags))
print("Resolutions:", dict(sizes))
tot = sum(cls.values())
with open(OUT/"class_counts.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["class","count","occluded","percent"])
    for c,n in cls.most_common():
        w.writerow([c,n,occ[c],f"{100*n/tot:.1f}"]); print(f"{c:7s} {n:6d}  occluded: {occ[c]}")
print(f"Width: median {st.median(widths):.1f}px, mean {st.mean(widths):.1f}, min {min(widths):.2f}, max {max(widths):.1f}")
print(f"Height: median {st.median(heights):.1f}px")
for t in [5,10,20]:
    print(f"Share of boxes narrower than {t}px: {100*sum(x<t for x in widths)/len(widths):.1f}%")
plt.hist(widths, bins=60, range=(0,60))
plt.xlabel("Bounding-box width (px)"); plt.ylabel("Count"); plt.title("Bosch train set: traffic light width distribution")
plt.savefig(OUT/"box_width_hist.png", dpi=150, bbox_inches="tight")
print("Saved:", OUT/"box_width_hist.png")
