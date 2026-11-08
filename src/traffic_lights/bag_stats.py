import yaml
from pathlib import Path
from collections import Counter, defaultdict
from inspect_labels import ROOT, OUT, map_state

data = yaml.load(open(ROOT/"train.yaml"), Loader=getattr(yaml,"CSafeLoader",yaml.SafeLoader))
n_img, cls = Counter(), defaultdict(Counter)
for d in data:
    bag = Path(d["path"]).parent.name
    n_img[bag] += 1
    for b in d["boxes"]:
        cls[bag][map_state(b["label"])] += 1

print(f"{'bag':45s} {'img':>5s} {'red':>5s} {'yel':>5s} {'grn':>5s} {'off':>5s}")
lines = ["bag,images,red,yellow,green,off"]
for bag in sorted(n_img):
    c = cls[bag]
    print(f"{bag:45s} {n_img[bag]:5d} {c['red']:5d} {c['yellow']:5d} {c['green']:5d} {c['off']:5d}")
    lines.append(f"{bag},{n_img[bag]},{c['red']},{c['yellow']},{c['green']},{c['off']}")
(OUT/"bag_class_counts.csv").write_text("\n".join(lines))
