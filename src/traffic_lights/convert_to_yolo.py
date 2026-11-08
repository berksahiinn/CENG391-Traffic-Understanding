import os, shutil, yaml
from pathlib import Path
from collections import Counter
from inspect_labels import ROOT, OUT, map_state

DST = Path("datasets/bosch_yolo").resolve()
CLASSES = ["red", "yellow", "green", "off"]; CID = {c: i for i, c in enumerate(CLASSES)}
W, H = 1280, 720
VAL_BAGS = {
    "2017-02-03-11-44-56_los_altos_mountain_view_traffic_lights_bag",
    "2015-05-29-15-29-39_arastradero_traffic_light_loop_bag",
}

data = yaml.load(open(ROOT/"train.yaml"), Loader=getattr(yaml,"CSafeLoader",yaml.SafeLoader))
box_stats = {"train": Counter(), "val": Counter()}; img_stats = Counter(); skipped = 0
for split in ["train", "val"]:
    (DST/"images"/split).mkdir(parents=True, exist_ok=True)
    (DST/"labels"/split).mkdir(parents=True, exist_ok=True)

for d in data:
    src = (ROOT/d["path"]).resolve()
    bag = src.parent.name
    split = "val" if bag in VAL_BAGS else "train"
    name = f"{bag}_{src.stem}"            # different sequences may share file names
    dst_img = DST/"images"/split/f"{name}.png"
    if not dst_img.exists():
        try: os.link(src, dst_img)         # hard link: uses no extra disk space
        except OSError: shutil.copy(src, dst_img)
    lines = []
    for b in d["boxes"]:
        s = map_state(b["label"])
        x1, x2 = sorted([b["x_min"], b["x_max"]]); y1, y2 = sorted([b["y_min"], b["y_max"]])
        x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2)   # clip to image borders
        if s is None or x2 - x1 < 1 or y2 - y1 < 1:
            skipped += 1; continue
        lines.append(f"{CID[s]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        box_stats[split][s] += 1
    (DST/"labels"/split/f"{name}.txt").write_text("\n".join(lines))   # empty file = background image
    img_stats[split] += 1

yaml.safe_dump({"path": str(DST), "train": "images/train", "val": "images/val",
                "names": {i: c for i, c in enumerate(CLASSES)}},
               open(DST/"data.yaml", "w"), sort_keys=False)
(OUT/"val_bags.txt").write_text("\n".join(sorted(VAL_BAGS)))
print("Skipped boxes (invalid/too small):", skipped)
for s in ["train", "val"]:
    print(f"{s:5s} images: {img_stats[s]:5d} | boxes: {dict(box_stats[s])}")
print("data.yaml ->", DST/"data.yaml")
