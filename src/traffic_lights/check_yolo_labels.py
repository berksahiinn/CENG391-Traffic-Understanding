import cv2, random, yaml
from pathlib import Path
from inspect_labels import ROOT, OUT, map_state

DST = Path("datasets/bosch_yolo"); W, H = 1280, 720
data = yaml.load(open(ROOT/"train.yaml"), Loader=getattr(yaml,"CSafeLoader",yaml.SafeLoader))
orig = {}
for d in data:
    p = Path(d["path"]); orig[f"{p.parent.name}_{p.stem}"] = d["boxes"]

def yolo_to_px(line):
    c, cx, cy, bw, bh = line.split(); cx, cy, bw, bh = float(cx)*W, float(cy)*H, float(bw)*W, float(bh)*H
    return int(c), [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]

# 1) Numerical check: all labels
max_err, n_checked = 0.0, 0
for split in ["train", "val"]:
    for txt in (DST/"labels"/split).glob("*.txt"):
        yolo = [yolo_to_px(l) for l in txt.read_text().splitlines()]
        ref = []
        for b in orig[txt.stem]:
            x1, x2 = sorted([b["x_min"], b["x_max"]]); y1, y2 = sorted([b["y_min"], b["y_max"]])
            x1, x2, y1, y2 = max(0,x1), min(W,x2), max(0,y1), min(H,y2)
            if x2-x1 >= 1 and y2-y1 >= 1: ref.append([x1, y1, x2, y2])
        assert len(ref) == len(yolo), f"Box count mismatch: {txt.name}"
        for (_, a), b in zip(yolo, ref):
            max_err = max(max_err, max(abs(u-v) for u, v in zip(a, b))); n_checked += 1
print(f"Boxes checked: {n_checked} | max coordinate deviation: {max_err:.4f} px")

# 2) Visual check: 5 val images (green = original, magenta = reconstructed from YOLO)
C = ["red", "yellow", "green", "off"]
cands = [p for p in sorted((DST/"images/val").glob("*.png")) if (DST/"labels/val"/f"{p.stem}.txt").read_text()]
for p in random.Random(1).sample(cands, 5):
    im = cv2.imread(str(p))
    for b in orig[p.stem]:
        cv2.rectangle(im, (int(b["x_min"])-3, int(b["y_min"])-3), (int(b["x_max"])+3, int(b["y_max"])+3), (0,255,0), 1)
    for l in (DST/"labels/val"/f"{p.stem}.txt").read_text().splitlines():
        c, (x1, y1, x2, y2) = yolo_to_px(l)
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,255), 1)
        cv2.putText(im, C[c], (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1)
    cv2.imwrite(str(OUT/f"yolo_check_{p.stem}.png"), im)
print("Images:", OUT, "(yolo_check_*.png)")
