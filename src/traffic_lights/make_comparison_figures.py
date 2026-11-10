import cv2, numpy as np
from pathlib import Path

BASE = Path("results/traffic_lights/pretrained_samples")
FT = Path("results/traffic_lights/finetuned/samples")
LBL = Path("datasets/bosch_yolo/labels/val")
OUT = Path("results/traffic_lights/finetuned/compare"); OUT.mkdir(parents=True, exist_ok=True)
W, H, PAD, SCALE = 1280, 720, 80, 2

def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

for ft_path in sorted(FT.glob("*.jpg")):
    stem = ft_path.stem
    base_path = BASE/f"{stem}.png"
    if not base_path.exists(): print("baseline missing:", stem); continue
    b, f = cv2.imread(str(base_path)), cv2.imread(str(ft_path))
    boxes = []
    for ln in (LBL/f"{stem}.txt").read_text().splitlines():
        _, cx, cy, bw, bh = map(float, ln.split())
        boxes.append([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
    boxes = np.array(boxes)
    x1, y1 = int(max(0, boxes[:,0].min()-PAD)), int(max(0, boxes[:,1].min()-PAD))
    x2, y2 = int(min(W, boxes[:,2].max()+PAD)), int(min(H, boxes[:,3].max()+PAD))
    crops = [cv2.resize(im[y1:y2, x1:x2], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST) for im in (b, f)]
    top = label(crops[0], "COCO YOLO11n (baseline, no state)")
    bottom = label(crops[1], "Fine-tuned YOLO11n (4 states)")
    sep = np.full((8, top.shape[1], 3), 255, np.uint8)
    cv2.imwrite(str(OUT/f"{stem}.jpg"), np.vstack([top, sep, bottom]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved:", stem)
