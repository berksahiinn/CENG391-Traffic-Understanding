import numpy as np, cv2, platform
from pathlib import Path
import ultralytics, torch
from ultralytics import YOLO
from inspect_labels import OUT

DST = Path("datasets/bosch_yolo"); W, H = 1280, 720
C = ["red", "yellow", "green", "off"]
MODEL, IMGSZ, CONF = "yolo11n.pt", 1280, 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
imgs = sorted((DST/"images/val").glob("*.png"))

def gt_boxes(p):
    out = []
    for ln in (DST/"labels/val"/f"{p.stem}.txt").read_text().splitlines():
        c, cx, cy, bw, bh = map(float, ln.split())
        out.append([int(c), (cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
    return np.array(out).reshape(-1, 5)

def iou(a, b):
    x1 = np.maximum(a[0], b[:,0]); y1 = np.maximum(a[1], b[:,1])
    x2 = np.minimum(a[2], b[:,2]); y2 = np.minimum(a[3], b[:,3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - inter + 1e-9)

def size_bin(w): return "<10px" if w < 10 else "10-20px" if w < 20 else ">=20px"

model = YOLO(MODEL)
res = {t: {"TP":0, "FP":0, "size":{}, "cls":{}} for t in [0.5, 0.3]}
speeds = []; vis_dir = OUT/"pretrained_samples"; vis_dir.mkdir(exist_ok=True)
for i, r in enumerate(model.predict(source=str(DST/"images/val"), batch=1, imgsz=IMGSZ, conf=CONF,
                                    classes=[9], device=DEVICE, stream=True, verbose=False)):
    speeds.append(r.speed["inference"])
    g = gt_boxes(Path(r.path)); conf = r.boxes.conf.cpu().numpy()
    pr = r.boxes.xyxy.cpu().numpy()[np.argsort(-conf)]
    for t, R in res.items():
        used = np.zeros(len(g), bool)
        for b in pr:
            if len(g):
                ious = iou(b, g[:,1:]); ious[used] = 0; k = ious.argmax()
                if ious[k] >= t: used[k] = True; R["TP"] += 1; continue
            R["FP"] += 1
        for k, gb in enumerate(g):
            for key, d in [(size_bin(gb[3]-gb[1]), R["size"]), (C[int(gb[0])], R["cls"])]:
                d.setdefault(key, [0, 0]); d[key][0] += int(used[k]); d[key][1] += 1
    if i % 120 == 0 and len(g):   # sample images: green = ground truth, model drawing = prediction
        im = r.plot(line_width=1, font_size=8)
        for gb in g: cv2.rectangle(im, (int(gb[1])-3, int(gb[2])-3), (int(gb[3])+3, int(gb[4])+3), (0,255,0), 1)
        cv2.imwrite(str(vis_dir/f"{Path(r.path).stem}.png"), im)

lines = [f"Model: {MODEL} (COCO, class 9 'traffic light') | ultralytics {ultralytics.__version__} | torch {torch.__version__}",
         f"Device: {DEVICE} ({platform.machine()}) | imgsz={IMGSZ} conf={CONF} | val images: {len(imgs)}",
         f"Mean inference time: {np.mean(speeds):.1f} ms/image (~{1000/np.mean(speeds):.1f} FPS, model only)"]
for t, R in res.items():
    n_gt = sum(v[1] for v in R["size"].values())
    lines.append(f"\n--- IoU >= {t} (class-agnostic) ---")
    lines.append(f"Precision {R['TP']/max(R['TP']+R['FP'],1):.3f} | Recall {R['TP']/max(n_gt,1):.3f} | TP {R['TP']} FP {R['FP']} GT {n_gt}")
    for name, d in [("Recall by size", R["size"]), ("Recall by state", R["cls"])]:
        lines.append(name + ":")
        for k in sorted(d): lines.append(f"   {k:8s} {d[k][0]/max(d[k][1],1):.3f}  ({d[k][0]}/{d[k][1]})")
txt = "\n".join(lines); print(txt)
(OUT/"baseline_pretrained_results.txt").write_text(txt)
