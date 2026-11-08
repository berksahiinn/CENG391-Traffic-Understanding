import cv2, random, yaml
from inspect_labels import ROOT, OUT, map_state

COL = {"red":(0,0,255),"yellow":(0,255,255),"green":(0,255,0),"off":(200,200,200)}
data = yaml.load(open(ROOT/"train.yaml"), Loader=getattr(yaml,"CSafeLoader",yaml.SafeLoader))
random.seed(0)
picked = []
for c in ["red","yellow","green","off"]:
    cand = [d for d in data if any(map_state(b["label"])==c for b in d["boxes"])]
    picked += random.sample(cand, min(3, len(cand)))
picked += random.sample([d for d in data if d["boxes"]], 4)

out = OUT/"annot_check"; out.mkdir(exist_ok=True)
for i, d in enumerate(picked):
    im = cv2.imread(str((ROOT/d["path"]).resolve())); raw = im.copy()
    for j, b in enumerate(d["boxes"]):
        s = map_state(b["label"])
        x1,y1,x2,y2 = map(int,[b["x_min"],b["y_min"],b["x_max"],b["y_max"]])
        pad=15; crop = raw[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
        if crop.size:
            big = cv2.resize(crop,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"{b['label']} occ={b['occluded']}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imwrite(str(out/f"{i:02d}_box{j}_{s}.png"), big)
        cv2.rectangle(im,(x1,y1),(x2,y2),COL[s],2)
        cv2.putText(im,s,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,COL[s],1)
    cv2.imwrite(str(out/f"{i:02d}_full.png"), im)
print("Saved:", len(picked), "images ->", out)
