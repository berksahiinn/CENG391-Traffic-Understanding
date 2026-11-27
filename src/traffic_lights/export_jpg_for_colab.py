import cv2, shutil, yaml
from pathlib import Path

SRC = Path("datasets/bosch_yolo"); DST = Path("datasets/bosch_yolo_jpg")
QUALITY = 95
n = 0
for split in ["train", "val"]:
    (DST/"images"/split).mkdir(parents=True, exist_ok=True)
    (DST/"labels"/split).mkdir(parents=True, exist_ok=True)
    for p in sorted((SRC/"images"/split).glob("*.png")):
        out = DST/"images"/split/f"{p.stem}.jpg"
        if not out.exists():
            cv2.imwrite(str(out), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        shutil.copy(SRC/"labels"/split/f"{p.stem}.txt", DST/"labels"/split/f"{p.stem}.txt")
        n += 1
        if n % 500 == 0: print(n, "images done")
yaml.safe_dump({"path": "/content/bosch_yolo_jpg", "train": "images/train", "val": "images/val",
                "names": {0: "red", 1: "yellow", 2: "green", 3: "off"}},
               open(DST/"data.yaml", "w"), sort_keys=False)
print("Done:", n, "images ->", DST)
