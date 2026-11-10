"""Prepare the hand-over files for teammates (Person 3: video pipeline, Person 4: evaluation).

Creates:
  models/traffic_signs_y11n_1280.pt       copy of best.pt with a clear name, to upload as a
                                          GitHub Release asset (models/ *.pt is ignored by git)
  results/traffic_signs/handover/
    splits/{train,val,test}.txt           image names of each split (seed 42 random split)
    gt_val.csv, gt_test.csv               ground-truth boxes in pixels, same columns as predictions
    sample_predictions/                   a few test images with predicted boxes drawn

Predictions themselves were already written by 06_evaluate.py:
  results/traffic_signs/train_<run>/predictions_val.csv and predictions_test.csv

Usage (repo root, venv active):
  python src/traffic_signs/08_export_handover.py --run y11n_1280
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

DATA_DIR = Path(r"C:\data\signs_yolo")
NAMES = ["stop", "yield", "speed_limit", "no_entry", "priority_road", "no_parking"]
IMGSZ, CONF = 1280, 0.25
N_SAMPLES = 6
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run name, e.g. y11n_1280")
    args = p.parse_args()

    out_dir = Path("results") / "traffic_signs" / "handover"
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_predictions").mkdir(exist_ok=True)

    # 1) Model file with a clear name for the GitHub Release
    best = Path("runs") / "traffic_signs" / args.run / "weights" / "best.pt"
    model_out = Path("models") / f"traffic_signs_{args.run}.pt"
    model_out.parent.mkdir(exist_ok=True)
    shutil.copy(best, model_out)
    print(f"Copied {best} -> {model_out} ({model_out.stat().st_size / 1e6:.1f} MB)")

    # 2) Split lists
    for split in ["train", "val", "test"]:
        names = sorted(p.name for p in (DATA_DIR / "images" / split).glob("*.jpg"))
        (out_dir / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")
        print(f"{split}: {len(names)} images")

    # 3) Ground truth in pixels (needs the image size)
    for split in ["val", "test"]:
        rows = []
        img_paths = sorted((DATA_DIR / "images" / split).glob("*.jpg"))
        for frame_index, img_path in enumerate(img_paths):
            with Image.open(img_path) as im:   # reads only the header, fast
                W, H = im.size
            label = DATA_DIR / "labels" / split / (img_path.stem + ".txt")
            for line in label.read_text().splitlines():
                c, xc, yc, w, h = map(float, line.split())
                rows.append([frame_index, img_path.name,
                             f"{(xc - w / 2) * W:.1f}", f"{(yc - h / 2) * H:.1f}",
                             f"{(xc + w / 2) * W:.1f}", f"{(yc + h / 2) * H:.1f}", NAMES[int(c)]])
        with open(out_dir / f"gt_{split}.csv", "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["frame_index", "image", "x1", "y1", "x2", "y2", "class_name"])
            w_.writerows(rows)
        print(f"gt_{split}.csv: {len(rows)} boxes")

    # 4) Sample predictions on random test images
    random.seed(SEED)
    test_imgs = sorted((DATA_DIR / "images" / "test").glob("*.jpg"))
    model = YOLO(str(model_out))
    for img_path in random.sample(test_imgs, N_SAMPLES):
        r = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
        im = r.plot(line_width=4, font_size=30)
        im = cv2.resize(im, (1600, int(im.shape[0] * 1600 / im.shape[1])))
        cv2.imwrite(str(out_dir / "sample_predictions" / img_path.name), im)
    print(f"Saved {N_SAMPLES} sample prediction images to {out_dir / 'sample_predictions'}")


if __name__ == "__main__":
    main()
