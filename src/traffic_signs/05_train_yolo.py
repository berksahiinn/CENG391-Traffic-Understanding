"""Fine-tune YOLO11n (COCO-pretrained) on our 6 traffic sign classes.

Usage (from repo root, venv active):
  Smoke test (fast check that everything works, results are NOT meaningful):
    python src/traffic_signs/05_train_yolo.py --name smoke --epochs 2 --fraction 0.1
  Full run:
    python src/traffic_signs/05_train_yolo.py --name y11n_1280 --epochs 100

Ultralytics writes weights and plots to runs/traffic_signs/<name>/ (ignored by git).
This script also writes results/traffic_signs/train_<name>/run_info.txt with the
exact command, hyperparameters, GPU and library versions (for the report).
"""
import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

DATA_YAML = r"C:\data\signs_yolo\data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, e.g. smoke or y11n_1280")
    p.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO-pretrained)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8, help="lower to 4 if CUDA out of memory")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of train images to use")
    p.add_argument("--patience", type=int, default=20, help="stop early after N epochs without val improvement")
    p.add_argument("--workers", type=int, default=4, help="data loading processes")
    args = p.parse_args()

    hyp = dict(data=DATA_YAML, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
               fraction=args.fraction, patience=args.patience, workers=args.workers,
               seed=42, deterministic=True, device=0 if torch.cuda.is_available() else "cpu",
               # Absolute path: a relative one gets nested under runs/detect/ by ultralytics
               project=str((Path("runs") / "traffic_signs").resolve()), name=args.name, exist_ok=True)

    # Save run information before training, so it exists even if training crashes
    info_dir = Path("results") / "traffic_signs" / f"train_{args.name}"
    info_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    info = [f"Date: {datetime.now():%Y-%m-%d %H:%M}",
            f"Command: python {' '.join(sys.argv)}",
            f"Start weights: {args.model}",
            f"GPU: {gpu} | Python {platform.python_version()} | torch {torch.__version__}"
            f" | ultralytics {ultralytics.__version__}",
            "Hyperparameters (others are ultralytics defaults):"]
    info += [f"  {k}: {v}" for k, v in hyp.items()]
    (info_dir / "run_info.txt").write_text("\n".join(info) + "\n")
    print("\n".join(info))

    model = YOLO(args.model)
    model.train(**hyp)


# The guard is required on Windows: data loader workers re-import this file.
if __name__ == "__main__":
    main()
