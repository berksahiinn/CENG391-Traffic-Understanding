"""Speed of the light + sign detectors on one machine, with one fixed method.

    python src/evaluation/speed.py --video datasets/videos/sample_drive.mp4 --device mps --name m3pro_mps

Method (same for every run, so runs are comparable):
- The first --warmup frames (default 20) are run but NOT measured (model loading, GPU start-up).
- The next --frames frames (default 300) are measured, one frame at a time (batch 1).
- For every measured frame we time, with time.perf_counter():
    decode_ms  reading the frame from the video (OpenCV)
    light_ms   traffic light model: preprocess + inference + postprocess (wall clock)
    sign_ms    traffic sign model:  preprocess + inference + postprocess (wall clock)
  model-only time = light_ms + sign_ms
  end-to-end time = decode_ms + light_ms + sign_ms   (no drawing, no video writing)
- With --track, both models run ByteTrack like the video pipeline (conf 0.1, persist=True).

Writes results/metrics/speed/<name>/per_frame.csv and summary.txt (with hardware and versions).
"""
import argparse
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

LIGHT_W = "models/traffic_lights_y11n_1280_best.pt"
SIGN_W = "models/traffic_signs_y11n_1280.pt"


def hardware():
    chip = platform.processor()
    if platform.system() == "Darwin":                      # macOS: real chip name, e.g. "Apple M3 Pro"
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    elif torch.cuda.is_available():
        chip = torch.cuda.get_device_name(0)
    return chip


def sync(device):
    """Wait until the GPU has really finished, otherwise the timer stops too early."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def run_model(model, frame, a):
    if a.track:
        return model.track(frame, imgsz=a.imgsz, conf=0.1, persist=True, tracker="bytetrack.yaml",
                           device=a.device, verbose=False)
    return model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)


def timed(fn, device):
    sync(device)
    t = time.perf_counter()
    out = fn()
    sync(device)
    return out, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True, help="output folder under results/metrics/speed/")
    ap.add_argument("--device", default="mps", help="mps, cpu, or cuda:0")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--track", action="store_true")
    a = ap.parse_args()

    light, sign = YOLO(LIGHT_W), YOLO(SIGN_W)
    cap = cv2.VideoCapture(a.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    if n_video < a.warmup + a.frames:
        raise SystemExit(f"video has {n_video} frames, need {a.warmup + a.frames}")

    rows = []
    for i in range(a.warmup + a.frames):
        (ok, frame), decode_ms = timed(cap.read, "cpu")
        if not ok:
            raise SystemExit(f"could not read frame {i}")
        rl, light_ms = timed(lambda: run_model(light, frame, a), a.device)
        rs, sign_ms = timed(lambda: run_model(sign, frame, a), a.device)
        if i >= a.warmup:                                   # warm-up frames are not kept
            rows.append(dict(frame=i, decode_ms=decode_ms, light_ms=light_ms, sign_ms=sign_ms,
                             light_infer_ms=rl[0].speed["inference"], sign_infer_ms=rs[0].speed["inference"]))
    cap.release()

    df = pd.DataFrame(rows)
    df["model_only_ms"] = df["light_ms"] + df["sign_ms"]
    df["end_to_end_ms"] = df["decode_ms"] + df["model_only_ms"]

    out = Path("results/metrics/speed") / a.name
    out.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out / "per_frame.csv", index=False)

    lines = [
        f"video: {a.video}  ({n_video} frames, {size})",
        f"hardware: {hardware()}  device: {a.device}  os: {platform.platform()}",
        f"python {platform.python_version()}  torch {torch.__version__}  ultralytics {ultralytics.__version__}"
        f"  opencv {cv2.__version__}",
        f"weights: {LIGHT_W}, {SIGN_W}",
        f"imgsz {a.imgsz}  conf {a.conf}  track {a.track}  batch 1",
        f"warm-up frames (not measured): {a.warmup}  measured frames: {len(df)}",
        "",
        f"{'per frame (ms)':<22}{'mean':>8}{'median':>8}{'p95':>8}",
    ]
    for c in ["decode_ms", "light_ms", "sign_ms", "model_only_ms", "end_to_end_ms",
              "light_infer_ms", "sign_infer_ms"]:
        s = df[c]
        lines.append(f"{c:<22}{s.mean():>8.2f}{s.median():>8.2f}{np.percentile(s, 95):>8.2f}")
    lines += ["",
              f"model-only FPS (both models): {1000 / df['model_only_ms'].mean():.1f}",
              f"end-to-end FPS (decode + both models): {1000 / df['end_to_end_ms'].mean():.1f}",
              "light_infer_ms / sign_infer_ms = Ultralytics' own 'inference' timing (model forward only)."]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
