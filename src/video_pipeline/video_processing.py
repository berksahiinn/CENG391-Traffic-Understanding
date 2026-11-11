"""Video -> frames -> detection (lights + signs) -> tracking -> annotated video + CSV.

Usage (from the repo root):
  # COCO-pretrained model, detection only
  python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --baseline --name baseline_detect
  # same model with ByteTrack ids
  python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --baseline --track --name baseline_track
  # both fine-tuned models with ByteTrack ids
  python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --track --name finetuned_track

Outputs (results/video_pipeline/<name>/):
  annotated.mp4    input video with boxes, labels, confidences and track ids (mp4v, not in git)
  detections.csv   one row per box, common detection format (pipeline_utils.CSV_FIELDS)
  run_info.txt     settings and speed
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from pipeline_utils import CONF, CSV_FIELDS, draw_rows, keep_rows, make_rows

OUT_ROOT = Path("results") / "video_pipeline"
IMGSZ = 1280                    # agreed with Person 1 and 2; a smaller imgsz loses the tiny objects
TRACK_CONF = 0.1                # ByteTrack's second matching pass uses boxes between 0.1 and 0.25
TRACKER = "bytetrack.yaml"      # ships with Ultralytics

# source name -> weights file. The source name ends up in the CSV "source" column.
MODELS = {
    "light": "models/traffic_lights_y11n_1280_best.pt",   # Person 1, Google Drive
    "sign": "models/traffic_signs_y11n_1280.pt",          # Person 2, GitHub Release signs-y11n-1280-v1
}
BASELINE = {"coco": "yolo11n.pt"}   # downloaded automatically by Ultralytics
BASELINE_CLASSES = [9, 11]          # COCO ids: 9 = traffic light, 11 = stop sign

DEVICE = "mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu"


def load_models(baseline):
    """Return {source name: YOLO model}. BASELINE when baseline is True, else MODELS."""
    paths = BASELINE if baseline else MODELS
    missing = [p for p in paths.values() if p != "yolo11n.pt" and not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing weights: {', '.join(missing)}. "
                                "See src/video_pipeline/README.md, section 'How to run'.")
    return {source: YOLO(path) for source, path in paths.items()}


def detect(model, frame, track, classes):
    """Run one model on one frame and return its Ultralytics Results object."""
    if not track:
        return model.predict(frame, imgsz=IMGSZ, conf=CONF, classes=classes,
                             device=DEVICE, verbose=False)[0]
    # persist=True keeps this model's tracker alive between calls; without it ids restart every frame.
    return model.track(frame, imgsz=IMGSZ, conf=TRACK_CONF, classes=classes, device=DEVICE,
                       persist=True, tracker=TRACKER, verbose=False)[0]


def run(video, name, baseline, track):
    """Process the whole video and write the three output files."""
    models = load_models(baseline)  # fails before any output is created
    classes = BASELINE_CLASSES if baseline else None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # some files report 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # The writer size must equal the frame size, otherwise OpenCV writes a broken file silently.
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open annotated.mp4 (no mp4v codec in this OpenCV build?)")
    infer_ms = {source: [] for source in models}
    frame_index, n_rows = 0, 0
    start = time.perf_counter()
    with open(out_dir / "detections.csv", "w", newline="") as f:
        csv_writer = csv.DictWriter(f, CSV_FIELDS)
        csv_writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows = []
            for source, model in models.items():
                r = detect(model, frame, track, classes)
                infer_ms[source].append(r.speed["inference"])
                ids = r.boxes.id.int().tolist() if r.boxes.id is not None else None
                rows += make_rows(frame_index, source, model.names, r.boxes.xyxy.tolist(),
                                  r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids)
            rows = keep_rows(rows, track)  # tracking ran at TRACK_CONF
            draw_rows(frame, rows)
            writer.write(frame)
            csv_writer.writerows(rows)
            n_rows += len(rows)
            frame_index += 1
    seconds = time.perf_counter() - start
    cap.release()
    writer.release()  # without release() the mp4 is left unplayable

    info = [f"video: {video}",
            f"weights: {', '.join(f'{s}={p}' for s, p in (BASELINE if baseline else MODELS).items())}",
            f"imgsz: {IMGSZ}  conf: {CONF}  track: {track}"
            + (f"  (tracker {TRACKER}, tracked at conf {TRACK_CONF}, output filtered at {CONF})" if track else ""),
            f"device: {DEVICE}",
            f"frames: {frame_index}  rows: {n_rows}",
            f"total_s: {seconds:.1f}  end_to_end_fps: {frame_index / seconds:.1f}"
            "  (decode + all models + tracking + drawing + writing)",
            *[f"mean_inference_ms[{s}]: {np.mean(v):.1f}" for s, v in infer_ms.items() if v],
            f"ultralytics: {ultralytics.__version__}  torch: {torch.__version__}  opencv: {cv2.__version__}"]
    (out_dir / "run_info.txt").write_text("\n".join(info) + "\n")
    print("\n".join(info))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="path to the input video")
    parser.add_argument("--name", required=True, help="output folder under results/video_pipeline/")
    parser.add_argument("--baseline", action="store_true", help="use the COCO model instead of the fine-tuned ones")
    parser.add_argument("--track", action="store_true", help="turn ByteTrack on")
    args = parser.parse_args()
    run(args.video, args.name, args.baseline, args.track)


if __name__ == "__main__":
    main()
