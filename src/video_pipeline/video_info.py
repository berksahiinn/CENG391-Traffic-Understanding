"""Read a driving video with OpenCV and report what is in it.

Usage (from the repo root):
  python src/video_pipeline/video_info.py --video datasets/videos/sample_drive.mp4

Outputs (results/video_pipeline/):
  video_info.txt   FPS, frame count, resolution, duration
  sample_frames/   a few evenly spaced frames as jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("results") / "video_pipeline"
N_SAMPLES = 6


def read_video_info(video_path):
    """Return {"fps", "frame_count", "width", "height", "duration_s"} for a video.

    Raises FileNotFoundError when the file is missing or OpenCV cannot open it.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():  # OpenCV does not raise on a bad path, it returns an empty capture
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "frame_count": frame_count,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": frame_count / fps if fps else 0.0}
    cap.release()
    return info


def save_sample_frames(video_path, out_dir, n=N_SAMPLES):
    """Save n evenly spaced frames as out_dir/frame_000123.jpg. Return the list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, read_video_info(video_path)["frame_count"] - 1, n, dtype=int)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            path = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
    cap.release()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="path to the input video")
    args = parser.parse_args()

    info = read_video_info(args.video)
    lines = [f"video: {args.video}",
             f"resolution: {info['width']}x{info['height']}",
             f"fps: {info['fps']:.2f}",
             f"frame_count: {info['frame_count']}",
             f"duration_s: {info['duration_s']:.2f}"]
    print("\n".join(lines))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "video_info.txt").write_text("\n".join(lines) + "\n")
    paths = save_sample_frames(args.video, OUT_DIR / "sample_frames")
    print(f"saved {len(paths)} sample frames to {OUT_DIR / 'sample_frames'}")


if __name__ == "__main__":
    main()
