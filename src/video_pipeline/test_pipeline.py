"""Self-checks for the video pipeline helpers. No model weights and no real video needed.

Run from the repo root:
  python src/video_pipeline/test_pipeline.py

Each line prints ok, TODO (function not written yet) or FAIL (written but wrong).
The exit code is 0 only when every test prints ok.
"""
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np

from frames_to_video import zip_frames_to_video
from pipeline_utils import CSV_FIELDS, draw_rows, keep_rows, make_rows
from video_info import read_video_info, save_sample_frames

NAMES = {0: "red", 1: "yellow", 2: "green", 3: "off"}


def _make_video(path, n=10, size=(64, 48), fps=5):
    """Write a tiny synthetic video: n grey frames, size is (width, height)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    writer.release()


# ---- Task 2: video_info.py ----

def test_read_video_info():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.mp4"
        _make_video(path)
        info = read_video_info(path)
        assert (info["width"], info["height"], info["frame_count"]) == (64, 48, 10)
        assert round(info["fps"]) == 5
        assert round(info["duration_s"]) == 2


def test_read_video_info_missing_file():
    try:
        read_video_info("does_not_exist.mp4")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing video")


def test_save_sample_frames():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.mp4"
        _make_video(path)
        paths = save_sample_frames(path, Path(d) / "frames", n=3)
        assert len(paths) == 3 and all(Path(p).exists() for p in paths)
        assert cv2.imread(str(paths[0])).shape == (48, 64, 3)   # (height, width, channels)


# ---- Task 3: pipeline_utils.py ----

def test_make_rows_with_track_ids():
    rows = make_rows(7, "light", NAMES, [[10.26, 20.0, 30.0, 60.0]], [2], [0.87654], [5])
    assert rows == [{"frame_index": 7, "x1": 10.3, "y1": 20.0, "x2": 30.0, "y2": 60.0,
                     "class_name": "green", "confidence": 0.8765, "track_id": 5,
                     "source": "light"}]
    assert list(rows[0]) == CSV_FIELDS   # same keys, same order as the contract


def test_make_rows_without_tracking():
    rows = make_rows(0, "sign", {0: "stop"}, [[1, 2, 3, 4], [5, 6, 7, 8]], [0, 0], [0.5, 0.6])
    assert [r["track_id"] for r in rows] == [None, None]
    assert [r["source"] for r in rows] == ["sign", "sign"]


def test_make_rows_no_detections():
    assert make_rows(3, "light", NAMES, [], [], [], None) == []


def test_draw_rows_draws_box_in_class_color():
    frame = np.zeros((100, 100, 3), np.uint8)
    draw_rows(frame, make_rows(0, "light", NAMES, [[10, 40, 60, 90]], [0], [0.9], [3]))
    assert frame[65, 10].tolist() == [0, 0, 255]   # left edge of the box is red in BGR


def test_draw_rows_empty_keeps_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    draw_rows(frame, [])
    assert not frame.any()


def test_keep_rows_track_mode_keeps_only_confirmed_tracks():
    rows = make_rows(0, "sign", {0: "stop"}, [[0, 0, 1, 1]] * 3, [0, 0, 0], [0.9, 0.9, 0.2], [4, None, 5])
    assert [r["track_id"] for r in keep_rows(rows, track=True)] == [4]   # no id -> dropped, 0.2 -> dropped


def test_keep_rows_detect_mode_keeps_rows_without_ids():
    rows = make_rows(0, "light", NAMES, [[0, 0, 1, 1]] * 2, [2, 2], [0.25, 0.2499])
    assert [r["confidence"] for r in keep_rows(rows, track=False)] == [0.25]


def test_zip_frames_to_video_numeric_order():
    with tempfile.TemporaryDirectory() as d:
        zip_path, out = Path(d) / "frames.zip", Path(d) / "out.mp4"
        with zipfile.ZipFile(zip_path, "w") as z:
            for n in [10, 9, 11]:   # alphabetical order would be 10, 11, 9
                z.writestr(f"set/rgb/test/{n}.png", cv2.imencode(".png", np.full((48, 64, 3), n * 10, np.uint8))[1].tobytes())
            z.writestr("set/test.yaml", "")   # non-PNG entries are skipped
        assert zip_frames_to_video(zip_path, out, fps=5) == 3
        cap = cv2.VideoCapture(str(out)); means = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            means.append(f.mean())
        cap.release()
        assert len(means) == 3 and means[0] < means[1] < means[2]   # 9, 10, 11


def test_zip_frames_to_video_removes_output_on_error():
    with tempfile.TemporaryDirectory() as d:
        zip_path, out = Path(d) / "frames.zip", Path(d) / "out.mp4"
        pngs = [cv2.imencode(".png", np.full((48, 64, 3), v, np.uint8))[1].tobytes() for v in (10, 200, 30)]
        with zipfile.ZipFile(zip_path, "w") as z:
            for n, png in zip([1, 2, 3], pngs):
                z.writestr(f"{n}.png", png)
        raw = bytearray(zip_path.read_bytes())
        raw[raw.find(pngs[1]) + 30] ^= 0xFF   # frame 2 fails its CRC check when read
        zip_path.write_bytes(bytes(raw))
        # Run the script as its own process: a leftover ffmpeg finishes the file only after Python exits.
        script = Path(__file__).parent / "frames_to_video.py"
        result = subprocess.run([sys.executable, str(script), "--zip", str(zip_path), "--out", str(out)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "BadZipFile" in result.stderr
        for _ in range(20):   # a leftover ffmpeg would write a truncated file within this time
            assert not out.exists(), "a truncated video was left behind"
            time.sleep(0.1)


if __name__ == "__main__":
    failed = 0
    for test_name, test in list(globals().items()):
        if not test_name.startswith("test_"):
            continue
        try:
            test()
            print("ok   ", test_name)
        except NotImplementedError as e:
            failed += 1
            print("TODO ", test_name, "->", e)
        except Exception:
            failed += 1
            print("FAIL ", test_name)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
