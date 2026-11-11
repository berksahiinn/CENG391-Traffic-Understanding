"""Self-checks for the video pipeline helpers. No model weights and no real video needed.

Run from the repo root:
  python src/video_pipeline/test_pipeline.py

Each line prints ok, TODO (function not written yet) or FAIL (written but wrong).
The exit code is 0 only when every test prints ok.
"""
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

from pipeline_utils import CSV_FIELDS, draw_rows, make_rows
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
