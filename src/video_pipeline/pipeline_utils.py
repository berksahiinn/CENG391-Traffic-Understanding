"""Small helpers for the video pipeline. No model code here, so they are easy to test.

The common detection format (see README.md in this folder): one dict per
detected box, with exactly the keys in CSV_FIELDS, in that order.

Example row:
  {"frame_index": 7, "x1": 10.3, "y1": 20.0, "x2": 30.0, "y2": 60.0,
   "class_name": "green", "confidence": 0.8765, "track_id": 5, "source": "light"}
"""
import cv2

# The team contract. x1, y1, x2, y2 is bbox_xyxy in pixels of the original frame.
CSV_FIELDS = ["frame_index", "x1", "y1", "x2", "y2",
              "class_name", "confidence", "track_id", "source"]

# Box colours in BGR (OpenCV order, not RGB). Sign classes use DEFAULT_COLOR.
COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255),
          "green": (0, 255, 0), "off": (128, 128, 128)}
DEFAULT_COLOR = (255, 128, 0)


def make_rows(frame_index, source, names, xyxy, cls, conf, ids=None):
    """Turn the boxes of ONE frame from ONE model into common-format rows."""
    raise NotImplementedError("make_rows")


def draw_rows(frame, rows):
    """Draw every row on the frame in place."""
    raise NotImplementedError("draw_rows")
