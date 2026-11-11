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
CONF = 0.25   # team confidence threshold, agreed with Person 1 and 2

# Box colours in BGR (OpenCV order, not RGB). Sign classes use DEFAULT_COLOR.
COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255),
          "green": (0, 255, 0), "off": (128, 128, 128)}
DEFAULT_COLOR = (255, 128, 0)


def make_rows(frame_index, source, names, xyxy, cls, conf, ids=None):
    """Turn the boxes of ONE frame from ONE model into common-format rows.

    frame_index  int, 0 for the first frame
    source       "light", "sign" or "coco" (which model produced the boxes)
    names        dict class id -> class name, i.e. model.names
    xyxy         list of [x1, y1, x2, y2], one per box
    cls          list of class ids (ints), one per box
    conf         list of confidences (floats), one per box
    ids          list of track ids (ints), or None when there are no track ids

    Returns a list of dicts. An empty frame returns [].
    """
    if ids is None:
        ids = [None] * len(xyxy)
    return [{"frame_index": frame_index,
             "x1": round(b[0], 1), "y1": round(b[1], 1),
             "x2": round(b[2], 1), "y2": round(b[3], 1),
             "class_name": names[c], "confidence": round(p, 4),
             "track_id": t, "source": source}
            for b, c, p, t in zip(xyxy, cls, conf, ids)]


def draw_rows(frame, rows):
    """Draw every row on the frame in place.

    Each box gets a 2 px rectangle in its class colour and a label above it:
      "green #5 0.88"   when the row has a track_id
      "green 0.88"      when track_id is None
    """
    for r in rows:
        colour = COLORS.get(r["class_name"], DEFAULT_COLOR)
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        tid = f" #{r['track_id']}" if r["track_id"] is not None else ""
        # Small font on purpose: the objects are tiny and a big label would hide them.
        cv2.putText(frame, f"{r['class_name']}{tid} {r['confidence']:.2f}",
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def keep_rows(rows, track, conf=CONF):
    """Keep rows with confidence >= conf. In track mode, also drop rows without a track id.

    When a tracker has no confirmed track in a frame, Ultralytics returns the raw boxes
    without ids; when it has one, it returns only tracked boxes. Keeping only confirmed
    tracks makes track-mode output the same in both cases.
    """
    return [r for r in rows if r["confidence"] >= conf and (not track or r["track_id"] is not None)]
