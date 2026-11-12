# Video processing, tracking and integration

Owner: Person 3. Reads a recorded driving video, runs the traffic light and traffic sign detectors on every frame, tracks the boxes with ByteTrack, and writes an annotated video and a CSV in the common detection format.

## Architecture

![Video pipeline architecture](../../results/video_pipeline/architecture.png)

```mermaid
flowchart LR
    V["Recorded driving video (.mp4)"] --> R["Frame reader<br/>cv2.VideoCapture"]
    R -->|"frame + frame_index"| L["Traffic light model<br/>YOLO11n, imgsz 1280"]
    R -->|"frame + frame_index"| S["Traffic sign model<br/>YOLO11n, imgsz 1280"]
    L --> TL["ByteTrack (lights)"]
    S --> TS["ByteTrack (signs)"]
    TL --> M["Merge rows, keep conf ≥ 0.25<br/>common detection format"]
    TS --> M
    M --> D["Draw box, class, confidence, track id"]
    M --> C["detections.csv"]
    D --> W["annotated.mp4"]
    C --> E["Person 4: evaluation"]
```

Each frame goes through both models one after the other. Each model keeps its own ByteTrack tracker. The rows of both models are concatenated, because the two models share no classes.

## Common detection format

One row per detected box in `detections.csv`:

| Field | Type | Meaning | Example |
|---|---|---|---|
| `frame_index` | int | 0 for the first frame of the video | `24` |
| `x1, y1, x2, y2` | float | `bbox_xyxy`: top-left and bottom-right corner, in pixels of the original frame | `915.0, 291.6, 957.7, 320.3` |
| `class_name` | str | From `model.names`. Lights: red, yellow, green, off. Signs: stop, yield, speed_limit, no_entry, priority_road, no_parking | `traffic light` |
| `confidence` | float | 0 to 1 | `0.2668` |
| `track_id` | int or empty | ByteTrack id. Empty only in detect-only runs; tracking runs keep confirmed tracks only | (empty) |
| `source` | str | Which model produced the box: `light`, `sign` or `coco`. Track ids are counted per model, so `(source, track_id)` identifies one object | `coco` |

The example is a real row from `results/video_pipeline/baseline_detect/detections.csv` (COCO model, detection only).

Fixed settings: `imgsz=1280`, `conf=0.25`.

## Sample video

| | |
|---|---|
| File | `datasets/videos/sample_drive.mp4` (not in git) |
| Source | [City Driving 4K- Kraków Poland 2024](https://commons.wikimedia.org/wiki/File:City_Driving_4K-_Krak%C3%B3w_Poland_2024.webm), Wikimedia Commons, by Relaxing Roads 4K |
| Licence | CC BY 3.0 (reuse allowed with attribution) |
| Part used | 58–78 s of the 1080p transcode, scaled to 1280×720, 30 fps, H.264, no audio |
| Info | 1280×720, 30 fps, 600 frames, 20 s (`results/video_pipeline/video_info.txt`) |

To rebuild it (the first 100 MB of the transcode cover the first two minutes):

```bash
mkdir -p datasets/videos/raw
curl -r 0-99999999 -o datasets/videos/raw/krakow_1080p_head.webm "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/98/City_Driving_4K-_Krak%C3%B3w_Poland_2024.webm/City_Driving_4K-_Krak%C3%B3w_Poland_2024.webm.1080p.vp9.webm"
ffmpeg -ss 58 -t 20 -i datasets/videos/raw/krakow_1080p_head.webm -an -vf "fps=30,scale=1280:-2" -c:v libx264 -crf 18 -pix_fmt yuv420p datasets/videos/sample_drive.mp4
```

## How to run

From the repo root, with Python 3.10, `pip install -r requirements.txt` and ffmpeg installed:

```bash
python src/video_pipeline/test_pipeline.py      # self-checks, no weights or video needed
python src/video_pipeline/video_info.py --video datasets/videos/sample_drive.mp4
python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --baseline --name baseline_detect
```

Weights go in `models/` (ignored by git): `traffic_lights_y11n_1280_best.pt` from Person 1's [Drive link](https://drive.google.com/file/d/1XzQtAnm4rvAYFkEH2tfSxi2aus6NLnha/view?usp=sharing), `traffic_signs_y11n_1280.pt` from the GitHub Release `signs-y11n-1280-v1`.

OpenCV writes `annotated.mp4` with the mp4v codec (about 50 MB for 20 s, and browsers cannot play it), so it is not in git. The committed copy is made with:

```bash
ffmpeg -i results/video_pipeline/<name>/annotated.mp4 -c:v libx264 -crf 28 -pix_fmt yuv420p results/video_pipeline/<name>/annotated_small.mp4
```

Tracking runs:

```bash
python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --baseline --track --name baseline_track
python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --track --name finetuned_track
```

## Tracking observations

From `results/video_pipeline/baseline_track/` and `finetuned_track/` on the 20 s test clip (600 frames). Frame numbers are `frame_index`.

- **Signs are tracked well.** A yield sign keeps id 20 from frame 105 to 207 (93 rows), a closer yield sign keeps id 41 from 284 to 339, and a far no_entry sign of about 10 px keeps id 51 from 362 to 584 even though it has no row at conf ≥ 0.25 in 55 of those frames (longest gap 23 frames). ByteTrack's low-score pass (conf 0.1–0.25) and its 30-frame buffer for lost tracks bridge these gaps.
- **Lost and back with a new id.** The yield track 20 ends at frame 207 and track 41 starts at 284. The gap (77 frames) is longer than ByteTrack's 30-frame buffer, so if this is the same sign it comes back with a new id. Without labels I cannot say for sure.
- **Tiny objects rarely get an id.** Speed limit signs of 10–12 px produce single-frame tracks (ids 8, 54) or a track with more gaps than hits (id 62: 5 hits over 32 frames).
- **Lights are the weak part on this clip.** Only one light becomes a confirmed track (id 56, red, frames 412–413); the other light boxes never become a track. The light model was trained on US footage (Bosch), and Polish lights look different, so this is mostly a domain gap and not a tracking problem. The Bosch test sequence below supports this.
- **No class flicker inside a track.** No track changes its class in either run, so the majority-vote smoothing that Person 1 suggested has nothing to fix on this clip yet.
- **Only confirmed tracks in tracking runs.** When a tracker has no confirmed track in a frame, Ultralytics returns that frame's raw boxes without ids; when it has one, it returns only the tracked boxes. So whether a new box showed up depended on other objects in the frame. The tracking runs therefore keep only rows with a track id (`keep_rows` in `pipeline_utils.py`). Before this rule, 54 of 381 rows in `finetuned_track` and 104 of 123 in `baseline_track` had no id. The COCO model's scores sit close to 0.25 and flicker, so only 2 of its tracks are ever confirmed (19 rows).
- **Track mode changes the boxes.** In track mode the boxes are ByteTrack's smoothed boxes, and an object's first frame can be missing, so `baseline_track` (19 rows) and `baseline_detect` (124 rows) do not match box for box. Detection metrics should come from a detect-only run.
- **False positives.** The COCO model calls a red oval shop sign a stop sign (frame 2).

## Bosch test sequence (US traffic lights)

The Kraków clip has few lights, and they look different from the US lights the light model was trained on. To check whether the weak light results come from the light type or from the pipeline, I also ran the pipeline on the official Bosch Small Traffic Lights test set (suggested by Person 1).

| | |
|---|---|
| Source | `dataset_test_rgb.zip` from [Zenodo record 12706046](https://zenodo.org/records/12706046) (Bosch Small Traffic Lights Dataset, Behrendt et al., ICRA 2017; licence: Bosch "License for NonCommercial Purposes", included in the zip) |
| Video | `datasets/videos/bosch_test.mp4` (not in git): 8334 frames, 1280×720, 15 fps, 9 min 16 s |
| Frame ↔ image | `frame_index` i is image `rgb/test/<24068 + 2·i>.png` (the images are numbered in steps of 2), so the labels in `test.yaml` can be joined directly |
| Use | demo and evaluation only; no thresholds are tuned on it. The committed CSVs and the annotated excerpt are derived from the dataset and fall under the same licence |

The video is built straight from the zip. macOS `unzip` cannot open zips over 4 GB, so `frames_to_video.py` reads the frames with Python in numeric order and pipes them to ffmpeg; nothing is extracted:

```bash
python src/video_pipeline/frames_to_video.py --zip datasets/dataset_test_rgb.zip --out datasets/videos/bosch_test.mp4
python src/video_pipeline/video_processing.py --video datasets/videos/bosch_test.mp4 --name bosch_test_detect
python src/video_pipeline/video_processing.py --video datasets/videos/bosch_test.mp4 --track --name bosch_test_track
```

Results (`results/video_pipeline/bosch_test_detect/` and `bosch_test_track/`):

- **Lights work on US footage.** The light model finds at least one light in 6496 of 8334 frames (78%). Tracking gives 147 light tracks; 68 of them have at least 15 boxes (one second of video) and the longest has 697. On the Kraków clip there was a single 2-frame light track. Together with the detection rate, this points to the light type (domain gap), not the pipeline, as the reason for the weak Kraków result. Track counts are only roughly comparable: ByteTrack keeps a lost track for 30 frames, which is 2 s at 15 fps (Bosch) but 1 s at 30 fps (Kraków).
- **No single-frame colour flicker.** No light track flips its colour for one frame and back. 19 of the 147 tracks change colour (22 changes in total). 21 of the changes follow the US light cycle (17 red → green, 2 green → yellow, 2 yellow → red), which looks like real light changes; the single red → yellow change is probably a misclassification. The labels can confirm this.
- **Speed.** 18.1 FPS with tracking and 20.2 FPS detection only, on the same MacBook (M4, MPS).
- **Annotated excerpt.** `bosch_test_track/annotated_excerpt.mp4` shows frames 7734–8183, the 30 s with the most tracked light boxes (926). The full annotated video is too large for git.
- Sign results on this sequence are not checked, because the Bosch labels only cover traffic lights.

## Integration plan

How the two detection modules are called and combined (code: `video_processing.py`):

1. **Calling the models.** Both are YOLO11n `.pt` files loaded with `YOLO(path)`: lights from Person 1 (classes red, yellow, green, off) and signs from Person 2 (stop, yield, speed_limit, no_entry, priority_road, no_parking). Each frame goes to each model with `imgsz=1280`. Class names come from `model.names`, so nothing is hard-coded.
2. **Tracking.** Each model is called with `model.track(frame, conf=0.1, persist=True, tracker="bytetrack.yaml")`. Every model keeps its own tracker, and afterwards only rows with conf ≥ 0.25 and a confirmed track id are kept.
3. **Combining the boxes.** The rows of both models for a frame are concatenated, each tagged with `source`. There is no cross-model NMS because the class sets do not overlap.
4. **Output.** One annotated video and one `detections.csv` in the common format, plus `run_info.txt` with settings and speed.

Next stage:
- Evaluate on video with labels (Person 4). The Bosch test sequences have light labels; there is no labelled video for signs yet.
- Majority-vote smoothing of the light colour per track (Person 1's suggestion), once a clip shows colour flicker inside a track.
- Run on the RTX 4070 machine to get speed numbers comparable with the teammates' GPU numbers.

## Implementation status

| Item | Status | Evidence |
|---|---|---|
| Architecture diagram and shared fields | done | this README, `results/video_pipeline/architecture.png` |
| Sample video, info and sample frames | done | `results/video_pipeline/video_info.txt`, `sample_frames/` |
| Detection on video, COCO baseline | done | `results/video_pipeline/baseline_detect/` |
| Tracking with ids (ByteTrack) | done | `results/video_pipeline/baseline_track/` |
| Both fine-tuned models in one pipeline | done | `results/video_pipeline/finetuned_track/` |
| Speed | measured | MacBook, Apple M4 (MPS), 1280×720 at imgsz 1280: both models + tracking 19.7 FPS end to end (light 3.7 ms, sign 3.5 ms inference per frame); COCO baseline 35.4 FPS detect only, 32.0 FPS with tracking |
| Second test video with labelled US lights (Bosch test set) | done | `results/video_pipeline/bosch_test_detect/`, `bosch_test_track/` |
| Evaluation on labelled video | next stage | Person 4 |
| Colour smoothing per track | next stage | no single-frame colour flicker on either test video |
