# Traffic Light and Traffic Sign Understanding

CENG391 course project. A computer vision prototype that detects and classifies traffic lights and traffic signs in driving videos, tracks them across frames, and evaluates the system quantitatively.

## Pipeline

Driving video → Object detection (YOLO) → Classification (light state / sign type) → Tracking (ByteTrack) → Annotated video + metrics

## Team

| Member | Responsibility | Folder |
|---|---|---|
| [Name] | Traffic light detection and state classification | `src/traffic_lights/` |
| [Name] | Traffic sign detection and type classification | `src/traffic_signs/` |
| Ege Ayhan | Video processing, tracking, integration | `src/video_pipeline/` |
| [Name] | Evaluation, robustness, experiments | `src/evaluation/` |

## Installation

```bash
git clone https://github.com/berksahiinn/CENG391-Traffic-Understanding.git
cd CENG391-Traffic-Understanding
pip install -r requirements.txt
```

## Datasets

Datasets and model weights are not included in this repository. Place downloaded data under `datasets/`.

- **Traffic lights:** [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)
- **Traffic signs:** [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign) (research edition)

## Detection Output Format

All modules use the same output fields so they can be combined in the video pipeline:

```
frame_index, bbox_xyxy, class_name, confidence, track_id (optional)
```

In CSV files the box is written as four columns `x1, y1, x2, y2` (pixels of the original frame). The video pipeline adds a `source` column (`light`, `sign` or `coco`) because track ids are counted per model, so one tracked object is `(source, track_id)`:

```
frame_index, x1, y1, x2, y2, class_name, confidence, track_id, source
```

Details: `src/video_pipeline/README.md`.

## Project Structure

```
src/
├── traffic_lights/    # light detection and state classification
├── traffic_signs/     # sign detection and type classification
├── video_pipeline/    # video processing and tracking
└── evaluation/        # metrics and experiments
results/               # outputs and figures
report/                # progress report (LaTeX)
```

## Usage

Run commands will be added by each module owner once they are tested.

Video pipeline (Person 3), from the repo root:

```bash
python src/video_pipeline/test_pipeline.py
python src/video_pipeline/video_info.py --video datasets/videos/sample_drive.mp4
python src/video_pipeline/video_processing.py --video datasets/videos/sample_drive.mp4 --track --name my_run
```

`--name` picks the output folder under `results/video_pipeline/`; the folders already there hold the committed results. Weights, the test video and all options: `src/video_pipeline/README.md`.

## References

- Ertler et al., *The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale*, ECCV 2020.
- Behrendt et al., *A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification*, ICRA 2017.
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- [Ultralytics YOLO](https://docs.ultralytics.com)