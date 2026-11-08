# CENG391 Traffic Light and Traffic Sign Understanding

Course project for CENG391 (Introduction to Image Understanding).
The goal is to build a computer vision prototype that detects and classifies traffic lights and traffic signs in driving videos, tracks them across frames, and evaluates the system quantitatively.

## Team

| Member | Area | Folder |
|---|---|---|
| [Name] | Traffic light detection and state classification | `src/traffic_lights/` |
| [Name] | Traffic sign detection and type classification | `src/traffic_signs/` |
| [Name] | Video processing, tracking, integration | `src/video_pipeline/` |
| [Name] | Evaluation, robustness, experiments | `src/evaluation/` |

Group ID: [GroupID]

## Project Status

This repository is under active development. The current milestone is the Progress Report (due November 20, 2026).

| Component | Status |
|---|---|
| Repository setup | Done |
| Traffic sign dataset analysis | In progress |
| Traffic sign baseline detection | Planned |
| Traffic sign fine-tuning | Planned |

Update this table honestly as work progresses (Planned / In progress / Done).

## Repository Structure

```
CENG391-Traffic-Understanding/
├── datasets/          # Not tracked by git (see Datasets)
├── models/            # Not tracked by git (model weights)
├── src/
│   ├── traffic_signs/
│   ├── traffic_lights/
│   ├── video_pipeline/
│   └── evaluation/
├── results/
│   ├── traffic_signs/
│   ├── traffic_lights/
│   ├── videos/
│   └── metrics/
├── report/
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/berksahiinn/CENG391-Traffic-Understanding.git
cd CENG391-Traffic-Understanding
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Record the exact versions used for experiments:

```bash
pip show ultralytics
python --version
```

## Datasets

Datasets and model weights are NOT stored in this repository.

### Mapillary Traffic Sign Dataset (MTSD), traffic signs

- Page: https://www.mapillary.com/dataset/trafficsign
- Edition used: Research Edition
- Download the annotations (JSON) and the image packages, then place them as follows:

```
datasets/mtsd/
├── annotations/    # one JSON file per image
└── images/         # .jpg files
```

- Subset used: [number of images], split into train/val/test as [split description].
- License: see the dataset page. Used for academic research only.

### Traffic light dataset

- [Dataset name, link, and download instructions to be added by the traffic light owner]

## Usage

Traffic sign scripts (run from the repository root, in this order):

```bash
python src/traffic_signs/01_count_classes.py     # count objects per class
python src/traffic_signs/02_convert_to_yolo.py   # convert annotations to YOLO format
python src/traffic_signs/03_visualize.py         # draw ground-truth boxes for verification
```

Training and evaluation commands will be added here once they are run and verified.

## Class Definitions (Traffic Signs)

The class mapping is defined in `src/traffic_signs/class_map.yaml`.

| ID | Class |
|---|---|
| 0 | stop |
| 1 | yield |
| 2 | speed_limit |
| 3 | go_straight |
| 4 | no_u_turn |
| 5 | warning |

This mapping is a draft and may change after class counting. Notify the team before changing it.

## Common Detection Output Format

All modules should output the following fields so they can be combined in the video pipeline:

```
frame_index, bbox_xyxy, class_name, confidence, track_id (optional)
```

## Reproducibility Notes

For every experiment, record:

- Dataset subset and split
- Model name and version
- Training command and hyperparameters (epochs, image size, batch size, seed)
- Confidence/IoU thresholds
- Hardware (GPU/CPU)
- Library versions

## Results

Results will be added here after real experiments are completed. No placeholder numbers are reported.

## References

- C. Ertler, J. Mislej, T. Ollmann, L. Porzi, G. Neuhold, Y. Kuang. *The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale.* ECCV 2020.
- Ultralytics YOLO: https://docs.ultralytics.com

## Generative AI Usage Disclosure

[Tool name and exact version] was used for [planning, code scaffolding, debugging, etc.]. All code was run, verified, and can be explained by the team members.