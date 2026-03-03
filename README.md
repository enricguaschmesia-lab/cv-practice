# cv-practice

Repository for multiple computer vision test projects.

## Repository Structure

- `projects/assistive_gesture_control/`: self-contained AGCP project (code, configs, docs, tests, models, data).
- `projects/air_canvas_studio/`: standalone Air Canvas project.
- `experiments/scratch/`: one-off scripts and prototypes (not full projects).

## Current Projects

### 1) Assistive Gesture Control Platform (AGCP)

Core code lives in `projects/assistive_gesture_control/agcp`.

Project assets:
- `projects/assistive_gesture_control/agcp/`
- `projects/assistive_gesture_control/configs/`
- `projects/assistive_gesture_control/docs/`
- `projects/assistive_gesture_control/tests/`
- `projects/assistive_gesture_control/models/`
- `projects/assistive_gesture_control/data/samples/`

Run:
- `python projects/assistive_gesture_control/run.py`
- custom config: `python projects/assistive_gesture_control/run.py --config projects/assistive_gesture_control/configs/assistive.default.yaml`

Evaluate recordings:
- `python projects/assistive_gesture_control/evaluate.py projects/assistive_gesture_control/outputs/recordings/<file>.jsonl`

### 2) Air Canvas Studio

Script:
- `projects/air_canvas_studio/AirCanvasStudio.py`

Run:
- `python projects/air_canvas_studio/AirCanvasStudio.py`

## Setup

1. `python -m venv .venv`
2. Activate:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
3. Install:
   - `pip install -U pip`
   - `pip install -e .`
   - `pip install opencv-python numpy matplotlib mediapipe pycaw pyyaml pytest ruff`

## Testing

- Full suite: `pytest -q`
- Gesture image test:
  - PowerShell: `$env:RUN_GESTURE_IMAGE_TESTS='1'; pytest -q projects/assistive_gesture_control/tests/test_assistive_gesture_samples.py`

## Lint

- `ruff check .`
