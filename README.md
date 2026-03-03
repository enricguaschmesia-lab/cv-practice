# cv-practice

Personal portfolio of computer vision and applied ML projects.

## Repository Structure

- `projects/`: complete, self-contained projects.
- `experiments/scratch/`: quick prototypes and idea validation scripts.

## Current Projects

### 1) Assistive Gesture Control Platform (AGCP)

A real-time hand-gesture control system designed for assistive interaction.

Highlights:
- hand landmark tracking + feature-based gesture inference,
- calibrated user profiles for robust personalization,
- safety-first state machine (lock/unlock, hold/cooldown gating),
- command execution for media/volume workflows,
- debug telemetry and evaluation tooling.

Project folder: `projects/assistive_gesture_control/`

Main code: `projects/assistive_gesture_control/agcp/`

Run:
- `python projects/assistive_gesture_control/run.py`
- Custom config: `python projects/assistive_gesture_control/run.py --config projects/assistive_gesture_control/configs/assistive.default.yaml`

Evaluate recordings:
- `python projects/assistive_gesture_control/evaluate.py projects/assistive_gesture_control/outputs/recordings/<file>.jsonl`

### 2) Air Canvas Studio

An interactive vision project where the user draws in the air using hand tracking.

Highlights:
- gesture-driven drawing and erase modes,
- real-time visual feedback and interaction controls,
- practical HCI-focused computer vision prototype.

Project folder: `projects/air_canvas_studio/`

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
- AGCP gesture image test:
  - PowerShell: `$env:RUN_GESTURE_IMAGE_TESTS='1'; pytest -q projects/assistive_gesture_control/tests/test_assistive_gesture_samples.py`

## Lint

- `ruff check .`
