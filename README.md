# cv-practice

Personal computer vision practice project (Python).

## AGCP MVP (Assistive Gesture Control Platform)
This repo now includes an MVP under `src/cv_practice/assistive`:
- real-time gesture inference from MediaPipe hand landmarks
- lock/unlock safety state machine with hold/cooldown gating
- volume/media command routing
- calibration profiles (`configs/profiles/*.json`)
- session telemetry and optional labeled recording
- offline evaluation for recorded clips

### Run AGCP
1) Install runtime dependencies (inside your venv):
- `pip install mediapipe opencv-python pycaw pyyaml`

2) Run:
- `python -m cv_practice.assistive`
- optional config: `python -m cv_practice.assistive --config configs/assistive.default.yaml`

### AGCP Controls
- `Q`: quit
- `C`: run calibration and save profile
- `R`: toggle recording mode
- `1..5`: set recording ground-truth label (`idle/open/fist/pinch/tap`)

### Evaluate Recordings
- `python -m cv_practice.assistive.evaluation outputs/assistive/recordings/<file>.jsonl`

## Setup
1) Create venv
- `python -m venv .venv`

2) Activate it
- macOS/Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

3) Install dependencies
- `pip install -U pip`
- `pip install opencv-python numpy matplotlib`
- `pip install pytest ruff`

(Optional for notebooks)
- `pip install jupyter`

## Run
- `python -m cv_practice.main`

## Tests
- `pytest`

## Lint
- `ruff check .`
