# Assistive Gesture Control Platform (AGCP)

AGCP is a real-time hand-gesture control project for assistive interaction.

## Project Layout

- `agcp/`: AGCP source code package.
- `configs/`: runtime config and profiles.
- `docs/`: design and implementation notes.
- `tests/`: AGCP-specific tests.
- `models/`: local model assets.
- `data/samples/`: local sample datasets (ignored in git).
- `outputs/`: runtime outputs (ignored in git).

## Run

- `python projects/assistive_gesture_control/run.py`
- optional config:
  - `python projects/assistive_gesture_control/run.py --config projects/assistive_gesture_control/configs/assistive.default.yaml`

## Tests

- `pytest -q projects/assistive_gesture_control/tests`
- gesture image test:
  - `$env:RUN_GESTURE_IMAGE_TESTS='1'; pytest -q projects/assistive_gesture_control/tests/test_assistive_gesture_samples.py`

## Evaluate

- `python projects/assistive_gesture_control/evaluate.py projects/assistive_gesture_control/outputs/recordings/<file>.jsonl`
