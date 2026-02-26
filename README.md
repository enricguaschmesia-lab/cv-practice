# cv-practice

Personal computer vision practice project (Python).

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