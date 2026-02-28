from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def evaluate_recordings(paths: list[str | Path]) -> dict:
    labels = set()
    tp = Counter()
    fp = Counter()
    fn = Counter()
    command_events = 0
    false_triggers = 0
    min_ts = 10**18
    max_ts = 0

    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ts = int(row.get("timestamp_ms", 0))
                if ts > 0:
                    min_ts = min(min_ts, ts)
                    max_ts = max(max_ts, ts)
                if row.get("kind") == "frame":
                    gt = row.get("ground_truth", "unknown")
                    pred = row.get("predicted", "unknown")
                    labels.add(gt)
                    labels.add(pred)
                    if gt == pred:
                        tp[gt] += 1
                    else:
                        fp[pred] += 1
                        fn[gt] += 1
                if row.get("kind") == "command":
                    command_events += 1
                    if row.get("ground_truth", "unknown") in {"unknown", "none", "idle"}:
                        false_triggers += 1

    metrics = defaultdict(dict)
    for label in sorted(labels):
        if label in {"unknown", "none"}:
            continue
        precision = tp[label] / max(1, tp[label] + fp[label])
        recall = tp[label] / max(1, tp[label] + fn[label])
        metrics[label]["precision"] = round(precision, 4)
        metrics[label]["recall"] = round(recall, 4)
        metrics[label]["tp"] = tp[label]
        metrics[label]["fp"] = fp[label]
        metrics[label]["fn"] = fn[label]

    runtime_min = max(1e-6, (max_ts - min_ts) / 1000 / 60) if max_ts > min_ts else 1e-6
    summary = {
        "labels": metrics,
        "false_triggers_per_min": round(false_triggers / runtime_min, 4),
        "command_events": command_events,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AGCP recorded clips (.jsonl).")
    parser.add_argument("recordings", nargs="+", help="Paths to recording jsonl files.")
    args = parser.parse_args()
    report = evaluate_recordings(args.recordings)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

