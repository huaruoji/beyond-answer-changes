#!/usr/bin/env python3
"""Inspect JSONL manifests containing audio and timestamped evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    durations: list[float] = []
    event_counts: list[int] = []
    overlap_clips = 0
    bad_timestamps: list[str] = []
    classes: Counter[str] = Counter()

    for index, row in enumerate(rows):
        path = Path(row["audio"])
        info = sf.info(path)
        duration = info.frames / info.samplerate
        durations.append(duration)
        events = sorted(row.get("events", row.get("evidence", [])), key=lambda event: event["start"])
        event_counts.append(len(events))
        classes.update(str(event.get("event", "unknown")) for event in events)
        if any(float(event["start"]) < 0 or float(event["end"]) > duration or float(event["start"]) >= float(event["end"]) for event in events):
            bad_timestamps.append(str(row.get("id", index)))
        if any(float(left["end"]) > float(right["start"]) for left, right in zip(events, events[1:])):
            overlap_clips += 1

    summary = {
        "audio_count": len(rows),
        "duration_seconds": describe(durations),
        "events_per_clip": describe(event_counts),
        "event_classes": len(classes),
        "top_event_classes": classes.most_common(20),
        "overlapping_event_clips": overlap_clips,
        "invalid_timestamp_rows": bad_timestamps,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nExamples:")
    for row in rows[: args.examples]:
        print(json.dumps(row, ensure_ascii=False))


def describe(values: list[float | int]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {}
    return {key: float(value) for key, value in zip(
        ("min", "median", "mean", "p95", "max"),
        (x.min(), np.median(x), x.mean(), np.quantile(x, 0.95), x.max()),
    )}


if __name__ == "__main__":
    main()

