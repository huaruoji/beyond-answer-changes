#!/usr/bin/env python3
"""Decide whether a source corpus can support *local* audio counterfactual items.

Motivation
----------
The invalidated L2/sweep manifests selected items with no constraint on how much
of the clip the annotated evidence covers. Measured after the fact, the median
target span covered 99.4% of its clip, so "edit the labelled evidence" was very
often "edit the whole clip" and the local-attribution claim had no support.

This script answers the gating question *before* any waveform is generated and
*before* any model is run: how many independent clips admit an item that is
local, well posed, and carries a matched control?

Eligibility (all three required)
--------------------------------
C1 local target
    an event whose annotation is a single unambiguous interval occupying
    ``--min-coverage`` .. ``--max-coverage`` of its clip.
C2 separable control
    a second, textually different single-interval event in the same clip whose
    interval does not overlap the target interval and is itself not near
    whole-clip (so removing it is a local edit too).
C3 spliceable donor
    a single-interval event in a *different* clip whose description does not
    occur anywhere in the target clip, and whose duration is within
    ``--donor-tolerance`` of the target interval (the donor is resampled into
    the target span, so a large mismatch means an audible time-stretch).

Everything is computed from annotations plus decoded durations. No audio is
written, no model is called.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

INTERVAL_RE = re.compile(
    r"<\|([0-9]+(?:\.[0-9]+)?)\|>\s*to\s*<\|([0-9]+(?:\.[0-9]+)?)\|>"
)
QUOTE_RE = re.compile(r"['\u2018](.+?)['\u2019]")
WORD_RE = re.compile(r"[a-z0-9]+")

# The corpus mixes two question families. Some splits (source `tacos`) always
# quote the queried event; others (source `audioset_strong`) embed it in a
# sentence with no delimiter. Deterministic pattern extraction keeps this
# ADR-010 compliant: no language model touches the labels.
EVENT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"identify when (.+?) (?:happens|occurs|is heard|can be heard|starts|begins|ends)",
        r"when does (.+?) (?:occur|happen|start|begin|appear|end|take place)",
        r"when do (.+?) (?:occur|happen|start|begin|appear|end|take place)",
        r"locate (.+?) in the audio",
        r"timestamps? does (.+?) (?:appear|occur|happen|start|begin|end)",
        r"timestamps? do (.+?) (?:appear|occur|happen|start|begin|end)",
        r"at what (?:time|timestamp|point)s? (?:does|do|is|are) (.+?) (?:appear|occur|happen|start|begin|end|heard)",
        r"when (?:is|are) (.+?) (?:heard|present|audible|played)",
        r"where does (.+?) occur",
        # Catch-alls for intransitive verbs the lists above do not enumerate
        # ("the frog croaks", "the mosquito buzz", "the baby laughs").
        r"identify when (.+?)\?",
        r"timestamps? (?:does|do) (.+?)(?: in the audio)?\?",
    )
]
ALLOWED_CHANNELS = (1, 2)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "mean": round(float(array.mean()), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def _extract_event(question: str) -> str | None:
    """Recover the queried event phrase from either question family."""

    quoted = QUOTE_RE.findall(question or "")
    if quoted:
        candidate = quoted[0].strip()
        if candidate:
            return candidate
    for pattern in EVENT_PATTERNS:
        match = pattern.search(question or "")
        if match:
            candidate = match.group(1).strip(" .,?")
            if 2 < len(candidate) < 120:
                return candidate
    return None


def _decode_duration(payload: bytes) -> float | None:
    """Duration in seconds, or None when the codec cannot be read."""

    try:
        info = sf.info(io.BytesIO(payload))
    except Exception:
        return None
    if info.samplerate <= 0 or info.frames <= 0:
        return None
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--max-coverage", type=float, default=0.30)
    parser.add_argument("--control-max-coverage", type=float, default=0.50)
    parser.add_argument("--min-gap-s", type=float, default=0.20)
    parser.add_argument("--donor-tolerance", type=float, default=0.50)
    parser.add_argument(
        "--control-tolerance",
        type=float,
        default=1.0,
        help="max relative span mismatch between target and control edit; "
        "a matched control must perturb a comparable amount of audio",
    )
    parser.add_argument("--min-target-span-s", type=float, default=0.0)
    args = parser.parse_args()

    raw_rows = 0
    single_interval_rows = 0
    multi_interval_rows = 0
    unreadable_rows = 0
    quoted_rows = 0

    payload_by_key: dict[str, bytes] = {}
    # event description -> every interval observed for it, per clip
    events_by_key: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for parquet in args.parquets:
        table = pq.read_table(
            parquet, columns=["audio_key", "question", "answer", "audio"]
        ).to_pylist()
        for row in table:
            raw_rows += 1
            key = row.get("audio_key")
            if not key:
                unreadable_rows += 1
                continue
            payload = (row.get("audio") or {}).get("bytes")
            if payload:
                payload_by_key.setdefault(key, payload)
            intervals = [
                (float(start), float(end))
                for start, end in INTERVAL_RE.findall(row.get("answer") or "")
                if float(end) > float(start)
            ]
            quoted = QUOTE_RE.findall(row.get("question") or "")
            if quoted:
                quoted_rows += 1
            description = _extract_event(row.get("question") or "")
            if not description or not intervals:
                unreadable_rows += 1
                continue
            if len(intervals) == 1:
                single_interval_rows += 1
            else:
                multi_interval_rows += 1
            description = description.casefold()
            if description:
                events_by_key[key][description].extend(intervals)

    durations: dict[str, float] = {}
    for key, payload in payload_by_key.items():
        duration = _decode_duration(payload)
        if duration is not None:
            durations[key] = duration

    # Only single-occurrence events can carry an unambiguous "the evidence is
    # here" claim; a description that repeats cannot be removed locally.
    single_events: dict[str, list[dict[str, Any]]] = {}
    coverage_values: list[float] = []
    for key, events in events_by_key.items():
        duration = durations.get(key)
        if not duration:
            continue
        usable = []
        for description, intervals in events.items():
            if len(intervals) != 1:
                continue
            start, end = intervals[0]
            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))
            if end <= start:
                continue
            span = end - start
            usable.append(
                {
                    "event": description,
                    "start": start,
                    "end": end,
                    "span": span,
                    "coverage": span / duration,
                    "words": set(WORD_RE.findall(description)),
                }
            )
        single_events[key] = usable
        coverage_values.extend(item["coverage"] for item in usable)

    # Donor pool: every single-interval event in the corpus, indexed by duration.
    donors = [
        (key, item)
        for key, items in single_events.items()
        for item in items
        if item["words"]
    ]

    eligible_clips: list[dict[str, Any]] = []
    target_pool: list[tuple[str, dict[str, Any]]] = []
    for key, items in single_events.items():
        for item in items:
            if item["span"] < args.min_target_span_s:
                continue
            if args.min_coverage <= item["coverage"] <= args.max_coverage:
                target_pool.append((key, item))

    for key, target in target_pool:
        duration = durations[key]
        existing = {item["event"] for item in single_events[key]}

        controls = []
        for other in single_events[key]:
            if other["event"] == target["event"]:
                continue
            if other["coverage"] > args.control_max_coverage:
                continue
            gap = max(
                other["start"] - target["end"],
                target["start"] - other["end"],
            )
            if gap < args.min_gap_s:
                continue
            # A control edit that touches far less (or far more) audio than the
            # target edit is not a matched contrast.
            if abs(other["span"] - target["span"]) / target["span"] > args.control_tolerance:
                continue
            controls.append(other)
        if not controls:
            continue

        chosen_control = min(controls, key=lambda item: item["coverage"])
        donor_matches = []
        for donor_key, donor in donors:
            if donor_key == key:
                continue
            if donor["event"] in existing:
                continue
            if donor["event"] == chosen_control["event"]:
                # A donor equal to the control makes the two edits confusable.
                continue
            if not donor["words"]:
                continue
            tolerance = abs(donor["span"] - target["span"]) / target["span"]
            if tolerance > args.donor_tolerance:
                continue
            donor_matches.append(
                {
                    "donor_key": donor_key,
                    "donor_event": donor["event"],
                    "donor_span": round(donor["span"], 3),
                    "duration_mismatch": round(tolerance, 3),
                }
            )
        if not donor_matches:
            continue

        donor_matches.sort(key=lambda item: item["duration_mismatch"])
        eligible_clips.append(
            {
                "audio_key": key,
                "clip_duration_s": round(duration, 3),
                "target_event": target["event"],
                "target_span_s": round(target["span"], 3),
                "target_coverage": round(target["coverage"], 4),
                "control_event": chosen_control["event"],
                "control_span_s": round(chosen_control["span"], 3),
                "control_coverage": round(chosen_control["coverage"], 4),
                "n_controls": len(controls),
                "n_donors": len(donor_matches),
                "best_donor": donor_matches[0],
            }
        )

    unique_eligible_clips = len({item["audio_key"] for item in eligible_clips})
    report = {
        "parquets": [str(path) for path in args.parquets],
        "thresholds": {
            "min_coverage": args.min_coverage,
            "max_coverage": args.max_coverage,
            "control_max_coverage": args.control_max_coverage,
            "control_tolerance": args.control_tolerance,
            "min_gap_s": args.min_gap_s,
            "min_target_span_s": args.min_target_span_s,
            "donor_tolerance": args.donor_tolerance,
        },
        "corpus": {
            "rows": raw_rows,
            "unique_clips": len(events_by_key),
            "decoded_clips": len(durations),
            "rows_with_single_interval": single_interval_rows,
            "rows_with_multiple_intervals": multi_interval_rows,
            "rows_unusable": unreadable_rows,
            "rows_with_quoted_event": quoted_rows,
            "sources": dict(Counter(path.name for path in args.parquets)),
        },
        "single_interval_events": len(coverage_values),
        "coverage_distribution": _percentiles(coverage_values),
        "coverage_buckets": {
            "le_0.05": int(np.sum(np.asarray(coverage_values) <= 0.05)),
            "0.05_to_0.30": int(
                np.sum(
                    (np.asarray(coverage_values) > 0.05)
                    & (np.asarray(coverage_values) <= 0.30)
                )
            ),
            "0.30_to_0.50": int(
                np.sum(
                    (np.asarray(coverage_values) > 0.30)
                    & (np.asarray(coverage_values) <= 0.50)
                )
            ),
            "0.50_to_0.80": int(
                np.sum(
                    (np.asarray(coverage_values) > 0.50)
                    & (np.asarray(coverage_values) <= 0.80)
                )
            ),
            "gt_0.80": int(np.sum(np.asarray(coverage_values) > 0.80)),
        }
        if coverage_values
        else {},
        "targets_passing_C1": len(target_pool),
        "eligible_items": len(eligible_clips),
        "eligible_unique_clips": unique_eligible_clips,
    }
    if eligible_clips:
        report["coverage_distribution_eligible"] = _percentiles(
            [item["target_coverage"] for item in eligible_clips]
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        args.output.with_suffix(".items.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in eligible_clips)
            + "\n"
        )


if __name__ == "__main__":
    main()
