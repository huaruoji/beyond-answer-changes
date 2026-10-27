#!/usr/bin/env python3
"""Build the natural-audio transfer manifest from TEMPO's tacos evaluation split.

This runs the same seven conditions and the same metrics as the synthetic grid,
on real recorded audio with human-authored annotations, so the synthetic
findings can be checked against something nobody synthesised.

What differs from the synthetic builder, and why:

* There is no generated background. The recording *is* the soundscape, so
  removal conditions splice in a different event-free window of the same
  recording rather than a separate background file. That keeps the ambience
  consistent but means the edit is a splice, which the synthetic route avoids.
* Annotation spans are only known to 0.1 s, and a description may occur more
  than once. Only events with exactly one interval are used, per the feasibility
  screen.
* Two donors are needed, from *other* clips: one of the same class as the query
  (for the truth-preserving in-place rewrite) and one of a different class (for
  the truth-flipping edit). The feasibility scan only looked for the latter.
* No event-free window is guaranteed, so items where the matched control cannot
  be placed are dropped rather than patched.

Everything downstream -- the conditions, the invariants, the metrics -- is the
same code path as the synthetic grid.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.feasibility_scan import (  # noqa: E402
    INTERVAL_RE,
    QUOTE_RE,
    _extract_event,
    _decode_duration,
)
from src.synth import (  # noqa: E402
    CONDITIONS,
    DEFAULT_TARGET_PEAK,
    DEFAULT_SAMPLE_RATE,
    InstrumentError,
    PlacedEvent,
    Soundscape,
    apply_condition,
    assert_placement_disjoint,
    find_free_window,
    read_audio_mono,
    resolve_condition_span,
)

#: The natural route cannot build target_replaced_same: descriptions are free
#: text and essentially unique per clip (3848 distinct over 2000 clips, 92 in
#: more than one). The matched control is nuisance_replaced instead -- the same
#: donor waveform written by the same operation and the same crossfade into a
#: different span, which flips nothing.
NATURAL_CONDITIONS = (
    "original",
    "target_replaced_diff",
    "nuisance_replaced",
    "nuisance_removed",
    "background_removed",
    "sham",
    "silence",
)

ROLES = {
    "original": "none",
    "target_replaced_diff": "relevant_change",
    "nuisance_replaced": "irrelevant",
    "nuisance_removed": "irrelevant",
    "background_removed": "irrelevant",
    "sham": "sham",
    "silence": "global_removal",
}

QUESTION = (
    "Listen to the audio. Which of these two sounds is present in it? "
    "Option A: '{a}'. Option B: '{b}'. "
    "Respond with exactly Answer: A or Answer: B."
)

DONOR_TOLERANCE = 0.5
MIN_GAP_S = 0.30
MIN_TARGET_SPAN_S = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquets", nargs="+", type=Path)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-coverage", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--control-tolerance", type=float, default=0.40)
    parser.add_argument("--split", default="natural")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, bytes] = {}
    events_by_key: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for parquet in args.parquets:
        for row in pq_read(parquet):
            key = row.get("audio_key")
            if not key:
                continue
            data = (row.get("audio") or {}).get("bytes")
            if data:
                payload.setdefault(key, data)
            intervals = [
                (float(a), float(b))
                for a, b in INTERVAL_RE.findall(row.get("answer") or "")
                if float(b) > float(a)
            ]
            description = _extract_event(row.get("question") or "")
            if not description or not intervals:
                continue
            events_by_key[key][description.casefold()].extend(intervals)

    durations = {
        key: duration
        for key, data in payload.items()
        if (duration := _decode_duration(data)) is not None
    }

    # Only single-occurrence events can carry "the evidence is here".
    single: dict[str, list[dict[str, Any]]] = {}
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
            if end - start < MIN_TARGET_SPAN_S:
                continue
            usable.append(
                {
                    "event": description,
                    "start": start,
                    "end": end,
                    "span": end - start,
                    "coverage": (end - start) / duration,
                }
            )
        single[key] = usable

    # Donor pools indexed by class description.
    by_class: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, items in single.items():
        for item in items:
            by_class[item["event"]].append((key, item))

    rejections: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    built = 0
    attempts = 0
    max_attempts = args.target * 60
    keys = sorted(single)
    order = list(keys)
    rng.shuffle(order)

    while built < args.target and attempts < max_attempts:
        attempts += 1
        key = order[attempts % len(order)]
        duration = durations.get(key)
        if not duration:
            continue
        items = [
            item
            for item in single[key]
            if args.min_coverage <= item["coverage"] <= args.max_coverage
        ]
        if not items:
            rejections["no local target in clip"] += 1
            continue
        target = rng.choice(items)

        controls = [
            other
            for other in single[key]
            if other["event"] != target["event"]
            and other["coverage"] <= 0.5
            and max(other["start"] - target["end"], target["start"] - other["end"]) >= MIN_GAP_S
            and abs(other["span"] - target["span"]) / target["span"] <= args.control_tolerance
        ]
        if not controls:
            rejections["no matched control event in clip"] += 1
            continue
        control = min(controls, key=lambda item: item["coverage"])

        diff = _donor(
            by_class, None, key, target["span"], rng,
            exclude_descriptions={target["event"], control["event"]},
        )
        other = _donor(
            by_class, None, key, control["span"], rng,
            exclude_descriptions={target["event"], control["event"], _desc(diff)},
        ) if diff else None
        if diff is None or other is None:
            rejections["no usable donor pair"] += 1
            continue

        try:
            waveform, sample_rate = read_audio_mono(
                _materialise(payload[key], key, args.audio_dir), DEFAULT_SAMPLE_RATE
            )
        except Exception:
            rejections["clip unreadable"] += 1
            continue
        # Recorded clips arrive at whatever level they were published at, often
        # full scale. Match the synthetic route's target peak with one gain so
        # the two routes feed the models comparable levels and so the audit's
        # clipping guard means the same thing on both.
        peak = float(np.abs(waveform).max()) if waveform.size else 0.0
        if peak > 0:
            waveform = (waveform * (DEFAULT_TARGET_PEAK / peak)).astype(np.float32)

        scape = Soundscape(
            audio_path=args.audio_dir / f"{args.split}-{built:05d}__original.wav",
            sample_rate=sample_rate,
            duration_s=len(waveform) / sample_rate,
            background_path=str(key),
            events=[
                PlacedEvent(
                    role="query",
                    label=target["event"],
                    source_path=str(key),
                    start_s=target["start"],
                    end_s=target["end"],
                    snr_db=float("nan"),
                    source_span_s=target["span"],
                ),
                PlacedEvent(
                    role="nuisance",
                    label=control["event"],
                    source_path=str(key),
                    start_s=control["start"],
                    end_s=control["end"],
                    snr_db=float("nan"),
                    source_span_s=control["span"],
                ),
            ],
        )
        try:
            assert_placement_disjoint(scape)
        except InstrumentError:
            rejections["annotation overlap"] += 1
            continue

        query_samples = int(round(scape.query.duration_s * sample_rate))
        try:
            find_free_window(
                scape.events, len(waveform), sample_rate, query_samples, random.Random(args.seed + built)
            )
        except InstrumentError:
            rejections["no event-free control window"] += 1
            continue

        donors = {}
        donor_spans = {}
        for name, donor in (("target_replaced_diff", diff), ("nuisance_replaced", other)):
            samples = _read_donor(donor, sample_rate, payload, args.audio_dir)
            donors[name] = samples
            # Use the length actually read. round(end*sr) - round(start*sr) can
            # differ from round(span*sr) by one sample, and the fitter rejects a
            # span longer than the array, which dropped items for an off-by-one.
            donor_spans[name] = len(samples) / sample_rate

        item_id = f"{args.split}-{built:05d}"
        item_rows: list[dict[str, Any]] = []
        failed = False
        for condition in NATURAL_CONDITIONS:
            edit_rng = random.Random(args.seed * 1000 + built)
            try:
                span = resolve_condition_span(scape, condition, len(waveform), edit_rng)
                edited = apply_condition(
                    waveform,
                    scape,
                    condition,
                    donor_waveform=donors.get(condition),
                    donor_span_samples=(
                        int(round(donor_spans[condition] * sample_rate))
                        if condition in donor_spans
                        else None
                    ),
                    span=span,
                    rng=edit_rng,
                )
            except InstrumentError as exc:
                rejections[f"condition {condition}: {exc}"[:100]] += 1
                failed = True
                break
            audio_path = args.audio_dir / f"{item_id}__{condition}.wav"
            sf.write(audio_path, np.clip(edited, -1.0, 1.0), sample_rate, subtype="PCM_16")
            truth = diff[1]["event"] if condition == "target_replaced_diff" else target["event"]
            edited_span = (
                None if span is None else [round(span[0] / sample_rate, 4), round(span[1] / sample_rate, 4)]
            )
            for surface in (0, 1):
                options = (
                    {"A": target["event"], "B": diff[1]["event"]}
                    if surface == 0
                    else {"A": diff[1]["event"], "B": target["event"]}
                )
                item_rows.append(
                    {
                        "pair_id": f"{item_id}__o{surface}",
                        "source_pair_id": item_id,
                        "surface_order": surface,
                        "condition": condition,
                        "intervention_role": ROLES[condition],
                        "audio": str(audio_path),
                        "question": QUESTION.format(a=options["A"], b=options["B"]),
                        "ground_truth": "A" if options["A"] == truth else "B",
                        "option_events": options,
                        "evidence": [
                            {"event": target["event"], "start": target["start"], "end": target["end"]}
                        ],
                        "edited_span_s": edited_span,
                        "query_class": target["event"],
                        "absent_class": diff[1]["event"],
                        "nuisance_classes": [control["event"]],
                        "query_span_s": round(target["span"], 4),
                        "query_start_s": round(target["start"], 4),
                        "snr_db": None,
                        "background_class": str(key).split("/")[0],
                        "split": args.split,
                        "seed": args.seed,
                        "n_nuisance": 1,
                        "source_audio_key": str(key),
                    }
                )
        if failed:
            for row in item_rows:
                Path(row["audio"]).unlink(missing_ok=True)
            continue
        rows.extend(item_rows)
        built += 1
        if built % 25 == 0:
            print(f"  built {built}/{args.target}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    summary = {
        "split": args.split,
        "seed": args.seed,
        "items": built,
        "records": len(rows),
        "attempts": attempts,
        "sources": [str(p) for p in args.parquets],
        "condition_label_counts": {
            condition: dict(
                Counter(r["ground_truth"] for r in rows if r["condition"] == condition)
            )
            for condition in sorted({r["condition"] for r in rows})
        },
        "rejections": dict(rejections.most_common(15)),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def pq_read(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(
        path, columns=["audio_key", "question", "answer", "audio"]
    ).to_pylist()


def _donor(
    by_class,
    description,
    exclude_key,
    target_span,
    rng,
    exclude_descriptions: set[str] | None = None,
):
    """A donor from a different clip, trimmed to its own annotated span.

    ``description`` set means "same class as the query" (the truth-preserving
    rewrite); ``None`` means "any class not already in this clip and not one of
    the options" (the truth-flipping rewrite).
    """

    if description is not None:
        candidates = [
            (key, item)
            for key, item in by_class.get(description, [])
            if key != exclude_key
        ]
    else:
        blocked = exclude_descriptions or set()
        candidates = [
            (key, item)
            for other, items in by_class.items()
            if other not in blocked
            for key, item in items
            if key != exclude_key
        ]
    candidates = [
        (key, item)
        for key, item in candidates
        if abs(item["span"] - target_span) / target_span <= DONOR_TOLERANCE
    ]
    return rng.choice(candidates) if candidates else None


def _desc(donor) -> str:
    return donor[1]["event"] if donor else ""


def _read_donor(donor, sample_rate: int, payload: dict[str, bytes], audio_dir: Path) -> np.ndarray:
    """Read only the donor's own annotated span, resampled to the clip rate."""

    key, item = donor
    path = _materialise(payload[key], key, audio_dir)
    waveform, _ = read_audio_mono(path, sample_rate)
    start = int(round(item["start"] * sample_rate))
    end = int(round(item["end"] * sample_rate))
    return waveform[start:end].astype(np.float32)


def _materialise(data: bytes, key: str, directory: Path) -> Path:
    suffix = ".wav" if data[:4] == b"RIFF" else ".mp3"
    path = directory / f"__clip_{Path(key).stem}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


if __name__ == "__main__":
    main()
