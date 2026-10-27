#!/usr/bin/env python3
"""Build synthetic soundscape manifests for local evidence attribution.

One item is one soundscape containing a query event, a nuisance event, and a
background, rendered once and then edited into seven conditions. See
``docs/synthesis_design.md`` for the design and ``src/synth.py`` for the
invariants this script relies on rather than re-checks.

Item acceptance is strict. An item is dropped, never patched, when:

* a class lacks a long enough instance or enough distinct donor instances;
* placements cannot be made disjoint;
* the clip has no event-free window long enough for the matched control;
* the rendered audio clips or comes back at the wrong sample rate.

Every rejection is counted and reported, because the rejection rate is itself a
property of the event library worth knowing.
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

from src.synth import (  # noqa: E402
    CONDITIONS,
    DEFAULT_DURATION_S,
    InstrumentError,
    Soundscape,
    apply_condition,
    assert_placement_disjoint,
    find_free_window,
    index_backgrounds,
    index_wavs,
    load_vocab,
    plan_placements,
    read_audio_mono,
    render_soundscape,
    resolve_condition_span,
    wav_duration,
)

#: Condition -> intervention_role, so the existing metrics module keeps working.
ROLES = {
    "original": "none",
    "target_replaced_diff": "relevant_change",
    "target_replaced_same": "relevant_preserving",
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

ALLOWED_DONOR_MISMATCH = 0.5  # donor span must be within 50% of the query span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--foreground-dir", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--jams-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--n-nuisance", type=int, default=1)
    parser.add_argument("--min-event-s", type=float, default=0.4)
    parser.add_argument("--max-event-s", type=float, default=1.5)
    parser.add_argument("--classes", type=Path, help="restrict to a class list")
    parser.add_argument("--background-class", help="restrict to one background class")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="subset of conditions to render; vocabulary screening only needs "
        "original and silence, which is far less audio to generate and ship",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="screening mode: one event, options are the true class and a random other",
    )
    parser.add_argument(
        "--keep-base",
        action="store_true",
        help="keep scaper's 32-bit base render; it is redundant with the "
        "original condition and costs 640 KB per item",
    )
    args = parser.parse_args()
    if not set(args.conditions) <= set(CONDITIONS):
        raise SystemExit(f"unknown conditions: {sorted(set(args.conditions) - set(CONDITIONS))}")

    rng = random.Random(args.seed)
    vocab = load_vocab(args.vocab)
    if args.classes:
        keep = {line.strip().lower() for line in args.classes.read_text().splitlines() if line.strip()}
        vocab = [name for name in vocab if name.lower() in keep]
    events = index_wavs(args.foreground_dir, min_duration_s=args.min_event_s)
    vocab = [name for name in vocab if name in events]
    backgrounds = index_backgrounds(args.background_dir, min_duration_s=args.duration)
    if args.background_class:
        backgrounds = [p for p in backgrounds if p.parent.name == args.background_class]
    if len(vocab) < 3:
        raise SystemExit(f"need at least 3 usable classes, found {len(vocab)}")
    if not backgrounds:
        raise SystemExit("no background long enough to fill a clip")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.jams_dir.mkdir(parents=True, exist_ok=True)

    rejections: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    built = 0
    attempts = 0
    max_attempts = args.target * 40

    while built < args.target and attempts < max_attempts:
        attempts += 1
        try:
            item = _build_item(args, vocab, events, backgrounds, rng, built, rejections)
        except InstrumentError as exc:
            rejections[f"instrument: {exc}"[:110]] += 1
            continue
        if item is None:
            continue
        rows.extend(item)
        built += 1
        if built % 25 == 0:
            print(f"  built {built}/{args.target} (attempts {attempts})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    summary = {
        "split": args.split,
        "seed": args.seed,
        "target": args.target,
        "items": built,
        "records": len(rows),
        "attempts": attempts,
        "conditions": sorted({row["condition"] for row in rows}),
        "classes_available": len(vocab),
        "classes_used": len({row["query_class"] for row in rows}),
        "backgrounds_available": len(backgrounds),
        "condition_label_counts": {
            condition: dict(
                Counter(
                    row["ground_truth"]
                    for row in rows
                    if row["condition"] == condition
                )
            )
            for condition in sorted({row["condition"] for row in rows})
        },
        "rejections": dict(rejections.most_common(15)),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def _build_item(args, vocab, events, backgrounds, rng, index, rejections):
    """Render one item across all conditions, or return None to reject it."""

    if args.screen:
        query_class = rng.choice(vocab)
        other_choices = [name for name in vocab if name != query_class]
        if not other_choices:
            return None
        absent_class = rng.choice(other_choices)
        nuisance_classes: list[str] = []
    else:
        query_class, absent_class, *rest = rng.sample(vocab, 3)
        nuisance_classes = [rest[0]] if rest else []
        while len(nuisance_classes) < args.n_nuisance:
            candidate = rng.choice(vocab)
            if candidate not in {query_class, absent_class} and candidate not in nuisance_classes:
                nuisance_classes.append(candidate)
            if len(nuisance_classes) >= len(vocab) - 2:
                break

    query_source = _pick_instance(events[query_class], rng, args.min_event_s, args.max_event_s)
    if query_source is None:
        rejections["query class has no usable instance"] += 1
        return None
    query_span = wav_duration(query_source)

    # Donors are only needed by the two replacement conditions. Screening runs
    # original and silence, so demanding a donor there rejects items for a
    # resource nothing in that run consumes.
    needs_donors = bool(
        {"target_replaced_diff", "target_replaced_same"} & set(args.conditions)
    )
    same_donor = diff_donor = None
    if needs_donors:
        same_donor = _pick_donor(events[query_class], query_source, query_span, rng)
        diff_donor = _pick_donor(events[absent_class], None, query_span, rng)
        if same_donor is None or diff_donor is None:
            rejections["no duration-matched donor"] += 1
            return None

    nuisance_sources = []
    for nuisance_class in nuisance_classes:
        source = _pick_instance(
            events[nuisance_class], rng, args.min_event_s, args.max_event_s
        )
        if source is None:
            rejections["nuisance class has no usable instance"] += 1
            return None
        nuisance_sources.append((nuisance_class, source, wav_duration(source)))

    sample_rate = int(sf.info(str(backgrounds[0])).samplerate)
    spans = plan_placements(
        int(args.duration * sample_rate),
        sample_rate,
        min(query_span, args.max_event_s),
        [min(d, args.max_event_s) for _, _, d in nuisance_sources],
        rng,
        duration_s=args.duration,
    )
    if not spans:
        rejections["placement failed"] += 1
        return None

    background = rng.choice(backgrounds)
    plan = [
        {
            "role": "query",
            "label": query_class,
            "path": query_source,
            "start_s": spans[0][0],
            "duration_s": spans[0][1] - spans[0][0],
            "snr_db": args.snr_db,
            "span_s": query_span,
        }
    ]
    for (nuisance_class, source, _), span in zip(nuisance_sources, spans[1:]):
        plan.append(
            {
                "role": "nuisance",
                "label": nuisance_class,
                "path": source,
                "start_s": span[0],
                "duration_s": span[1] - span[0],
                "snr_db": args.snr_db,
                "span_s": wav_duration(source),
            }
        )

    item_id = f"{args.split}-{index:05d}"
    base_audio = args.audio_dir / f"{item_id}__base.wav"
    base_jams = args.jams_dir / f"{item_id}.jams"
    scape = render_soundscape(
        base_audio,
        base_jams,
        args.foreground_dir,
        args.background_dir,
        plan,
        {"label": background.parent.name, "path": background},
        duration_s=args.duration,
        seed=args.seed,
    )
    assert_placement_disjoint(scape)

    waveform, sample_rate = sf.read(base_audio, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if float(np.abs(waveform).max()) >= 0.999:
        rejections["rendered audio clips"] += 1
        return None
    if sample_rate != scape.sample_rate:
        rejections["sample rate mismatch"] += 1
        return None

    query_samples = int(round(scape.query.duration_s * sample_rate))
    # Only the background_removed condition needs an event-free window. Checking
    # unconditionally rejects crowded clips even when nothing in the run will use
    # that window, and because crowding is exactly what the polyphony knob
    # varies, it biases the knob: the items that survive 4 nuisance events are
    # systematically the least crowded ones.
    if "background_removed" in args.conditions:
        try:
            find_free_window(
                scape.events,
                len(waveform),
                sample_rate,
                query_samples,
                random.Random(args.seed + index),
            )
        except InstrumentError:
            rejections["no event-free control window"] += 1
            return None

    donor_waveforms: dict[str, np.ndarray] = {}
    donor_spans: dict[str, float] = {}
    if needs_donors:
        donor_waveforms = {
            "target_replaced_diff": _read_donor(diff_donor, sample_rate),
            "target_replaced_same": _read_donor(same_donor, sample_rate),
        }
        donor_spans = {
            "target_replaced_diff": wav_duration(diff_donor),
            "target_replaced_same": wav_duration(same_donor),
        }

    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        edit_rng = random.Random(args.seed * 1000 + index)
        try:
            span = resolve_condition_span(scape, condition, len(waveform), edit_rng)
            edited = apply_condition(
                waveform,
                scape,
                condition,
                donor_waveform=donor_waveforms.get(condition),
                donor_span_samples=(
                    int(round(donor_spans[condition] * sample_rate))
                    if condition in donor_spans
                    else None
                ),
                span=span,
                rng=edit_rng,
            )
        except InstrumentError as exc:
            rejections[f"condition {condition}: {exc}"[:110]] += 1
            return None
        # Every condition, including the original, is written through the same
        # PCM_16 path. scaper writes the base as 32-bit, so pointing `original`
        # at scaper's file would leave the reference and the edits in different
        # formats. `sham` is a no-op only if it is bit-identical to `original`,
        # and writing both from the same array is how that is guaranteed.
        audio_path = args.audio_dir / f"{item_id}__{condition}.wav"
        sf.write(audio_path, np.clip(edited, -1.0, 1.0), sample_rate, subtype="PCM_16")

        truth = _truth(condition, query_class, absent_class)
        edited_span_s = (
            None
            if span is None
            else [round(span[0] / sample_rate, 4), round(span[1] / sample_rate, 4)]
        )
        for order in (0, 1):
            options = (
                {"A": query_class, "B": absent_class}
                if order == 0
                else {"A": absent_class, "B": query_class}
            )
            rows.append(
                _row(
                    item_id=item_id,
                    scape=scape,
                    condition=condition,
                    audio=audio_path,
                    options=options,
                    order=order,
                    truth=truth,
                    args=args,
                    query_class=query_class,
                    absent_class=absent_class,
                    nuisance_classes=nuisance_classes,
                    background=background,
                    edited_span_s=edited_span_s,
                )
            )
    # scaper writes the base as 32-bit and every condition is re-rendered from
    # it, including `original`, so the base file is dead weight: 640 KB per item
    # of unused audio. The JAMS stays as the provenance record.
    if not args.keep_base:
        base_audio.unlink(missing_ok=True)
    return rows


def _truth(condition: str, query_class: str, absent_class: str) -> str:
    """Which option is correct after the edit.

    ``silence`` has no unique answer; it keeps the original label so the
    condition measures global audio dependence rather than asserting a truth,
    matching how the withdrawn MVP treated it.
    """

    if condition == "target_replaced_diff":
        return absent_class
    return query_class


def _row(*, item_id, scape, condition, audio, options, order, truth, args,
         query_class, absent_class, nuisance_classes, background,
         edited_span_s=None) -> dict[str, Any]:
    query = scape.query
    return {
        "pair_id": f"{item_id}__o{order}",
        "source_pair_id": item_id,
        "surface_order": order,
        "condition": condition,
        "intervention_role": ROLES[condition],
        "audio": str(audio),
        "question": QUESTION.format(a=options["A"], b=options["B"]),
        "ground_truth": "A" if options["A"] == truth else "B",
        "option_events": options,
        "evidence": [
            {"event": query.label, "start": query.start_s, "end": query.end_s}
        ],
        "edited_span_s": edited_span_s,
        "query_class": query_class,
        "absent_class": absent_class,
        "nuisance_classes": nuisance_classes,
        "query_span_s": round(query.duration_s, 4),
        "query_start_s": round(query.start_s, 4),
        "snr_db": query.snr_db,
        "background": background.name,
        "background_class": background.parent.name,
        "split": args.split,
        "seed": args.seed,
        "n_nuisance": len(nuisance_classes),
        "source_audio_key": f"synth/{item_id}",
    }


def _pick_instance(paths, rng, min_s, max_s):
    candidates = [p for p in paths if min_s <= wav_duration(p) <= max_s]
    return rng.choice(candidates) if candidates else None


def _pick_donor(paths, exclude, target_span, rng):
    candidates = [
        p
        for p in paths
        if p != exclude
        and wav_duration(p) > 0
        and abs(wav_duration(p) - target_span) / target_span <= ALLOWED_DONOR_MISMATCH
    ]
    return rng.choice(candidates) if candidates else None


def _read_donor(path, target_sample_rate: int) -> np.ndarray:
    """Read a donor at the soundscape's rate, so its samples match its span."""

    waveform, _ = read_audio_mono(path, target_sample_rate)
    return waveform


if __name__ == "__main__":
    main()
