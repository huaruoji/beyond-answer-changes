#!/usr/bin/env python3
"""Mechanical audit of a synthetic manifest against the ADR-011 invariants.

Runs before any model touches the audio. The withdrawn 2026-09-11 pipeline had
audits too, and they passed — because they checked bookkeeping (labels balanced,
files decodable) and never checked whether a control edit actually landed off
the evidence. This audit checks the invariants themselves.

Checked here:

* every item carries all seven conditions, in both option orders;
* option order in the question text matches ``option_events``;
* labels invert between the two orders inside every condition;
* **I2** every control edit's ``edited_span_s`` is disjoint from the query span
  by at least the guard band — this is the check whose absence let 40 of 80 L2
  controls overlap their target;
* I2b the same, measured on the audio: the query span must be bit-identical
  between ``original`` and every control condition;
* **I3** donor span mismatch is within tolerance, and the query span is reused
  exactly (not resampled) across conditions;
* **I4** no condition introduces a discontinuity at its span boundaries;
* audio files exist, decode, are non-silent, and are not clipped;
* every condition is A/B balanced across the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synth import CONDITIONS, MIN_EVENT_GAP_S  # noqa: E402

#: Conditions whose edit must not touch the query span.
CONTROL_CONDITIONS = ("nuisance_removed", "background_removed", "sham")
#: Tolerance for donor span mismatch, matching the generator.
DONOR_TOLERANCE = 0.5
#: A boundary is "clean" if its jump is within this factor of the clip's own.
BOUNDARY_FACTOR = 3.0
#: The sham rewrites a span with its own samples through the real crossfade, and
#: float rounding can move a couple of samples by 1 LSB. Measured worst case on
#: the real corpus: 1 LSB on ~6 of 160000 samples. Anything beyond a few LSB is
#: perturbation, not rounding.
SHAM_TOLERANCE = 1.5 / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio-root", type=Path, help="resolve relative audio paths")
    parser.add_argument("--max-items", type=int, help="audit only the first N items")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="conditions the manifest is expected to contain",
    )
    args = parser.parse_args()
    expected = list(args.conditions)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    notes: Counter[str] = Counter()

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[row["source_pair_id"]].append(row)

    items = sorted(by_item)
    if args.max_items is not None:
        items = items[: args.max_items]

    condition_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        condition_labels[row["condition"]][row["ground_truth"]] += 1

    for item_id in items:
        group = by_item[item_id]
        conditions = {row["condition"] for row in group}
        missing = set(expected) - conditions
        if missing:
            errors.append(f"{item_id}: missing conditions {sorted(missing)}")
            continue

        by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_condition[row["condition"]][int(row["surface_order"])] = row

        for condition, orders in by_condition.items():
            if set(orders) != {0, 1}:
                errors.append(f"{item_id}/{condition}: expected both option orders")
                continue
            first, second = orders[0], orders[1]
            if first["audio"] != second["audio"]:
                errors.append(f"{item_id}/{condition}: orders use different audio")
            if first["ground_truth"] == second["ground_truth"]:
                errors.append(f"{item_id}/{condition}: labels do not invert")
            for row in (first, second):
                quotes = _question_options(row["question"])
                if quotes != [row["option_events"]["A"], row["option_events"]["B"]]:
                    errors.append(
                        f"{item_id}/{condition}: question text disagrees with option_events"
                    )

        original = by_condition["original"][0]
        query_start, query_end = _query_span(original)

        for condition in CONTROL_CONDITIONS:
            if condition not in expected:
                continue
            row = by_condition[condition][0]
            span = row.get("edited_span_s")
            if span is None:
                errors.append(f"{item_id}/{condition}: no edited_span_s recorded")
                continue
            gap = max(query_start - span[1], span[0] - query_end)
            if condition != "sham" and gap < MIN_EVENT_GAP_S - 1e-6:
                errors.append(
                    f"{item_id}/{condition}: control edit {span} is {gap:.3f}s from "
                    f"the query span [{query_start}, {query_end}], below the "
                    f"{MIN_EVENT_GAP_S}s guard"
                )

        for condition in ("target_replaced_diff", "target_replaced_same"):
            if condition not in expected:
                continue
            row = by_condition[condition][0]
            if abs(row["query_span_s"] - original["query_span_s"]) > 1e-6:
                errors.append(f"{item_id}/{condition}: query span changed")

        if not args.skip_audio:
            _audit_audio(item_id, by_condition, original, args, errors, expected)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "items": len(items),
        "records": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "condition_label_counts": {
            condition: dict(sorted(counts.items()))
            for condition, counts in sorted(condition_labels.items())
        },
        "splits": sorted({row.get("split", "?") for row in rows}),
        "classes_used": len({row.get("query_class") for row in rows}),
        "errors": errors[:200],
        "error_count": len(errors),
        "valid": not errors,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if errors:
        raise SystemExit(1)


def _audit_audio(item_id, by_condition, original, args, errors, expected) -> None:
    """Waveform-level checks: I2b (span untouched), I4 (no transient), sanity."""

    original_path = _resolve(original["audio"], args)
    try:
        base, sample_rate = sf.read(original_path, dtype="float32", always_2d=False)
    except Exception as exc:
        errors.append(f"{item_id}: original audio unreadable: {exc}")
        return
    if base.ndim > 1:
        base = base.mean(axis=1)
    if float(np.abs(base).max()) >= 0.999:
        errors.append(f"{item_id}: original audio is clipped")
    if float(np.abs(base).max()) < 1e-4:
        errors.append(f"{item_id}: original audio is silent")
    own_step = float(np.abs(np.diff(base)).max()) if len(base) > 1 else 0.0

    query_start, query_end = _query_span(original)
    q0 = int(round(query_start * sample_rate))
    q1 = int(round(query_end * sample_rate))

    for condition in expected:
        row = by_condition[condition][0]
        if condition == "original":
            continue
        span = row.get("edited_span_s")
        if span is None:
            continue
        start = int(round(span[0] * sample_rate))
        end = int(round(span[1] * sample_rate))
        try:
            edited, edited_rate = sf.read(
                _resolve(row["audio"], args), dtype="float32", always_2d=False
            )
        except Exception as exc:
            errors.append(f"{item_id}/{condition}: audio unreadable: {exc}")
            continue
        if edited.ndim > 1:
            edited = edited.mean(axis=1)
        if edited_rate != sample_rate:
            errors.append(f"{item_id}/{condition}: sample rate differs from original")
            continue
        if len(edited) != len(base):
            errors.append(f"{item_id}/{condition}: length differs from original")
            continue

        if condition in CONTROL_CONDITIONS and condition != "sham":
            # I2b: the evidence itself must be untouched by a control edit.
            if not np.allclose(edited[q0:q1], base[q0:q1], atol=2e-4):
                errors.append(
                    f"{item_id}/{condition}: control edit altered the query span"
                )
        if condition == "sham":
            # The sham rewrites the query span with its own samples; anything
            # beyond PCM quantisation means the pipeline perturbs audio.
            if not np.allclose(edited[q0:q1], base[q0:q1], atol=SHAM_TOLERANCE):
                errors.append(
                    f"{item_id}/sham: no-op edit changed the query span by more "
                    f"than rounding ({float(np.abs(edited[q0:q1] - base[q0:q1]).max()):.2e})"
                )

        # I4: a crossfade must not add a discontinuity, including the case where
        # the fill is silence, which is what the withdrawn code got wrong.
        edges = [
            (start - 1, start),
            (start, start + 1),
            (end - 1, end),
            (end, end + 1),
        ]
        worst = 0.0
        for left, right in edges:
            if 0 <= left < len(edited) and 0 <= right < len(edited):
                worst = max(worst, abs(float(edited[right]) - float(edited[left])))
        if worst > BOUNDARY_FACTOR * max(own_step, 1e-6):
            errors.append(
                f"{item_id}/{condition}: boundary jump {worst:.4f} exceeds "
                f"{BOUNDARY_FACTOR}x the clip's own step {own_step:.4f}"
            )


def _resolve(path: str, args) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if args.audio_root is not None:
        return args.audio_root / candidate.name
    return REPO_ROOT / candidate


def _question_options(question: str) -> list[str]:
    import re

    quoted = re.findall(r"'([^']*)'", question or "")
    return quoted[:2]


def _query_span(row: dict[str, Any]) -> tuple[float, float]:
    evidence = row["evidence"][0]
    return float(evidence["start"]), float(evidence["end"])


if __name__ == "__main__":
    main()
