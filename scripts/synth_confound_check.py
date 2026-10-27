#!/usr/bin/env python3
"""Test whether the headline contrast survives matching on acoustic change.

The matched pair in the design is ``target_replaced_diff`` against
``target_replaced_same``: the same sample span, rewritten by the same operation
through the same crossfade, differing only in whether the replacement preserves
the queried identity. The obvious objection is that a different-class
replacement is simply a *bigger* acoustic change, so a lower answer-change rate
for the same-class edit would be trivial.

This script answers that objection rather than asserting it away. It measures a
log-mel spectral distance between the original span content and the edited span
for both conditions, then recomputes the contrast on the subset of items where
the two distances are matched. If the effect is an artefact of distance, it
disappears inside the matched band.

Reported per model:

* the distance distributions, so the size of the imbalance is visible;
* the contrast on the matched subset, with a clip-level bootstrap interval;
* the contrast as a function of distance quintile, so a residual trend shows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

MEL_BANDS = 40
N_FFT = 512
HOP = 160
MIN_SPAN_SAMPLES = 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_mel(x: np.ndarray, sample_rate: int) -> np.ndarray:
    _, _, spectrum = stft(
        x, fs=sample_rate, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None
    )
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)

    def hz_to_mel(value: float) -> float:
        return 2595.0 * np.log10(1.0 + value / 700.0)

    def mel_to_hz(value: float) -> float:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(50), hz_to_mel(sample_rate / 2 - 100), MEL_BANDS + 2))
    filters = np.zeros((MEL_BANDS, len(frequencies)))
    for index in range(MEL_BANDS):
        low, centre, high = edges[index], edges[index + 1], edges[index + 2]
        filters[index] = np.clip(
            np.minimum(
                (frequencies - low) / max(centre - low, 1e-9),
                (high - frequencies) / max(high - centre, 1e-9),
            ),
            0,
            None,
        )
    return np.log(filters @ power + 1e-10)


def distance(first: np.ndarray, second: np.ndarray) -> float | None:
    frames = min(first.shape[1], second.shape[1])
    if frames < 3:
        return None
    return float(
        np.mean(np.sqrt(np.mean((first[:, :frames] - second[:, :frames]) ** 2, axis=0)))
    )


def change_rates(
    items: list[str], changed: dict, aggregation: str
) -> tuple[float, float, int]:
    """Compare paired edits, either per option-order view or by item-level OR."""
    conditions = ("target_replaced_diff", "target_replaced_same")
    if aggregation == "view":
        # The headline table uses both orders, paired within each soundscape.
        if any(len(changed[item][condition]) != 2 for item in items for condition in conditions):
            raise ValueError("view-level comparison requires both option orders per item and condition")
        diff, same = (
            [flag for item in items for flag in changed[item][condition]]
            for condition in conditions
        )
    elif aggregation == "any-order":
        diff, same = (
            [any(changed[item][condition]) for item in items]
            for condition in conditions
        )
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    return (
        sum(diff) / len(diff) if diff else float("nan"),
        sum(same) / len(same) if same else float("nan"),
        len(items),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--band", type=float, default=0.15, help="relative distance match tolerance")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--aggregation", choices=("any-order", "view"), default="any-order",
        help="item-level OR (legacy) or Table 1's per-option-order view rate",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help=(
            "Model key to record for each predictions file, in the same order. "
            "Defaults to the input filename stem, which embeds the run name "
            "('r2v2_qwen3b') rather than the model identity; consolidate_evidence.py "
            "keys this artifact by model identity, so the committed artifact must "
            "be generated with explicit keys."
        ),
    )
    args = parser.parse_args()

    if args.keys and len(args.keys) != len(args.predictions):
        raise SystemExit(
            f"--keys has {len(args.keys)} entries for {len(args.predictions)} inputs"
        )
    keys = args.keys or [p.stem for p in args.predictions]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"--keys must be unique, got {keys}")

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_item[row["source_pair_id"]][row["condition"]] = row

    cache: dict[str, tuple[np.ndarray, int]] = {}

    def audio(name: str) -> tuple[np.ndarray, int]:
        if name not in cache:
            waveform, rate = sf.read(
                str(args.audio_root / Path(name).name), dtype="float32", always_2d=False
            )
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            cache[name] = (waveform.astype(np.float32), rate)
        return cache[name]

    distances: dict[str, dict[str, float]] = defaultdict(dict)
    for item, conditions in by_item.items():
        original = conditions.get("original")
        if original is None:
            continue
        evidence = original["evidence"][0]
        waveform, rate = audio(original["audio"])
        start, end = int(evidence["start"] * rate), int(evidence["end"] * rate)
        if end - start < MIN_SPAN_SAMPLES:
            continue
        reference = log_mel(waveform[start:end], rate)
        for condition in ("target_replaced_diff", "target_replaced_same"):
            entry = conditions.get(condition)
            span = entry.get("edited_span_s") if entry else None
            if not span:
                continue
            edited, _ = audio(entry["audio"])
            low, high = int(span[0] * rate), int(span[1] * rate)
            if high - low < MIN_SPAN_SAMPLES:
                continue
            value = distance(reference, log_mel(edited[low:high], rate))
            if value is not None:
                distances[item][condition] = value

    report: dict = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "audio_root": str(args.audio_root),
        "band": args.band,
        "items_with_distances": len(distances),
        # Provenance: without the input predictions and their hashes this artifact
        # cannot be tied to the grid it describes, which is why it was previously
        # excluded from the manuscript's evidence chain.
        "predictions": [
            {"key": key, "path": str(path), "sha256": sha256(path)}
            for key, path in zip(keys, args.predictions)
        ],
        "bootstrap": {"samples": args.samples, "seed": args.seed, "unit": "item (source pair)"},
        "models": {},
    }
    if args.aggregation == "view":
        report["aggregation"] = "per-option-order view, bootstrap resampled by soundscape"

    for path, key in zip(args.predictions, keys):
        predictions = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        by_view: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in predictions:
            if row.get("parsed_choice") and "error" not in row:
                by_view[row["pair_id"]][row["condition"]] = row

        # One record per item per condition: a view contributes only if the
        # model changed its answer, and both orders are pooled by item.
        changed: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for pair_id, conditions in by_view.items():
            original = conditions.get("original")
            if original is None:
                continue
            item = str(original.get("source_pair_id", pair_id))
            base = str(original["parsed_choice"])
            for condition in ("target_replaced_diff", "target_replaced_same"):
                entry = conditions.get(condition)
                if entry is not None:
                    changed[item][condition].append(str(entry["parsed_choice"]) != base)

        paired = [
            item
            for item in distances
            if item in changed
            and changed[item].get("target_replaced_diff")
            and changed[item].get("target_replaced_same")
        ]

        def rates(items: list[str]) -> tuple[float, float, int]:
            return change_rates(items, changed, args.aggregation)

        all_diff, all_same, n_all = rates(paired)
        matched = [
            item
            for item in paired
            if abs(
                distances[item]["target_replaced_diff"] - distances[item]["target_replaced_same"]
            )
            <= args.band
            * max(
                distances[item]["target_replaced_diff"],
                distances[item]["target_replaced_same"],
            )
        ]
        matched_diff, matched_same, n_matched = rates(matched)

        order = sorted(paired, key=lambda item: distances[item]["target_replaced_diff"])
        quintiles = []
        size = max(1, len(order) // 5)
        for index in range(5):
            chunk = order[index * size : (index + 1) * size] if index < 4 else order[4 * size :]
            if not chunk:
                continue
            diff_rate, same_rate, count = rates(chunk)
            quintiles.append(
                {
                    "quintile": index + 1,
                    "n": count,
                    "median_distance_diff": float(
                        np.median([distances[i]["target_replaced_diff"] for i in chunk])
                    ),
                    "A_diff": diff_rate,
                    "A_same": same_rate,
                    "selectivity": diff_rate - same_rate,
                }
            )

        rng = random.Random(args.seed)
        draws = []
        for _ in range(args.samples):
            picks = [matched[rng.randrange(len(matched))] for _ in range(len(matched))] if matched else []
            if not picks:
                break
            diff_rate, same_rate, _ = rates(picks)
            draws.append(diff_rate - same_rate)
        draws.sort()

        report["models"][key] = {
            "items": n_all,
            "A_diff": all_diff,
            "A_same": all_same,
            "selectivity": all_diff - all_same,
            "median_distance_diff": float(
                np.median([distances[i]["target_replaced_diff"] for i in paired])
            ),
            "median_distance_same": float(
                np.median([distances[i]["target_replaced_same"] for i in paired])
            ),
            "matched_items": n_matched,
            "matched_A_diff": matched_diff,
            "matched_A_same": matched_same,
            "matched_selectivity": matched_diff - matched_same,
            "matched_bootstrap_95": (
                [_quantile(draws, 0.025), _quantile(draws, 0.975)] if draws else None
            ),
            "by_distance_quintile": quintiles,
        }
        if args.aggregation == "view":
            report["models"][key]["views"] = 2 * n_all
            report["models"][key]["matched_views"] = 2 * n_matched

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


def _quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
    main()
