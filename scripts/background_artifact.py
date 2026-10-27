#!/usr/bin/env python3
"""Quantify the replacement's noise-floor artefact, and test whether models use it.

WHY THIS EXISTS
---------------
A listener (2026-09-21, confirm split) reported that the substituted sound has no
noise floor: "替换的异类声音没有底噪了，和环境音区别有点明显，接缝感比较强".
That is correct and it is a real property of the instrument, not a perception
error. ``crossfade_replace`` in ``src/synth.py`` overwrites the queried span with
the donor's own samples, so the background present in the original span is
removed along with the queried event. The replacement is donor-only, not
donor-mixed-into-background.

Two questions follow, and this script answers both from the data:

1. *How large is the artefact?* Measured as the drop in the span's noise floor
   (10th-percentile frame RMS), in dB, relative to the original span.

2. *Do the models exploit it?* This is the one that decides whether the paper's
   contrast survives. The artefact is **symmetric across the matched arms**: the
   same-class replacement removes the background exactly as the different-class
   one does. So a model that reasoned "the noise floor vanished, therefore an edit
   happened, therefore the queried class is gone" would answer *correctly* on
   ``diff`` and *incorrectly* on ``same``. The script therefore reads
   ``accuracy(same)`` and, decisively, whether a larger artefact predicts more
   errors on the ``same`` arm. If it does not, the cue is not being used.

USAGE
    python scripts/background_artifact.py \
        --manifest data/processed/synth/confirm.jsonl \
        --audio-root data/processed/synth/confirm_audio \
        --predictions outputs/round2/r2_qwen3b.jsonl \
        --output outputs/round2/background_artifact.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FRAME = 320  # 20 ms at 16 kHz
FLOOR_PERCENTILE = 10


def noise_floor_db(x: np.ndarray, frame: int = FRAME) -> float:
    """10th-percentile frame RMS in dBFS.

    The span contains an event plus background. The quietest frames are the
    background between onsets, so a low percentile reads the noise floor rather
    than the event's own level.
    """

    n = len(x) // frame
    if n == 0:
        return float("nan")
    frames = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n)]
    )
    return float(20 * np.log10(np.percentile(frames, FLOOR_PERCENTILE) + 1e-12))


def load_mono(path: Path) -> np.ndarray:
    x, _ = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--split", default="confirm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("split") == args.split and r.get("surface_order") == 0]
    items: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        items[row["source_pair_id"]][row["condition"]] = row

    per_item: dict[str, dict[str, float]] = {}
    for pid, conds in sorted(items.items()):
        original = conds.get("original")
        if original is None:
            continue
        start = original.get("query_start_s")
        span = original.get("query_span_s")
        if start is None or span is None:
            continue
        try:
            rate = sf.info(args.audio_root / Path(original["audio"]).name).samplerate
        except Exception:
            continue
        s, e = int(start * rate), int((start + span) * rate)
        if e - s < 5 * FRAME:
            continue
        try:
            orig = load_mono(args.audio_root / Path(original["audio"]).name)
            diff = load_mono(args.audio_root / Path(conds["target_replaced_diff"]["audio"]).name)
            same = load_mono(args.audio_root / Path(conds["target_replaced_same"]["audio"]).name)
        except Exception:
            continue
        # background reference: an event-free window elsewhere in the same clip
        bg = noise_floor_db(orig[: 10 * FRAME])
        per_item[pid] = {
            "floor_original": noise_floor_db(orig[s:e]),
            "floor_diff": noise_floor_db(diff[s:e]),
            "floor_same": noise_floor_db(same[s:e]),
            "floor_background_window": bg,
            "drop_diff_db": noise_floor_db(orig[s:e]) - noise_floor_db(diff[s:e]),
            "drop_same_db": noise_floor_db(orig[s:e]) - noise_floor_db(same[s:e]),
        }

    if not per_item:
        raise SystemExit("no usable items; check the manifest and audio root")

    dd = np.array([v["drop_diff_db"] for v in per_item.values()])
    ds = np.array([v["drop_same_db"] for v in per_item.values()])
    paired = dd - ds

    report: dict[str, Any] = {
        "n_items": len(per_item),
        "measure": "10th-percentile frame RMS of the queried span, dBFS",
        "artifact": {
            "drop_diff_db": {
                "median": float(np.median(dd)),
                "q25": float(np.percentile(dd, 25)),
                "q75": float(np.percentile(dd, 75)),
                "share_over_10db": float((dd > 10).mean()),
            },
            "drop_same_db": {
                "median": float(np.median(ds)),
                "q25": float(np.percentile(ds, 25)),
                "q75": float(np.percentile(ds, 75)),
                "share_over_10db": float((ds > 10).mean()),
            },
            "paired_diff_minus_same_db": {
                "median": float(np.median(paired)),
                "mean": float(paired.mean()),
                "sd": float(paired.std()),
            },
            "symmetric_across_arms": bool(abs(float(np.median(paired))) < 3.0),
            "interpretation": (
                "A positive drop means the span's noise floor fell, i.e. the "
                "replacement carries no background. The artefact is present on "
                "both matched arms, so it cannot by itself separate them."
            ),
        },
    }

    # Does a larger artefact predict more errors on the same arm?
    exploitation: dict[str, Any] = {}
    for path in args.predictions:
        pred: dict[str, dict[str, Any]] = defaultdict(dict)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == args.split and row.get("surface_order") == 0:
                pred[row["source_pair_id"]][row["condition"]] = row

        entry: dict[str, Any] = {}
        for arm, key in (("target_replaced_diff", "drop_diff_db"),
                         ("target_replaced_same", "drop_same_db")):
            xs, ys = [], []
            for pid, values in per_item.items():
                conds = pred.get(pid)
                if not conds or arm not in conds:
                    continue
                xs.append(values[key])
                ys.append(1.0 if conds[arm].get("correct") else 0.0)
            if len(xs) < 30:
                continue
            xs_a, ys_a = np.array(xs), np.array(ys)
            q = np.percentile(xs_a, [33, 67])
            small, large = xs_a <= q[0], xs_a >= q[1]
            entry[arm] = {
                "n": int(len(xs_a)),
                "accuracy": float(ys_a.mean()),
                "r_artifact_vs_correct": float(np.corrcoef(xs_a, ys_a)[0, 1]),
                "accuracy_small_artifact": float(ys_a[small].mean()),
                "accuracy_large_artifact": float(ys_a[large].mean()),
                "delta_pt_large_minus_small": float(
                    100 * (ys_a[large].mean() - ys_a[small].mean())
                ),
            }
        exploitation[path.stem] = entry
    report["exploitation_test"] = exploitation

    print(f"n_items = {report['n_items']}")
    art = report["artifact"]
    print(f"  noise-floor drop, diff arm: median {art['drop_diff_db']['median']:+.2f} dB "
          f"({art['drop_diff_db']['share_over_10db']*100:.0f}% over 10 dB)")
    print(f"  noise-floor drop, same arm: median {art['drop_same_db']['median']:+.2f} dB "
          f"({art['drop_same_db']['share_over_10db']*100:.0f}% over 10 dB)")
    print(f"  paired diff-same: median {art['paired_diff_minus_same_db']['median']:+.2f} dB "
          f"-> symmetric: {art['symmetric_across_arms']}")
    for model, entry in exploitation.items():
        print(f"  {model}:")
        for arm, block in entry.items():
            print(f"    {arm:22s} acc={block['accuracy']:.4f}  "
                  f"r(artifact,correct)={block['r_artifact_vs_correct']:+.3f}  "
                  f"large-vs-small = {block['delta_pt_large_minus_small']:+.1f} pt")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
