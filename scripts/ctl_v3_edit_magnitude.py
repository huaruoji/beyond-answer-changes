"""v3 probe step 2 (free, no GPU): is the same-class swap really a smaller edit?

Reading (B) explains MIG > 0 by claiming "a different-class donor alters the span more
than a same-class one". That premise is testable on the audio that already exists:
for every sample60_v2 item x cell, measure the span-level distance between the
`original` recording and the `diff` / `same` / `sham` recordings.

Three distances per (item, cell, arm), all free and offline:

* ``delta_rms_db``      — 20log10(||edit - original|| / ||original||) over the query span;
* ``change_audibility_db`` — the project's own AUD component
  (``src.audibility.audibility_margin_db``) applied to the difference signal against the
  original span as masker: how far the *change* sticks out of the content it replaces;
* ``lsd_db``            — log-spectral distance between original and edited span.

Then it joins the 3B/7B predictions and reports, per cell and pooled, the change rate
in the items where the same-class edit is at least as large as the different-class edit
(the subset on which reading (B) makes the opposite prediction to reading (A)).

Writes ``outputs/ctl/v3_probe/edit_magnitude.json``. Reads only; modifies nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audibility import audibility_margin_db  # noqa: E402

ARMS = ("diff", "same", "sham")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def lsd_db(original: np.ndarray, edited: np.ndarray, sample_rate: int, n_fft: int = 512, hop: int = 128) -> float:
    """Mean log-spectral distance over the span (magnitude spectra, Hann window)."""

    n = min(original.size, edited.size)
    window = np.hanning(n_fft).astype(np.float64)
    frames = list(range(0, max(1, n - n_fft + 1), hop))
    if not frames:
        return float("nan")
    diffs = []
    for start in frames:
        a = np.fft.rfft(original[start : start + n_fft] * window)
        b = np.fft.rfft(edited[start : start + n_fft] * window)
        la = 20.0 * np.log10(np.maximum(np.abs(a), 1e-10))
        lb = 20.0 * np.log10(np.maximum(np.abs(b), 1e-10))
        diffs.append(np.sqrt(np.mean((la - lb) ** 2)))
    return float(np.mean(diffs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/ctl/sample60_v2"))
    parser.add_argument("--models", default="3b=outputs/ctl/sample60_v2/pred_v2_3b.jsonl,"
                                          "7b=outputs/ctl/sample60_v2/pred_v2_7b.jsonl")
    parser.add_argument("--output", type=Path, default=Path("outputs/ctl/v3_probe/edit_magnitude.json"))
    args = parser.parse_args()

    manifest = load_jsonl(args.run_dir / "manifest.jsonl")
    by_pair = {row["pair_id"]: row for row in manifest}
    originals = {row["source_pair_id"]: row for row in manifest if row["condition"] == "original"}

    per_item: dict[str, dict] = {}
    for source_pair_id, original_row in sorted(originals.items()):
        sample_rate = int(original_row["sample_rate"])
        start = int(round(original_row["query_start_s"] * sample_rate))
        end = int(round(original_row["query_end_s"] * sample_rate))
        original_audio, _ = sf.read(original_row["audio"], dtype="float64", always_2d=False)
        original_span = original_audio[start:end]
        entry: dict[str, object] = {
            "cell": "__".join(source_pair_id.split("__")[-2:]),
            "item_id": original_row["item_id"],
            "reference_rms": float(np.sqrt(np.mean(original_span**2))),
            "arms": {},
        }
        for arm in ARMS:
            row = by_pair[f"{source_pair_id}__{arm}"]
            edited_audio, _ = sf.read(row["audio"], dtype="float64", always_2d=False)
            edited_span = edited_audio[start:end]
            delta = edited_span - original_span
            delta_rms = float(np.sqrt(np.mean(delta**2)))
            ref_rms = float(np.sqrt(np.mean(original_span**2)))
            entry["arms"][arm] = {
                "delta_rms_db": float(20.0 * np.log10(max(delta_rms, 1e-12) / max(ref_rms, 1e-12))),
                "change_audibility_db": float(
                    audibility_margin_db(delta.astype(np.float32), original_span.astype(np.float32), sample_rate)
                ),
                "lsd_db": lsd_db(original_span, edited_span, sample_rate),
                "edited_span_rms_db_re_original": float(
                    20.0 * np.log10(max(np.sqrt(np.mean(edited_span**2)), 1e-12) / max(ref_rms, 1e-12))
                ),
            }
        per_item[source_pair_id] = entry

    # join predictions
    models = dict(part.split("=", 1) for part in args.models.split(","))
    predictions: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in models.items():
        preds = load_jsonl(Path(path))
        table: dict[str, dict[str, str]] = defaultdict(dict)
        for row in preds:
            if row.get("parsed_choice") is None:
                continue
            table[row["source_pair_id"]][row["condition"]] = row["parsed_choice"]
        predictions[name] = table

    summary: dict[str, object] = {"cells": {}, "models": {}}
    for source_pair_id, entry in per_item.items():
        cell = entry["cell"]
        block = summary["cells"].setdefault(cell, {"n": 0, "same_minus_diff_lsd_db": [], "same_minus_diff_aud_db": []})
        block["n"] += 1
        block["same_minus_diff_lsd_db"].append(entry["arms"]["same"]["lsd_db"] - entry["arms"]["diff"]["lsd_db"])
        block["same_minus_diff_aud_db"].append(
            entry["arms"]["same"]["change_audibility_db"] - entry["arms"]["diff"]["change_audibility_db"]
        )
    for cell, block in summary["cells"].items():
        lsd = np.asarray(block.pop("same_minus_diff_lsd_db"))
        aud = np.asarray(block.pop("same_minus_diff_aud_db"))
        block["lsd_same_minus_diff_median_db"] = float(np.median(lsd))
        block["lsd_same_ge_diff_count"] = int(np.sum(lsd >= 0))
        block["aud_same_minus_diff_median_db"] = float(np.median(aud))
        block["aud_same_ge_diff_count"] = int(np.sum(aud >= 0))

    for name, table in predictions.items():
        model_block: dict[str, object] = {"cells": {}}
        changed_by_item: dict[str, dict[str, bool]] = {}
        for source_pair_id, answers in table.items():
            if "original" not in answers:
                continue
            changed_by_item[source_pair_id] = {
                arm: answers.get(arm) != answers["original"] for arm in ARMS if arm in answers
            }
        for cell in sorted(summary["cells"]):
            items = [sid for sid in per_item if per_item[sid]["cell"] == cell and sid in changed_by_item]
            if not items:
                continue
            lsd_same = np.array([per_item[sid]["arms"]["same"]["lsd_db"] for sid in items])
            lsd_diff = np.array([per_item[sid]["arms"]["diff"]["lsd_db"] for sid in items])
            aud_same = np.array([per_item[sid]["arms"]["same"]["change_audibility_db"] for sid in items])
            aud_diff = np.array([per_item[sid]["arms"]["diff"]["change_audibility_db"] for sid in items])
            ch_same = np.array([changed_by_item[sid]["same"] for sid in items])
            ch_diff = np.array([changed_by_item[sid]["diff"] for sid in items])
            reversed_lsd = lsd_same >= lsd_diff
            reversed_aud = aud_same >= aud_diff
            cell_block = {
                "n_items": len(items),
                "P_change_diff": float(ch_diff.mean()),
                "P_change_same": float(ch_same.mean()),
                "MIG": float(ch_diff.mean() - ch_same.mean()),
                "subset_same_lsd_ge_diff": {
                    "n": int(reversed_lsd.sum()),
                    "P_change_diff": float(ch_diff[reversed_lsd].mean()) if reversed_lsd.any() else None,
                    "P_change_same": float(ch_same[reversed_lsd].mean()) if reversed_lsd.any() else None,
                },
                "subset_same_aud_ge_diff": {
                    "n": int(reversed_aud.sum()),
                    "P_change_diff": float(ch_diff[reversed_aud].mean()) if reversed_aud.any() else None,
                    "P_change_same": float(ch_same[reversed_aud].mean()) if reversed_aud.any() else None,
                },
                "lsd_terciles": {},
            }
            # within-arm change rate by LSD tercile of the pooled diff+same edits
            pooled = np.concatenate([lsd_diff, lsd_same])
            edges = np.percentile(pooled, [33.333, 66.667])
            for label, mask in (
                ("low", pooled <= edges[0]),
                ("mid", (pooled > edges[0]) & (pooled <= edges[1])),
                ("high", pooled > edges[1]),
            ):
                mask_diff, mask_same = mask[: len(items)], mask[len(items) :]
                cell_block["lsd_terciles"][label] = {
                    "lsd_range_db": [float(pooled[mask].min()), float(pooled[mask].max())] if mask.any() else None,
                    "P_change_diff": float(ch_diff[mask_diff].mean()) if mask_diff.any() else None,
                    "P_change_same": float(ch_same[mask_same].mean()) if mask_same.any() else None,
                    "n_diff": int(mask_diff.sum()),
                    "n_same": int(mask_same.sum()),
                }
            model_block["cells"][cell] = cell_block
        summary["models"][name] = model_block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"source_run": str(args.run_dir), "per_item": per_item, "summary": summary}, indent=2),
        encoding="utf-8",
    )

    print(f"{'cell':34} {'LSD same-diff med':>18} {'same>=diff':>10} {'AUD same-diff med':>18} {'same>=diff':>10}")
    for cell, block in summary["cells"].items():
        print(f"{cell:34} {block['lsd_same_minus_diff_median_db']:+18.2f} {block['lsd_same_ge_diff_count']:10d} "
              f"{block['aud_same_minus_diff_median_db']:+18.2f} {block['aud_same_ge_diff_count']:10d}")
    for name, model_block in summary["models"].items():
        print(f"\n=== model {name}")
        for cell, block in model_block["cells"].items():
            print(f"{cell:34} n={block['n_items']} P(diff)={block['P_change_diff']:.3f} "
                  f"P(same)={block['P_change_same']:.3f}")
            for key in ("subset_same_lsd_ge_diff", "subset_same_aud_ge_diff"):
                sub = block[key]
                print(f"    {key:28} n={sub['n']:3d} P(diff)={sub['P_change_diff']} P(same)={sub['P_change_same']}")
            for label, t in block["lsd_terciles"].items():
                print(f"    lsd {label:5} diff n={t['n_diff']:3d} P={t['P_change_diff']} | "
                      f"same n={t['n_same']:3d} P={t['P_change_same']}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
