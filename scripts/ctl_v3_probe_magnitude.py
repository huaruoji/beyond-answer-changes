"""v3 probe, third measurement: how big is each probe edit, and does the answer follow it?

Reading (B) is a threshold story: the answer flips when the edit is detected, and a
different-class donor is supposed to be a bigger edit. This script measures the edit
size of every probe arm with the same two offline distances used on run 2
(log-spectral distance and the project's AUD component applied to the difference
signal), then joins them to the probe answers. It also checks the probe's 2AFC answers
against run 2's on the two arms that are byte-identical between the two runs.

Writes ``outputs/ctl/v3_probe/probe_edit_magnitude.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audibility import audibility_margin_db  # noqa: E402

ARMS = ("original", "diff", "same", "same_loud", "diff_loud")
KINDS = ("2afc", "presq", "presa")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lsd_db(original: np.ndarray, edited: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    n = min(original.size, edited.size)
    window = np.hanning(n_fft).astype(np.float64)
    diffs = []
    for start in range(0, max(1, n - n_fft + 1), hop):
        a = np.fft.rfft(original[start : start + n_fft] * window)
        b = np.fft.rfft(edited[start : start + n_fft] * window)
        la = 20.0 * np.log10(np.maximum(np.abs(a), 1e-10))
        lb = 20.0 * np.log10(np.maximum(np.abs(b), 1e-10))
        diffs.append(np.sqrt(np.mean((la - lb) ** 2)))
    return float(np.mean(diffs)) if diffs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=Path("outputs/ctl/v3_probe/probe"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/ctl/sample60_v2"))
    parser.add_argument("--models", default="3b=outputs/ctl/v3_probe/probe_pred_3b.jsonl,"
                                          "7b=outputs/ctl/v3_probe/probe_pred_7b.jsonl")
    parser.add_argument("--output", type=Path, default=Path("outputs/ctl/v3_probe/probe_edit_magnitude.json"))
    args = parser.parse_args()

    rows = load_jsonl(args.probe_dir / "manifest.jsonl")
    by_item: dict[str, dict[str, dict]] = {}
    for row in rows:
        item = row["item_id"]
        arm = row["condition"]
        kind = row["pair_id"].rsplit("__", 1)[1]
        by_item.setdefault(item, {}).setdefault(arm, {})[kind] = row

    per_item: dict[str, dict] = {}
    for item, arms in sorted(by_item.items()):
        ref_row = arms["original"]["2afc"]
        sr = int(ref_row["sample_rate"])
        start = int(round(ref_row["query_start_s"] * sr))
        end = int(round(ref_row["query_end_s"] * sr))
        original_audio, _ = sf.read(arms["original"]["2afc"]["audio"], dtype="float64")
        reference = original_audio[start:end]
        outside_mask = np.ones(original_audio.size, dtype=bool)
        outside_mask[start:end] = False
        reference_outside_rms = float(np.sqrt(np.mean(original_audio[outside_mask] ** 2)))
        entry: dict[str, object] = {"cell": ref_row["calibration"] + "__" + ref_row["masker"], "arms": {}}
        for arm in ARMS:
            audio, _ = sf.read(arms[arm]["2afc"]["audio"], dtype="float64")
            span = audio[start:end]
            delta = span - reference
            # the two `_loud` arms were rendered with a whole-file headroom gain so the
            # +10 dB span could not clip; recover it from the untouched surround so the
            # edit size can be compared against the reference at equal file gain
            headroom = float(np.sqrt(np.mean(audio[outside_mask] ** 2))) / max(reference_outside_rms, 1e-12)
            delta_renorm = delta / max(headroom, 1e-12)
            entry["arms"][arm] = {
                "lsd_db": lsd_db(reference, span),
                "delta_rms_db": float(
                    20.0 * np.log10(max(np.sqrt(np.mean(delta**2)), 1e-12) / max(np.sqrt(np.mean(reference**2)), 1e-12))
                ),
                "delta_rms_db_gain_normalised": float(
                    20.0
                    * np.log10(
                        max(np.sqrt(np.mean(delta_renorm**2)), 1e-12) / max(np.sqrt(np.mean(reference**2)), 1e-12)
                    )
                ),
                "headroom_gain_estimate": headroom,
                "change_audibility_db": float(
                    audibility_margin_db(delta.astype(np.float32), reference.astype(np.float32), sr)
                ),
                "change_audibility_db_gain_normalised": float(
                    audibility_margin_db(delta_renorm.astype(np.float32), reference.astype(np.float32), sr)
                ),
                "span_rms_db_re_surround": float(
                    20.0
                    * np.log10(
                        max(np.sqrt(np.mean(span**2)), 1e-12)
                        / max(np.sqrt(np.mean(audio[outside_mask] ** 2)), 1e-12)
                    )
                ),
                "peak": float(np.abs(audio).max()),
            }
        per_item[item] = entry

    models = dict(part.split("=", 1) for part in args.models.split(","))
    answers: dict[str, dict[tuple[str, str, str], str]] = {}
    run2_agreement: dict[str, dict] = {}
    for name, path in models.items():
        table: dict[tuple[str, str, str], str] = {}
        for row in load_jsonl(Path(path)):
            if row.get("parsed_choice") is None:
                continue
            item = row["source_pair_id"].split("__")[0]
            kind = row["pair_id"].rsplit("__", 1)[1]
            table[(item, row["condition"], kind)] = row["parsed_choice"]
        answers[name] = table

    # run-2 vs probe agreement on the byte-identical arms and identical 2AFC question
    run2 = {}
    for row in load_jsonl(args.run_dir / "manifest.jsonl"):
        if row["calibration"] == "equal_energy" and row["masker"] == "low_pass_400":
            run2[row["pair_id"]] = row
    run2_preds = {}
    for name, path in (("3b", args.run_dir / "pred_v2_3b.jsonl"), ("7b", args.run_dir / "pred_v2_7b.jsonl")):
        for row in load_jsonl(path):
            run2_preds[(name, row["pair_id"])] = row.get("parsed_choice")
    probe_items = set(by_item)
    for name in models:
        agree = total = 0
        for item in probe_items:
            # only `original` and `diff` are byte-identical to run 2; the probe's `same`
            # arm deliberately uses a different (magnitude-matched) same-class donor
            for arm in ("original", "diff"):
                key = f"{item}__equal_energy__low_pass_400__{arm}"
                if key not in run2:
                    continue
                want = run2_preds.get((name, key))
                got = answers[name].get((item, arm, "2afc"))
                if want is None or got is None:
                    continue
                total += 1
                agree += want == got
        run2_agreement[name] = {"n": total, "agreement": agree / total if total else None}

    out: dict[str, object] = {
        "per_item": per_item,
        "arm_summary": {},
        "models": {},
        "run2_vs_probe_2afc_agreement": run2_agreement,
    }
    for arm in ARMS:
        lsd = np.array([entry["arms"][arm]["lsd_db"] for entry in per_item.values()])
        aud = np.array([entry["arms"][arm]["change_audibility_db"] for entry in per_item.values()])
        out["arm_summary"][arm] = {
            "lsd_db_median": float(np.median(lsd)),
            "lsd_db_p5_p95": [float(np.percentile(lsd, 5)), float(np.percentile(lsd, 95))],
            "change_audibility_db_median": float(np.median(aud)),
            "change_audibility_db_p5_p95": [float(np.percentile(aud, 5)), float(np.percentile(aud, 95))],
            "delta_rms_db_median": float(np.median([entry["arms"][arm]["delta_rms_db"] for entry in per_item.values()])),
            "delta_rms_db_gain_normalised_median": float(
                np.median([entry["arms"][arm]["delta_rms_db_gain_normalised"] for entry in per_item.values()])
            ),
            "change_audibility_db_gain_normalised_median": float(
                np.median([entry["arms"][arm]["change_audibility_db_gain_normalised"] for entry in per_item.values()])
            ),
            "span_rms_db_re_surround_median": float(
                np.median([entry["arms"][arm]["span_rms_db_re_surround"] for entry in per_item.values()])
            ),
            "headroom_gain_estimate_median": float(
                np.median([entry["arms"][arm]["headroom_gain_estimate"] for entry in per_item.values()])
            ),
        }

    items = sorted(per_item)
    lsd_same = np.array([per_item[i]["arms"]["same"]["lsd_db"] for i in items])
    lsd_diff = np.array([per_item[i]["arms"]["diff"]["lsd_db"] for i in items])
    lsd_same_loud = np.array([per_item[i]["arms"]["same_loud"]["lsd_db"] for i in items])
    matched = lsd_same >= lsd_diff
    out["probe_matched_subset"] = {
        "n_same_lsd_ge_diff": int(matched.sum()),
        "n_same_loud_lsd_gt_diff": int((lsd_same_loud > lsd_diff).sum()),
        "median_lsd_same_minus_diff_db": float(np.median(lsd_same - lsd_diff)),
        "median_lsd_same_loud_minus_diff_db": float(np.median(lsd_same_loud - lsd_diff)),
    }
    for name, table in answers.items():
        block = {}
        for kind in KINDS:
            def changed(arm: str, subset: np.ndarray | None = None) -> float | None:
                hits = total = 0
                for idx, item in enumerate(items):
                    if subset is not None and not subset[idx]:
                        continue
                    base = table.get((item, "original", kind))
                    got = table.get((item, arm, kind))
                    if base is None or got is None:
                        continue
                    total += 1
                    hits += got != base
                return hits / total if total else None
            block[kind] = {
                "P_change_by_arm": {arm: changed(arm) for arm in ARMS},
                "P_change_by_arm_matched_subset": {arm: changed(arm, matched) for arm in ARMS},
            }
        out["models"][name] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'arm':10} {'LSD med':>8} {'dRMS med':>9} {'dRMS* med':>9} {'AUD med':>9} {'AUD* med':>9} "
          f"{'span/surr':>9} {'headroom':>9}")
    for arm in ARMS:
        s = out["arm_summary"][arm]
        print(f"{arm:10} {s['lsd_db_median']:8.2f} {s['delta_rms_db_median']:9.2f} "
              f"{s['delta_rms_db_gain_normalised_median']:9.2f} {s['change_audibility_db_median']:9.2f} "
              f"{s['change_audibility_db_gain_normalised_median']:9.2f} "
              f"{s['span_rms_db_re_surround_median']:9.2f} {s['headroom_gain_estimate_median']:9.3f}")
    print(json.dumps(out["probe_matched_subset"], indent=2))
    for name in models:
        print(f"\n=== model {name}  P(change vs its own original answer)")
        for kind in KINDS:
            row = out["models"][name][kind]["P_change_by_arm"]
            rowm = out["models"][name][kind]["P_change_by_arm_matched_subset"]
            print(f"  {kind:6} all: " + "  ".join(f"{a}={row[a]:.3f}" for a in ARMS if row[a] is not None))
            print(f"  {kind:6} mch: " + "  ".join(f"{a}={rowm[a]:.3f}" for a in ARMS if rowm[a] is not None))
    print(f"\nrun2 vs probe 2AFC agreement on identical arms: {run2_agreement}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
