"""Analyse the two-axis difficulty duel: accuracy, P(change), dose-response, matched contrast.

The build (``scripts/ctl_axis_duel_build.py``) lays five difficulty levels of two axes over
the same items — axis ``energy`` (white noise, level solved for energy SNR) and axis
``audibility`` (threshold-shaped noise, level solved for the audibility margin) — and
renders ``original`` and ``diff`` for each cell. This script turns the manifest plus a
predictions jsonl into the numbers the duel exists for:

* per axis and level: accuracy on the ``original`` arm, accuracy on the ``diff`` arm,
  ``P(change | diff)``, and the **realised** axis values of the cue that is actually in the
  audio (``span_audibility_db`` / ``span_energy_snr_db``, computed from the rendered
  queried span). The event-referenced ``audibility_db`` / ``energy_snr_db`` are reported
  next to them: on an edited arm the event is gone from the file, so fitting accuracy
  against them fits against a signal the model never heard;
* the dose-response per axis — the accuracy span across levels with a bootstrap CI over
  items, and a logistic fit of accuracy against the realised axis value;
* the between-axis difference at **matched realised energy** and at **matched realised
  audibility**, with CIs, so the two axes can be compared where they actually coincide
  rather than at their nominal targets;
* the **always-A floor**: this is a two-alternative question with ground truth A on the
  ``original`` arm and B on the ``diff`` arm, so a model that always answers A scores 1.0
  on one arm and 0.0 on the other and 0.5 pooled. Every pooled accuracy and every
  dose-response is reported against that floor, because a flat curve at 0.5 is a
  degenerate strategy, not an absence of sensitivity.

Bootstrap clusters on the **item**, not the recording: the five levels of one axis and the
two arms of one cell are the same item measured repeatedly, so resampling recordings would
understate every interval (docs/protocol-invariants.md, §2.5).

Usage::

    python scripts/ctl_axis_duel_analyse.py \
        --dir outputs/ctl/axis_duel --predictions outputs/ctl/axis_duel/pred_3b.jsonl

Output JSON is written next to the manifest as
``duel_analysis_<predictions-stem>.json`` unless ``--output`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ("original", "diff")
GROUND_TRUTH = {"original": "A", "diff": "B"}
#: Pooled accuracy of a model that always answers A: 1.0 on the original arm, 0.0 on the
#: diff arm, equal n on both, so exactly 0.5. Reported with every comparison.
ALWAYS_A_FLOOR = 0.5
SPAN_FIELDS = {
    "energy": "span_energy_snr_db",
    "audibility": "span_audibility_db",
}
EVENT_FIELDS = {
    "energy": "energy_snr_db",
    "audibility": "audibility_db",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- loading ------------------------------------------------------------------


def load_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_predictions(path: Path) -> dict[str, str | None]:
    """pair_id -> parsed choice, from the evaluator's jsonl.

    ``parsed_choice`` is the field the remote evaluator writes after parsing the
    assistant completion; ``prediction`` is only used as a fallback when it is exactly one
    of the two option letters, never as free text.
    """

    out: dict[str, str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        choice = record.get("parsed_choice")
        if choice is None:
            fallback = record.get("prediction")
            choice = fallback if fallback in {"A", "B"} else None
        if choice not in {"A", "B"}:
            choice = None
        out[str(record.get("pair_id"))] = choice
    return out


def collect(rows: list[dict], choices: dict[str, str | None]) -> dict:
    """Nest the manifest by item -> (axis, level) -> condition, with the model's answer.

    Values are read from the rendered span when the build recorded them (the fields added
    2026-09-18) and from the event-referenced pair otherwise, with ``span_fields`` saying
    which happened so a reader of the JSON can tell.
    """

    items: dict[str, dict[tuple[str, int], dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    span_fields = True
    missing_predictions = 0
    unparsed = 0
    for row in rows:
        key = (str(row["axis"]), int(row["level"]))
        condition = str(row["condition"])
        if "span_audibility_db" in row and "span_energy_snr_db" in row:
            span_audibility = float(row["span_audibility_db"])
            span_energy = float(row["span_energy_snr_db"])
        else:
            span_fields = False
            span_audibility = float(row["audibility_db"])
            span_energy = float(row["energy_snr_db"])
        pair_id = str(row["pair_id"])
        if pair_id not in choices:
            missing_predictions += 1
            choice = None
        else:
            choice = choices[pair_id]
            if choice is None:
                unparsed += 1
        items[str(row["item_id"])][key][condition] = {
            "choice": choice,
            "correct": None if choice is None else float(choice == GROUND_TRUTH[condition]),
            "span_audibility_db": span_audibility,
            "span_energy_snr_db": span_energy,
            "audibility_db": float(row["audibility_db"]),
            "energy_snr_db": float(row["energy_snr_db"]),
            "masker": str(row["masker"]),
            "target_value": float(row["target_value"]) if "target_value" in row else None,
        }
    return {
        "items": {k: dict(v) for k, v in items.items()},
        "span_fields": span_fields,
        "n_rows": len(rows),
        "n_predictions": len(choices),
        "n_pairs_without_prediction": missing_predictions,
        "n_unparsed": unparsed,
    }


# --- per-item primitives ------------------------------------------------------


def arm_accuracy(cell: dict[str, dict], condition: str) -> float | None:
    record = cell.get(condition)
    if record is None or record["correct"] is None:
        return None
    return float(record["correct"])


def pooled_accuracy(cell: dict[str, dict]) -> float | None:
    values = [v for c in CONDITIONS if (v := arm_accuracy(cell, c)) is not None]
    return float(np.mean(values)) if values else None


def p_change(cell: dict[str, dict]) -> float | None:
    original, diff = cell.get("original"), cell.get("diff")
    if original is None or diff is None:
        return None
    if original["choice"] is None or diff["choice"] is None:
        return None
    return float(original["choice"] != diff["choice"])


def item_matrix(items: dict, axis: str) -> tuple[list[str], list[int]]:
    levels = sorted({level for cells in items.values() for (a, level) in cells if a == axis})
    return sorted(items), levels


def accuracy_at(items: dict, item_ids: list[str], axis: str, level: int, arm: str) -> float | None:
    values = []
    for item_id in item_ids:
        cell = items[item_id].get((axis, level))
        if cell is None:
            continue
        value = pooled_accuracy(cell) if arm == "pooled" else arm_accuracy(cell, arm)
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


# --- statistics ---------------------------------------------------------------


def bootstrap_ci(
    item_ids: list[str],
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = 20260918,
) -> tuple[float, float, float, int]:
    """Percentile CI over a statistic of item-clustered data.

    ``statistic(sample_ids) -> float | None`` is recomputed on each resample; resampling
    items (not recordings) is what keeps the interval honest, since an item contributes
    ten correlated cells.
    """

    rng = random.Random(seed)
    point = statistic(item_ids)
    if point is None:
        return float("nan"), float("nan"), float("nan"), 0
    values = []
    for _ in range(n_boot):
        sample = [item_ids[rng.randrange(len(item_ids))] for _ in range(len(item_ids))]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return float(point), float("nan"), float("nan"), 0
    return (
        float(point),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        len(values),
    )


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """IRLS logistic fit ``p = sigmoid(b0 + b1 x)``; ``None`` if it cannot be fitted."""

    if x.size < 4 or len(set(y.tolist())) < 2:
        return None
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        weight = np.maximum(p * (1.0 - p), 1e-9)
        try:
            step = np.linalg.solve(
                design.T @ (design * weight[:, None]), design.T @ (weight * (eta + (y - p) / weight))
            )
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        if np.max(np.abs(step - beta)) < 1e-9:
            beta = step
            break
        beta = step
    return float(beta[0]), float(beta[1])


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))))


# --- reports ------------------------------------------------------------------


def realised_block(cells: list[dict[str, dict]], arm: str) -> dict[str, float]:
    def median(field: str) -> float:
        values = [
            float(cell[arm][field]) for cell in cells if arm in cell and cell[arm] is not None
        ]
        return float(np.median(values)) if values else float("nan")

    return {
        f"{arm}_span_energy_snr_db_median": median("span_energy_snr_db"),
        f"{arm}_span_audibility_db_median": median("span_audibility_db"),
        f"{arm}_event_energy_snr_db_median": median("energy_snr_db"),
        f"{arm}_event_audibility_db_median": median("audibility_db"),
    }


def per_axis_level(items: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for axis in ("energy", "audibility"):
        item_ids, levels = item_matrix(items, axis)
        for level in levels:
            cells = [
                items[item_id][(axis, level)]
                for item_id in item_ids
                if (axis, level) in items[item_id]
            ]
            block: dict[str, object] = {
                "axis": axis,
                "level": level,
                "masker": cells[0]["original"]["masker"] if cells else None,
                "n": {
                    "original": sum(
                        1 for c in cells if arm_accuracy(c, "original") is not None
                    ),
                    "diff": sum(1 for c in cells if arm_accuracy(c, "diff") is not None),
                },
                "accuracy": {
                    arm: accuracy_at(items, item_ids, axis, level, arm)
                    for arm in ("original", "diff", "pooled")
                },
                "P_change_given_diff": float(
                    np.mean([v for c in cells if (v := p_change(c)) is not None] or [float("nan")])
                ),
                "realised": {
                    **realised_block(cells, "original"),
                    **realised_block(cells, "diff"),
                },
                "target_value": cells[0]["original"]["target_value"] if cells else None,
                "always_a_floor": ALWAYS_A_FLOOR,
            }
            out[f"{axis}_L{level}"] = block
    return out


def dose_response(items: dict, *, n_boot: int, seed: int) -> dict[str, dict]:
    """Accuracy span across levels and a logistic fit, per axis, bootstrapped over items."""

    out: dict[str, dict] = {}
    for axis in ("energy", "audibility"):
        item_ids, levels = item_matrix(items, axis)

        def span(sample: list[str], arm: str = "pooled") -> float | None:
            values = [
                accuracy_at(items, sample, axis, level, arm)
                for level in levels
            ]
            values = [v for v in values if v is not None]
            return float(max(values) - min(values)) if len(values) >= 2 else None

        block: dict[str, object] = {"levels": levels, "always_a_floor": ALWAYS_A_FLOOR}
        for arm in ("pooled", "original", "diff"):
            mean, lo, hi, n = bootstrap_ci(
                item_ids, lambda s, arm=arm: span(s, arm), n_boot=n_boot, seed=seed
            )
            block[f"accuracy_span_{arm}_points"] = 100.0 * mean
            block[f"accuracy_span_{arm}_ci95_points"] = [100.0 * lo, 100.0 * hi]
            block[f"accuracy_span_{arm}_n_boot"] = n
            block[f"per_level_accuracy_{arm}"] = [
                accuracy_at(items, item_ids, axis, level, arm) for level in levels
            ]

        # logistic fit against the realised value of the axis's own calibrated quantity
        x_field = SPAN_FIELDS[axis]
        xs, ys = [], []
        for item_id in item_ids:
            for level in levels:
                cell = items[item_id].get((axis, level))
                if cell is None:
                    continue
                for condition in CONDITIONS:
                    record = cell.get(condition)
                    if record is None or record["correct"] is None:
                        continue
                    xs.append(record[x_field])
                    ys.append(record["correct"])
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        fit = logistic_fit(x, y)

        def fitted_drop(sample: list[str]) -> float | None:
            sx, sy = [], []
            for item_id in sample:
                for level in levels:
                    cell = items[item_id].get((axis, level))
                    if cell is None:
                        continue
                    for condition in CONDITIONS:
                        record = cell.get(condition)
                        if record is None or record["correct"] is None:
                            continue
                        sx.append(record[x_field])
                        sy.append(record["correct"])
            if len(sx) < 4:
                return None
            fitted = logistic_fit(np.asarray(sx, dtype=float), np.asarray(sy, dtype=float))
            if fitted is None:
                return None
            low, high = np.percentile(sx, [10, 90])
            return 100.0 * (sigmoid(fitted[0] + fitted[1] * high) - sigmoid(fitted[0] + fitted[1] * low))

        drop, drop_lo, drop_hi, drop_n = bootstrap_ci(
            item_ids, fitted_drop, n_boot=n_boot, seed=seed
        )
        block["dose_response_field"] = x_field
        block["realised_value_p10_p90"] = [float(v) for v in np.percentile(x, [10, 90])] if x.size else []
        block["logistic_slope_per_db"] = None if fit is None else fit[1]
        block["fitted_accuracy_drop_points_over_p10_p90"] = drop
        block["fitted_accuracy_drop_ci95_points"] = [drop_lo, drop_hi]
        block["fitted_accuracy_drop_n_boot"] = drop_n

        # slope difference between axes is reported in the matched comparison below; a
        # within-axis slope is only interpretable next to the realised range it spans
        out[axis] = block
    return out


def matched_difference(
    items: dict, match_on: str, *, n_boot: int, seed: int
) -> dict[str, object]:
    """Energy-axis accuracy minus audibility-axis accuracy at matched realised values.

    For every item and every level of the axis that *owns* ``match_on``, the closest cell
    of the other axis in that realised quantity is used as the match, so the comparison is
    between two cells the model actually heard rather than between two nominal targets.
    Sign convention: **positive means the energy axis is easier** at the matched value.
    """

    owner = "energy" if match_on == "span_energy_snr_db" else "audibility"
    other = "audibility" if owner == "energy" else "energy"
    item_ids, owner_levels = item_matrix(items, owner)
    _, other_levels = item_matrix(items, other)

    per_item: dict[str, dict[str, float]] = defaultdict(dict)
    mismatches: list[float] = []
    for item_id in item_ids:
        for arm in ("pooled", "original", "diff"):
            values = []
            for level in owner_levels:
                owner_cell = items[item_id].get((owner, level))
                if owner_cell is None or "original" not in owner_cell:
                    continue
                target = owner_cell["original"][match_on]
                candidates = [
                    (abs(items[item_id][(other, k)]["original"][match_on] - target), k)
                    for k in other_levels
                    if (other, k) in items[item_id]
                ]
                if not candidates:
                    continue
                mismatch, best = min(candidates)
                if arm == "pooled":
                    mismatches.append(float(mismatch))
                # the difference is always energy minus audibility, whichever axis owned
                # the matching; computing it as "owner minus other" silently flips the
                # sign of the audibility-matched comparison
                energy_cell = owner_cell if owner == "energy" else items[item_id][(other, best)]
                audibility_cell = owner_cell if owner == "audibility" else items[item_id][(other, best)]
                if arm == "pooled":
                    energy_acc, audibility_acc = (
                        pooled_accuracy(energy_cell),
                        pooled_accuracy(audibility_cell),
                    )
                else:
                    energy_acc, audibility_acc = (
                        arm_accuracy(energy_cell, arm),
                        arm_accuracy(audibility_cell, arm),
                    )
                if energy_acc is None or audibility_acc is None:
                    continue
                values.append(energy_acc - audibility_acc)
            if values:
                per_item[item_id][arm] = float(np.mean(values))

    block: dict[str, object] = {
        "matched_on": match_on,
        "owner_axis": owner,
        "difference_definition": "accuracy(energy axis) - accuracy(audibility axis); positive = energy axis easier",
        "realised_mismatch_median_db": float(np.median(mismatches)) if mismatches else None,
        "realised_mismatch_p90_db": float(np.percentile(mismatches, 90)) if mismatches else None,
        "n_pairs": len(mismatches),
        "always_a_floor": ALWAYS_A_FLOOR,
    }
    for arm in ("pooled", "original", "diff"):
        ids = [item_id for item_id in item_ids if arm in per_item.get(item_id, {})]
        mean, lo, hi, n = bootstrap_ci(
            ids,
            lambda s, arm=arm: float(np.mean([per_item[i][arm] for i in s]))
            if all(arm in per_item.get(i, {}) for i in s)
            else None,
            n_boot=n_boot,
            seed=seed,
        )
        block[f"mean_difference_{arm}_points"] = 100.0 * mean
        block[f"mean_difference_{arm}_ci95_points"] = [100.0 * lo, 100.0 * hi]
        block[f"n_items_{arm}"] = len(ids)
    return block


def analyse(
    manifest_path: Path,
    predictions_path: Path,
    *,
    n_boot: int = 2000,
    seed: int = 20260918,
) -> dict:
    rows = load_manifest(manifest_path)
    choices = load_predictions(predictions_path)
    collected = collect(rows, choices)
    items = collected["items"]
    if not items:
        raise SystemExit(f"manifest {manifest_path} has no rows")
    return {
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
        },
        "coverage": {
            "n_items": len(items),
            "n_manifest_rows": collected["n_rows"],
            "n_predictions": collected["n_predictions"],
            "n_pairs_without_prediction": collected["n_pairs_without_prediction"],
            "n_unparsed": collected["n_unparsed"],
            "span_fields_present": collected["span_fields"],
            "note": (
                "realised axis values are read from the rendered queried span "
                "(span_audibility_db / span_energy_snr_db) when the build recorded them"
                if collected["span_fields"]
                else "this manifest predates the span fields; realised values fall back to "
                "the event-referenced audibility_db / energy_snr_db, which on an edited "
                "arm describe a signal that is not in the audio"
            ),
        },
        "always_a_floor": {
            "pooled_accuracy": ALWAYS_A_FLOOR,
            "original_arm_accuracy": 1.0,
            "diff_arm_accuracy": 0.0,
            "by_construction": True,
            "note": (
                "2AFC with ground truth A on original and B on diff, equal n per arm: a "
                "model that always answers A scores exactly 0.5 pooled. Read every pooled "
                "accuracy and every dose-response against this floor."
            ),
        },
        "per_axis_level": per_axis_level(items),
        "dose_response": dose_response(items, n_boot=n_boot, seed=seed),
        "matched_comparison": {
            "at_matched_realised_energy": matched_difference(
                items, "span_energy_snr_db", n_boot=n_boot, seed=seed
            ),
            "at_matched_realised_audibility": matched_difference(
                items, "span_audibility_db", n_boot=n_boot, seed=seed
            ),
        },
        "bootstrap": {"n": n_boot, "seed": seed, "cluster": "item"},
    }


def _fmt(value: float | None, spec: str = "+.3f") -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else format(float(value), spec)


def print_report(results: dict) -> None:
    print(f"items {results['coverage']['n_items']}  "
          f"rows {results['coverage']['n_manifest_rows']}  "
          f"unparsed {results['coverage']['n_unparsed']}  "
          f"missing predictions {results['coverage']['n_pairs_without_prediction']}  "
          f"span fields {results['coverage']['span_fields_present']}")
    floor = results["always_a_floor"]
    print(f"\nALWAYS-A FLOOR: pooled {floor['pooled_accuracy']:.3f} by construction "
          f"(original {floor['original_arm_accuracy']:.1f}, diff {floor['diff_arm_accuracy']:.1f})")
    print(f"\n{'cell':16} {'n_o':>4} {'n_d':>4} {'acc_orig':>8} {'acc_diff':>8} "
          f"{'acc_pool':>8} {'P(chg|d)':>8} {'spanE':>7} {'spanA':>7} {'evE':>7} {'evA':>7}")
    for key, block in results["per_axis_level"].items():
        realised = block["realised"]
        print(
            f"{key:16} {block['n']['original']:>4} {block['n']['diff']:>4} "
            f"{_fmt(block['accuracy']['original'])} {_fmt(block['accuracy']['diff'])} "
            f"{_fmt(block['accuracy']['pooled'])} {_fmt(block['P_change_given_diff'])} "
            f"{_fmt(realised['original_span_energy_snr_db_median'], '+.1f')} "
            f"{_fmt(realised['original_span_audibility_db_median'], '+.1f')} "
            f"{_fmt(realised['original_event_energy_snr_db_median'], '+.1f')} "
            f"{_fmt(realised['original_event_audibility_db_median'], '+.1f')}"
        )
    print("\ndose-response (accuracy span across levels, bootstrap CI over items):")
    for axis, block in results["dose_response"].items():
        print(f"  {axis:10} pooled {block['accuracy_span_pooled_points']:+6.1f} pt "
              f"[{block['accuracy_span_pooled_ci95_points'][0]:+.1f},"
              f"{block['accuracy_span_pooled_ci95_points'][1]:+.1f}]  "
              f"original {block['accuracy_span_original_points']:+6.1f}  "
              f"diff {block['accuracy_span_diff_points']:+6.1f}  "
              f"| logistic slope {_fmt(block['logistic_slope_per_db'], '+.4f')} /dB, "
              f"fitted drop {_fmt(block['fitted_accuracy_drop_points_over_p10_p90'], '+.1f')} pt")
    print("\nbetween-axis difference at matched realised values "
          "(energy minus audibility; positive = energy axis easier):")
    for name, block in results["matched_comparison"].items():
        print(f"  {name:34} pooled {block['mean_difference_pooled_points']:+6.1f} pt "
              f"[{block['mean_difference_pooled_ci95_points'][0]:+.1f},"
              f"{block['mean_difference_pooled_ci95_points'][1]:+.1f}]  "
              f"original {block['mean_difference_original_points']:+6.1f}  "
              f"diff {block['mean_difference_diff_points']:+6.1f}  "
              f"(matching median {_fmt(block['realised_mismatch_median_db'], '+.2f')} dB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("outputs/ctl/axis_duel"))
    parser.add_argument("--manifest", type=Path, help="defaults to <dir>/manifest.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="defaults to <dir>/duel_analysis_<pred stem>.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    manifest = args.manifest or (args.dir / "manifest.jsonl")
    if not manifest.exists():
        print(f"no manifest at {manifest}", file=sys.stderr)
        return 2
    if not args.predictions.exists():
        print(f"no predictions at {args.predictions}", file=sys.stderr)
        return 2

    results = analyse(
        manifest, args.predictions, n_boot=args.bootstrap, seed=args.seed
    )
    output = args.output or (manifest.parent / f"duel_analysis_{args.predictions.stem}.json")
    output.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print_report(results)
    print(f"\nwritten: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
