#!/usr/bin/env python3
"""Per-class competence and selectivity profile, and whether one explains the other.

The whole-grid numbers hide a very wide spread: on the confirm split the same
model ranges from 1.00 to 0.61 accuracy across the 33 classes that appear there
(36 in the vocabulary). Two questions follow, and they have different answers.

1. *Is the spread real or sampling noise?* Each class has 24-58 views, so the
   per-class interval is wide and the script reports it rather than the point
   estimate alone.

2. *Is per-class selectivity just a mechanical consequence of per-class
   accuracy?* This is the important one. At chance accuracy a model that answers
   at random changes its answer on any edit about half the time, and a model with
   a fixed position bias never changes it, so the extremes are constrained. If
   accuracy and selectivity correlate strongly across classes, then "this class
   is not evidence-driven" is not a separable claim -- it is the same statement
   as "this class is not recognised", and the class-level dissociation the
   competence gradient failed to produce would not exist here either.

Reported per class: n, original accuracy with a Wilson interval, the change rates
under each condition, selectivity, and a note when a class looks competent but
poorly selective. Then the across-class correlations, and the residual after
regressing selectivity on accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DIFF = "target_replaced_diff"
SAME = "target_replaced_same"
CONTROLS = ("nuisance_replaced", "nuisance_removed", "background_removed", "sham")
SHAM = "sham"


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--min-items", type=int, default=8)
    parser.add_argument("--competent", type=float, default=0.85)
    parser.add_argument("--low-selectivity", type=float, default=0.35)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help=(
            "Model key to record for each input, in the same order. Defaults to "
            "the input filename stem, which embeds the run name ('r2v2_qwen3b') "
            "rather than the model identity. Pass explicit keys so the artifact "
            "matches evidence_table.json (see consolidate_evidence.py MODELS); "
            "otherwise the keys have to be renamed by hand, which is what made "
            "the committed artifact unreproducible."
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

    report: dict = {"splits": args.split, "min_items": args.min_items, "models": {}}
    for path, key in zip(args.predictions, keys):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Keyed by ``pair_id`` (one view per source pair x surface order), matching
        # synth_attribution.py. Keying by source_pair_id would let the second
        # surface order overwrite the first, silently halving the sample and
        # picking an order arbitrarily.
        by_item: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            if not row.get("parsed_choice") or "error" in row:
                continue
            if args.split and row.get("split") != args.split:
                continue
            by_item[row["pair_id"]][row["condition"]] = row

        per_class: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for item, conditions in by_item.items():
            original = conditions.get("original")
            if original is None or DIFF not in conditions or SAME not in conditions:
                continue
            if SHAM not in conditions:
                continue
            label = str(original.get("query_class", "?"))
            base = str(original["parsed_choice"])
            bucket = per_class[label]
            bucket["correct"].append(bool(original.get("correct")))
            bucket["diff"].append(str(conditions[DIFF]["parsed_choice"]) != base)
            bucket["same"].append(str(conditions[SAME]["parsed_choice"]) != base)
            bucket["sham"].append(str(conditions[SHAM]["parsed_choice"]) != base)
            for control in CONTROLS:
                if control in conditions:
                    bucket[f"c_{control}"].append(
                        str(conditions[control]["parsed_choice"]) != base
                    )

        table = []
        for label, values in per_class.items():
            n = len(values["correct"])
            if n < args.min_items:
                continue
            hits = sum(values["correct"])
            accuracy = hits / n
            low, high = wilson(hits, n)
            a_diff = sum(values["diff"]) / n
            a_same = sum(values["same"]) / n
            table.append(
                {
                    "class": label,
                    "n": n,
                    "original_accuracy": accuracy,
                    "accuracy_ci": [low, high],
                    "A_diff": a_diff,
                    "A_same": a_same,
                    "A_sham": sum(values["sham"]) / n,
                    "selectivity": a_diff - a_same,
                    "flag_competent_low_selectivity": bool(
                        accuracy >= args.competent
                        and (a_diff - a_same) < args.low_selectivity
                    ),
                }
            )
        table.sort(key=lambda row: -row["original_accuracy"])

        accuracies = [row["original_accuracy"] for row in table]
        selectivities = [row["selectivity"] for row in table]
        diffs = [row["A_diff"] for row in table]
        sames = [row["A_same"] for row in table]

        # Regress selectivity on accuracy and look at the residuals, so a class
        # that is poorly selective *for its accuracy* is visible rather than
        # hidden behind the overall correlation.
        r = pearson(accuracies, selectivities)
        residual_flagged = []
        if len(table) >= 3:
            n = len(table)
            mx, my = sum(accuracies) / n, sum(selectivities) / n
            var = sum((x - mx) ** 2 for x in accuracies)
            if var > 0:
                slope = (
                    sum((x - mx) * (y - my) for x, y in zip(accuracies, selectivities))
                    / var
                )
                intercept = my - slope * mx
                for row in table:
                    row["selectivity_residual"] = (
                        row["selectivity"]
                        - (intercept + slope * row["original_accuracy"])
                    )
                residual_flagged = sorted(
                    (row for row in table if row["n"] >= args.min_items),
                    key=lambda row: row.get("selectivity_residual", 0.0),
                )[:5]

        report["models"][key] = {
            "classes": len(table),
            "items": sum(row["n"] for row in table),
            "accuracy_spread": {
                "min": min(accuracies),
                "median": sorted(accuracies)[len(accuracies) // 2],
                "max": max(accuracies),
                "stdev": (
                    math.sqrt(
                        sum((x - sum(accuracies) / len(accuracies)) ** 2 for x in accuracies)
                        / len(accuracies)
                    )
                    if accuracies
                    else float("nan")
                ),
            },
            "selectivity_spread": {
                "min": min(selectivities),
                "median": sorted(selectivities)[len(selectivities) // 2],
                "max": max(selectivities),
            },
            "correlations_across_classes": {
                "accuracy_vs_selectivity": r,
                "accuracy_vs_A_diff": pearson(accuracies, diffs),
                "accuracy_vs_A_same": pearson(accuracies, sames),
            },
            "flagged_competent_but_low_selectivity": [
                row["class"] for row in table if row["flag_competent_low_selectivity"]
            ],
            "largest_negative_selectivity_residuals": [
                {
                    "class": row["class"],
                    "n": row["n"],
                    "accuracy": round(row["original_accuracy"], 3),
                    "selectivity": round(row["selectivity"], 3),
                    "residual": round(row.get("selectivity_residual", float("nan")), 3),
                }
                for row in residual_flagged
            ],
            "per_class": table,
        }

    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    for model, values in report["models"].items():
        spread = values["accuracy_spread"]
        corr = values["correlations_across_classes"]
        print(
            f"{model}: {values['classes']} classes, accuracy {spread['min']:.2f}"
            f"-{spread['max']:.2f} (sd {spread['stdev']:.3f}), "
            f"r(acc,sel)={corr['accuracy_vs_selectivity']:.2f}, "
            f"r(acc,A_same)={corr['accuracy_vs_A_same']:.2f}"
        )
    print(f"wrote {args.output}" if args.output else "")


if __name__ == "__main__":
    main()
