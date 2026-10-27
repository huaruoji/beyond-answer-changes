#!/usr/bin/env python3
"""Competence-free evidence attribution for the synthetic seven-condition grid.

Everything here is measured against the model's *own* answer, so the numbers are
defined even when accuracy is low. That is the lesson carried over from the
withdrawn pipeline, whose correctness-conditioned metrics became uninterpretable
at chance.

The headline contrast is matched by construction. ``target_replaced_diff`` and
``target_replaced_same`` rewrite the *same* sample span with the *same* operation
through the *same* crossfade path; they differ only in whether the replacement
preserves the queried identity. So the difference between their answer-change
rates cannot be explained by low-level perturbation:

    selectivity  = A_diff - A_same      evidence identity drives the answer
    specificity  = A_same - A_sham      is the region special, or is any edit visible?
    perturbation = max(A_nuis, A_bg, A_sham)

Intervals resample whole source clips (ADR-011), never individual records: within
an item the conditions are dependent, and the paired views are two renderings of
one physical soundscape.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

#: Condition sets differ between routes. The synthetic grid has a same-class
#: donor and therefore a truth-preserving edit of the *same span*; the natural
#: route cannot build one, because TEMPO's event descriptions are free text and
#: essentially unique per clip, so its matched control writes the same donor
#: into a different span instead. Everything downstream is keyed off condition
#: names, so the analysis derives the set from the data and reports which
#: contrasts are actually available.
SYNTHETIC_CONDITIONS = (
    "original",
    "target_replaced_diff",
    "target_replaced_same",
    "nuisance_removed",
    "background_removed",
    "sham",
    "silence",
)
NATURAL_CONDITIONS = (
    "original",
    "target_replaced_diff",
    "nuisance_replaced",
    "nuisance_removed",
    "background_removed",
    "sham",
    "silence",
)

#: Gate thresholds, from docs/synthesis_design.md section 8.
COMPETENCE_ACCURACY = 0.75
COMPETENCE_DEPENDENCE = 0.20
SHAM_TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split",
        nargs="+",
        default=None,
        help="restrict to these manifest splits. The design forbids pooling the "
        "pilot and confirmatory splits, so the filter has to be explicit",
    )
    parser.add_argument("--diff-condition", default="target_replaced_diff")
    parser.add_argument(
        "--same-condition",
        default="target_replaced_same",
        help="truth-preserving edit of the same span; absent on the natural route",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["nuisance_replaced", "nuisance_removed", "background_removed", "sham", "silence"],
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text().splitlines()
        if line.strip()
    ]
    usable = [row for row in rows if row.get("parsed_choice") and "error" not in row]
    dropped = len(rows) - len(usable)
    if args.split:
        allowed_splits = set(args.split)
        usable = [row for row in usable if row.get("split") in allowed_splits]
    if not usable:
        raise SystemExit(f"no usable records for splits {args.split}")

    by_view: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in usable:
        by_view[str(row["pair_id"])][row["condition"]] = row

    present = {str(row["condition"]) for row in usable}
    if "original" not in present:
        raise SystemExit("no original condition in the predictions")
    if args.diff_condition not in present:
        raise SystemExit(f"diff condition {args.diff_condition!r} absent")
    has_same = args.same_condition in present

    # A view is usable only if the item produced every condition the contrast
    # needs. Views missing a control are dropped rather than silently scored
    # against a different denominator.
    needed = {"original", args.diff_condition} | (
        {args.same_condition} if has_same else set()
    )
    views: dict[str, dict[str, dict[str, Any]]] = {
        pair_id: conditions
        for pair_id, conditions in by_view.items()
        if needed.issubset(conditions)
    }

    per_view = [_view_changes(conditions) for conditions in views.values()]
    clusters: dict[str, list[dict[str, bool]]] = defaultdict(list)
    for pair_id, conditions in views.items():
        clusters[str(conditions["original"].get("source_pair_id", pair_id))].append(
            _view_changes(conditions)
        )

    point = _aggregate(per_view) | _contrasts(
        _aggregate(per_view), args.diff_condition,
        args.same_condition if has_same else None, args.controls,
    )
    intervals = _bootstrap(
        clusters,
        args.samples,
        args.seed,
        args.diff_condition,
        args.same_condition if has_same else None,
        args.controls,
    )

    accuracy = _accuracy_by_condition(views)
    classes = _class_breakdown(views)

    report: dict[str, Any] = {
        "predictions": str(args.predictions),
        "splits": sorted({str(row.get("split", "?")) for row in usable}),
        "conditions_present": sorted(present),
        "matched_same_span_available": has_same,
        "diff_condition": args.diff_condition,
        "same_condition": args.same_condition if has_same else None,
        "records": len(rows),
        "records_dropped_unparsed": dropped,
        "views": len(views),
        "clips": len(clusters),
        "accuracy_by_condition": accuracy,
        "metrics": point,
        "bootstrap_95": intervals,
        "gates": _gates(accuracy, point),
        "per_class": classes,
    }
    report["verdict"] = _verdict(report)

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


def _view_changes(conditions: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Did the model's answer change under each edit, for one surface view?"""

    base = str(conditions["original"]["parsed_choice"])
    return {
        condition: str(row["parsed_choice"]) != base
        for condition, row in conditions.items()
        if condition != "original"
    }


def _aggregate(per_view: list[dict[str, bool]]) -> dict[str, float]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for view in per_view:
        for condition, changed in view.items():
            totals[condition].append(changed)
    means = {name: _mean(values) for name, values in totals.items()}
    return {"A_" + name: value for name, value in means.items()} | {
        "n_views": float(len(per_view))
    }


def _contrasts(
    metrics: dict[str, float], diff: str, same: str | None, controls: list[str]
) -> dict[str, float | None]:
    """Headline contrasts, with the natural route's missing control marked.

    On the synthetic grid the headline is ``A_diff - A_same``: the same span,
    edited the same way, differing only in whether the replacement preserves the
    queried identity, so low-level perturbation cancels by construction. On the
    natural route that condition does not exist, and the honest report is the
    difference against the available controls plus an explicit flag, not a
    number computed from a substitute and labelled as if it were the same thing.
    """

    a_diff = metrics.get(f"A_{diff}", float("nan"))
    a_same = metrics.get(f"A_{same}", float("nan")) if same else float("nan")
    available_controls = {
        name: metrics.get(f"A_{name}", float("nan"))
        for name in controls
        if f"A_{name}" in metrics
    }
    finite = [value for value in available_controls.values() if value == value]
    out: dict[str, float | None] = {
        "A_diff": a_diff,
        "A_same": a_same if same else None,
        "selectivity": (a_diff - a_same) if same and a_same == a_same else None,
        "specificity": (a_same - available_controls.get("sham", float("nan")))
        if same
        else None,
        "perturbation": max(finite) if finite else float("nan"),
        "controls": available_controls,
    }
    return out


def _bootstrap(
    clusters: dict[str, list[dict[str, bool]]],
    samples: int,
    seed: int,
    diff_condition: str,
    same_condition: str | None,
    controls: list[str],
) -> dict[str, list[float]]:
    """Resample whole clips, so dependent views never enter separately."""

    keys = sorted(clusters)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        picked: list[dict[str, bool]] = []
        for _ in keys:
            picked.extend(clusters[keys[rng.randrange(len(keys))]])
        if not picked:
            continue
        aggregated = _aggregate(picked)
        metrics = aggregated | _contrasts(
            aggregated, diff_condition, same_condition, controls
        )
        for name, value in _flatten(metrics):
            if value == value:  # omit NaN from interval calculations
                draws[name].append(value)
    return {
        name: [_quantile(sorted(values), 0.025), _quantile(sorted(values), 0.975)]
        for name, values in draws.items()
        if values
    }


def _flatten(metrics: dict) -> list[tuple[str, float]]:
    """Numeric entries only, with the control sub-dict lifted to its own names."""

    out: list[tuple[str, float]] = []
    for name, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out.append((name, float(value)))
        elif isinstance(value, dict):
            out.extend((f"{name}_{key}", float(item)) for key, item in value.items()
                       if isinstance(item, (int, float)) and not isinstance(item, bool))
    return out


def _accuracy_by_condition(views: dict[str, dict[str, dict[str, Any]]]) -> dict[str, float]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for conditions in views.values():
        for condition, row in conditions.items():
            totals[condition].append(bool(row.get("correct")))
    return {condition: _mean(values) for condition, values in sorted(totals.items())}


def _class_breakdown(
    views: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for conditions in views.values():
        original = conditions["original"]
        label = str(original.get("query_class", "?"))
        buckets[label]["correct"].append(bool(original.get("correct")))
        change = _view_changes(conditions)
        buckets[label]["diff"].append(change.get("target_replaced_diff", False))
        buckets[label]["same"].append(change.get("target_replaced_same", False))
    out: dict[str, dict[str, float]] = {}
    for label, values in sorted(buckets.items()):
        accuracy = _mean(values["correct"])
        diff = _mean(values["diff"])
        same = _mean(values["same"])
        out[label] = {
            "n_views": float(len(values["correct"])),
            "original_accuracy": accuracy,
            "A_diff": diff,
            "A_same": same,
            "selectivity": diff - same,
        }
    return out


def _gates(accuracy: dict[str, float], metrics: dict[str, float]) -> dict[str, Any]:
    competence_accuracy = accuracy.get("original", float("nan"))
    dependence = (metrics.get("controls") or {}).get("silence", float("nan"))
    sham = (metrics.get("controls") or {}).get("sham", float("nan"))
    return {
        "G_competence": {
            "original_accuracy": competence_accuracy,
            "required": COMPETENCE_ACCURACY,
            "silence_answer_change": dependence,
            "silence_required": COMPETENCE_DEPENDENCE,
            "passed": bool(
                competence_accuracy == competence_accuracy
                and competence_accuracy >= COMPETENCE_ACCURACY
                and dependence == dependence
                and dependence >= COMPETENCE_DEPENDENCE
            ),
        },
        "G_sham": {
            "A_sham": sham,
            "required_max": SHAM_TOLERANCE,
            "passed": bool(sham == sham and sham <= SHAM_TOLERANCE),
        },
    }


def _verdict(report: dict[str, Any]) -> str:
    """ADR-009's three branches, with the fourth state named explicitly."""

    gates = report["gates"]
    if not gates["G_sham"]["passed"]:
        return (
            "INSTRUMENT INVALID: the sham edit moved answers, so the pipeline "
            "perturbs audio and no other number here is interpretable"
        )
    if not gates["G_competence"]["passed"]:
        return (
            "NO CLAIM: the model is not competent on this task, so selectivity "
            "cannot be read (ADR-009 branch 3)"
        )
    metrics = report["metrics"]
    if metrics.get("selectivity") is None:
        return (
            "NO MATCHED SAME-SPAN CONTROL: this route cannot build a "
            "truth-preserving edit of the same span, so selectivity is not "
            "defined; report the diff-versus-control differences instead"
        )
    low, high = report["bootstrap_95"].get("selectivity", [float("nan")] * 2)
    if low > 0:
        return (
            "SELECTIVITY POSITIVE: answers follow the evidence identity more "
            "than a matched in-place rewrite"
        )
    if high < 0:
        return "SELECTIVITY NEGATIVE: the matched rewrite moves answers more"
    return (
        "NO DISSOCIATION ESTABLISHED: selectivity is indistinguishable from "
        "zero at this competence level"
    )


def _mean(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    main()
