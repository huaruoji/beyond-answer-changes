#!/usr/bin/env python3
"""Select the class vocabulary a model can actually name.

The design's competence floor requires the model to be able to answer the
question at all. A class it cannot name in this exact question format cannot
carry a causal reading: an item the model answers at chance says nothing about
whether it used the evidence, which is the trap the withdrawn temporal-order
task fell into.

Screening items contain one event and offer the true class against a random
other class, so chance is 0.5. A class survives when the model is reliably above
that. Survivors are frozen to a file before the main grid is generated, so the
main grid cannot be tuned to the classes that happen to look good later.

Also reported, because a high score can come from the wrong place: the share of
answers choosing option A. A model that simply prefers one position gains
nothing here (options are balanced), but a strong skew is worth seeing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CHANCE = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--min-classes", type=int, default=20)
    parser.add_argument("--vocab-out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", default="original")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text().splitlines()
        if line.strip()
    ]
    usable = [
        row
        for row in rows
        if row.get("parsed_choice")
        and "error" not in row
        and row.get("condition") == args.condition
    ]

    per_class: dict[str, dict[str, list]] = defaultdict(
        lambda: {"correct": [], "choices": []}
    )
    for row in usable:
        label = str(row.get("query_class", "?"))
        per_class[label]["correct"].append(bool(row.get("correct")))
        per_class[label]["choices"].append(str(row["parsed_choice"]).upper())

    scored: dict[str, dict[str, float]] = {}
    for label, values in per_class.items():
        n = len(values["correct"])
        accuracy = sum(values["correct"]) / n if n else float("nan")
        a_share = (
            sum(1 for choice in values["choices"] if choice == "A") / n if n else float("nan")
        )
        scored[label] = {
            "n": n,
            "accuracy": accuracy,
            "a_share": a_share,
            "above_chance_p": _binomial_tail(sum(values["correct"]), n, CHANCE),
        }

    evaluated = {
        label: stats
        for label, stats in scored.items()
        if stats["n"] >= args.min_items
    }
    survivors = sorted(
        label
        for label, stats in evaluated.items()
        if stats["accuracy"] >= args.min_accuracy
    )
    dropped_low = sorted(
        label
        for label, stats in evaluated.items()
        if stats["accuracy"] < args.min_accuracy
    )
    dropped_thin = sorted(set(scored) - set(evaluated))

    report = {
        "predictions": str(args.predictions),
        "condition": args.condition,
        "records": len(rows),
        "records_usable": len(usable),
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "min_items": args.min_items,
            "min_classes": args.min_classes,
        },
        "classes_scored": len(scored),
        "classes_evaluated": len(evaluated),
        "survivors": survivors,
        "n_survivors": len(survivors),
        "dropped_below_threshold": dropped_low,
        "dropped_too_few_items": dropped_thin,
        "per_class": dict(sorted(scored.items())),
        "gate_passed": len(survivors) >= args.min_classes,
    }

    text = json.dumps(report, indent=2)
    print(
        json.dumps(
            {key: report[key] for key in (
                "classes_scored",
                "classes_evaluated",
                "n_survivors",
                "gate_passed",
            )},
            indent=2,
        )
    )
    print(f"survivors: {', '.join(survivors[:25])}{' ...' if len(survivors) > 25 else ''}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.vocab_out:
        args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
        args.vocab_out.write_text("\n".join(survivors) + "\n" if survivors else "")
    if not report["gate_passed"]:
        raise SystemExit(
            f"only {len(survivors)} classes survived, need {args.min_classes}"
        )


def _binomial_tail(successes: int, trials: int, probability: float) -> float:
    """P(X >= successes) under Binomial(trials, probability).

    Reported so a high per-class score can be checked against luck rather than
    assumed. Exact, and cheap at these sizes.
    """

    if trials == 0:
        return float("nan")
    total = 0.0
    for k in range(successes, trials + 1):
        total += (
            math.comb(trials, k)
            * probability**k
            * (1 - probability) ** (trials - k)
        )
    return min(1.0, total)


if __name__ == "__main__":
    main()
