#!/usr/bin/env python3
"""Difficulty-knob evidence: does the manipulation move accuracy before selectivity?

WHY THIS EXISTS
---------------
ADR-008 says the manipulation must be shown to move accuracy *before* selectivity
is read across it -- otherwise a flat selectivity curve is uninterpretable: it
could mean "no dissociation" or merely "this knob did not make the task harder".

The README quotes this table, but no script produced it, so the numbers were not
reproducible from the repository. This is that script.

TWO FAMILIES OF INPUT, and they answer different questions:

  * accuracy knobs -- the `k_*` splits (`k_snr5`, `k_snr0`, `k_snrm5` for the SNR
    gradient; `k_n2`, `k_n4` for polyphony). Each carries only ``original`` and
    ``silence``, so they can show that a knob moves *accuracy* but they cannot
    produce a selectivity, which needs the matched diff/same pair.
  * selectivity across SNR -- the full seven-condition `snr*_full` splits, which
    do carry ``target_replaced_diff`` and ``target_replaced_same``. The baseline
    for both is the ``pilot`` split at +10 dB.

So the output has two tables, and neither is derivable from the other. Do not
quote an SNR selectivity from the `k_*` files; they do not contain the contrast.

USAGE
    python scripts/difficulty_knobs.py \
        --pilot outputs/pilot/pilot_qwen3b.jsonl \
        --knobs outputs/round2/r2_qwen3b.jsonl \
        --snr outputs/snr/snr_qwen3b.jsonl \
        --output outputs/round2/difficulty_qwen3b.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ORIG = "original"
DIFF = "target_replaced_diff"
SAME = "target_replaced_same"
SHAM = "sham"

# accuracy-only knobs: split -> human label
KNOBS = [
    ("k_snr5", "SNR +5 dB"),
    ("k_snr0", "SNR 0 dB"),
    ("k_snrm5", "SNR -5 dB"),
    ("k_n2", "polyphony 2"),
    ("k_n4", "polyphony 4"),
]
# selectivity-capable splits
SNR_FULL = [("snr0_full", "SNR 0 dB"), ("snrm5_full", "SNR -5 dB")]


def load(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split and row.get("split") != split:
            continue
        rows.append(row)
    return rows


def accuracy(rows: list[dict[str, Any]], condition: str = ORIG) -> dict[str, Any]:
    sel = [r for r in rows if r["condition"] == condition]
    if not sel:
        return {"n": 0, "accuracy": None}
    hits = sum(1 for r in sel if r.get("correct"))
    return {"n": len(sel), "accuracy": hits / len(sel)}


def views(rows: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    """Group by pair_id: one view per source pair x surface order."""

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["condition"]] = row
    return list(grouped.values())


def selectivity(rows: list[dict[str, Any]], *, samples: int = 3000, seed: int = 20260912) -> dict[str, Any]:
    """P(change|diff) - P(change|same) with a clip-clustered bootstrap interval."""

    usable = [
        v for v in views(rows)
        if all(c in v for c in (ORIG, DIFF, SAME))
    ]
    if not usable:
        return {"n": 0, "selectivity": None, "ci95": None}

    def changed(view, condition):
        before = view[ORIG].get("parsed_choice")
        after = view[condition].get("parsed_choice")
        return before is not None and after is not None and after != before

    def point(subset):
        d = sum(1 for v in subset if changed(v, DIFF)) / len(subset)
        s = sum(1 for v in subset if changed(v, SAME)) / len(subset)
        return d - s

    observed = point(usable)

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in usable:
        by_pair[v[ORIG]["source_pair_id"]].append(v)
    pairs = sorted(by_pair)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        picked = [v for _ in range(len(pairs)) for v in by_pair[pairs[rng.randrange(len(pairs))]]]
        draws.append(point(picked))
    draws.sort()

    acc = sum(1 for v in usable if v[ORIG].get("correct")) / len(usable)
    sham = sum(1 for v in usable if changed(v, SHAM)) / len(usable) if all(
        SHAM in v for v in usable
    ) else None
    return {
        "n": len(usable),
        "n_source_pairs": len(pairs),
        "original_accuracy": acc,
        "A_diff": sum(1 for v in usable if changed(v, DIFF)) / len(usable),
        "A_same": sum(1 for v in usable if changed(v, SAME)) / len(usable),
        "A_sham": sham,
        "selectivity": observed,
        "ci95": [draws[int(0.025 * samples)], draws[int(0.975 * samples)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True,
                        help="+10 dB baseline predictions (seven conditions)")
    parser.add_argument("--knobs", type=Path, required=True,
                        help="predictions containing the k_* accuracy-knob splits")
    parser.add_argument("--snr", type=Path,
                        help="predictions containing the snr*_full selectivity splits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = accuracy(load(args.pilot, "pilot"))
    base_acc = baseline["accuracy"]
    report: dict[str, Any] = {
        "baseline": {"split": "pilot", "label": "+10 dB, 1 event", **baseline},
        "accuracy_knobs": {},
    }

    for split, label in KNOBS:
        rows = load(args.knobs, split)
        block = accuracy(rows)
        block["label"] = label
        block["delta_pt"] = (
            round(100 * (block["accuracy"] - base_acc), 1)
            if block["accuracy"] is not None and base_acc is not None
            else None
        )
        report["accuracy_knobs"][split] = block

    if args.snr:
        report["selectivity_across_snr"] = {}
        for split, label in SNR_FULL:
            block = selectivity(load(args.snr, split))
            block["label"] = label
            report["selectivity_across_snr"][split] = block

    print(f"baseline (pilot, +10 dB): acc={base_acc:.4f} (n={baseline['n']})")
    print()
    print(f"{'knob':16s} {'acc':>7s} {'delta':>9s}  n")
    for split, block in report["accuracy_knobs"].items():
        acc = block["accuracy"]
        if acc is None:
            print(f"{block['label']:16s} {'--':>7s} {'--':>9s}  0")
            continue
        print(f"{block['label']:16s} {acc:7.4f} {block['delta_pt']:+8.1f}pt {block['n']:4d}")

    if "selectivity_across_snr" in report:
        print()
        print("selectivity across SNR (needs the matched diff/same pair):")
        for split, block in report["selectivity_across_snr"].items():
            if block.get("selectivity") is None:
                print(f"  {block['label']:12s} no matched views")
                continue
            lo, hi = block["ci95"]
            print(f"  {block['label']:12s} acc={block['original_accuracy']:.4f} "
                  f"sel={block['selectivity']:+.4f} [{lo:+.3f}, {hi:+.3f}] n={block['n']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
