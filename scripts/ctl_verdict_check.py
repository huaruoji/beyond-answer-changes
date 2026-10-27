"""Check that the pilot's decision rule separates the three possible verdicts.

The pilot is worth GPU time only if its analysis can tell the outcomes apart. This
runs the real pipeline (`ctl_analyse.py`) over synthetic prediction sets built from the
real manifest, each representing one of the three branches of the plan's decision rule:

===========================  ======================================================
``evidence_user``            follows the evidence; expect gates 5+6 and a large MIG
``audio_ignoring``           ignores the audio; expect MIG ~ 0 with gates still passing
``audibility_gated``         follows the evidence only while it is audible; expect a
                             positive equal-energy contrast (the "bonus" criterion)
===========================  ======================================================

Why this exists: the gates establish that the *manipulation* is real, and the contrast
establishes whether the model *responds* to it. Those are different claims, and the
second one is the whole point of the branch. If the harness cannot separate the three
regimes on data whose answer is known, the pilot's verdict would not be readable and
the run should not be launched.

Usage::

    python scripts/ctl_verdict_check.py --dir outputs/ctl/sample60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ctl_analyse import analyse, usable_items  # noqa: E402


def synthesise(rows: list[dict], regime: str, seed: int = 11) -> dict:
    """Predictions for one behavioural regime, keyed by (pair_id, condition)."""

    rng = random.Random(seed)
    choices: dict[tuple[str, str, str], str | None] = {}
    for row in rows:
        truth = row["ground_truth"]
        audibility = float(row["audibility_db"])
        condition = row["condition"]

        if regime == "evidence_user":
            correct = rng.random() < 0.95
        elif regime == "audio_ignoring":
            # answers "A" unless a coin says otherwise: nothing to do with the edit
            choices[(row["pair_id"], condition, "")] = "A" if rng.random() < 0.9 else "B"
            continue
        elif regime == "audibility_gated":
            # tracks evidence only when the queried sound is comfortably audible;
            # applied to `diff` this decides whether the answer follows the removal
            p_track = 0.95 if audibility > -5.0 else 0.10
            follows = rng.random() < p_track
            correct = follows if condition in {"diff", "same"} else rng.random() < 0.95
        else:
            raise ValueError(regime)

        choices[(row["pair_id"], condition, "")] = truth if correct else ("B" if truth == "A" else "A")
    return _collect_from(rows, choices)


def _collect_from(rows: list[dict], choices: dict) -> dict:
    from collections import defaultdict

    items: dict[str, dict[str, dict[str, str | None]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, dict] = {}
    for row in rows:
        cell = f"{row['calibration']}__{row['masker']}"
        items[row["item_id"]][cell][row["condition"]] = choices.get(
            (row["pair_id"], row["condition"], "")
        )
        entry = meta.setdefault(
            row["item_id"], {"class": row["query_class"], "audibility_db": {}}
        )
        entry["audibility_db"][cell] = row["audibility_db"]
    return {"items": dict(items), "meta": meta}


def verdict(result: dict, min_gap_db: float) -> dict:
    """Apply the plan's three-branch reading to one analysis result."""

    cells = result["cells"]
    ee_low = cells.get("equal_energy__low_pass_400")
    ee_high = cells.get("equal_energy__event_matched")
    gates = {
        "5_equal_energy_separates": bool(
            ee_low and ee_high
            and abs(ee_low["P_change_diff"] - ee_high["P_change_diff"]) > 0.0
        ),
        "6_equal_audibility_matches": bool(
            cells.get("equal_audibility__low_pass_400")
            and cells.get("equal_audibility__event_matched")
        ),
    }
    sham = max((b["P_change_sham"] for b in cells.values()), default=float("nan"))
    mig_full = max((b["MIG"] for b in cells.values()), default=float("nan"))

    subset = result.get("subset_analysis") or {}
    contrast = None
    if subset.get("cells") and subset.get("equal_energy_mig_contrast"):
        contrast = subset["equal_energy_mig_contrast"]["difference"]

    if contrast is not None and contrast > 0.15:
        label = "bonus: model responds to audibility"
    elif mig_full > 0.5 and (contrast is None or abs(contrast) <= 0.15):
        label = "pass without bonus: uses evidence, not modulated by audibility"
    else:
        label = "manipulation real but model does not respond"
    return {
        "gates": gates,
        "sham_max_change_rate": sham,
        "mig_max_over_cells": mig_full,
        "subset_contrast_difference": contrast,
        "reading": label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/ctl/sample60"))
    parser.add_argument("--min-audibility-gap", type=float, default=10.0)
    args = parser.parse_args()

    manifest = args.dir / "manifest.jsonl"
    rows = [json.loads(ln) for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cells = sorted({f"{r['calibration']}__{r['masker']}" for r in rows})

    readings = {}
    for regime in ("evidence_user", "audio_ignoring", "audibility_gated"):
        collected = synthesise(rows, regime)
        result = analyse(collected, cells)
        subset = analyse(collected, cells, restrict_to=usable_items(collected, args.min_audibility_gap))
        result["subset_analysis"] = {
            "n_items": subset["n_items"],
            "cells": subset["cells"],
            "equal_energy_mig_contrast": subset.get("equal_energy_mig_contrast"),
        }
        v = verdict(result, args.min_audibility_gap)
        readings[regime] = v
        print(f"\n=== {regime} ===")
        print(f"  MIG (max over cells)        : {v['mig_max_over_cells']:+.3f}")
        print(f"  sham change rate (max)      : {v['sham_max_change_rate']:.3f}")
        contrast = v["subset_contrast_difference"]
        shown = "n/a" if contrast is None else f"{contrast:+.3f}"
        print(f"  equal-energy contrast       : {shown}")
        print(f"  reading                     : {v['reading']}")

    distinct = len({v["reading"] for v in readings.values()})
    print(f"\ndistinct readings: {distinct}/3")
    if distinct < 3:
        print("FAIL: the decision rule cannot separate the regimes; the pilot's verdict "
              "would not be readable")
        return 3
    print("OK: all three branches of the decision rule are distinguishable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
