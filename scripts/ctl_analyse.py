"""Analyse a CTL pilot: the matched intervention gap, per-cell metrics, and the gates.

The estimand is paired at the item level (docs/protocol-invariants.md, §2.1)::

    m_i  = 1[f(x_i^diff) != f(x_i^orig)] - 1[f(x_i^same) != f(x_i^orig)]
    MIG  = E_i[m_i]

Pairing matters. A difference of two aggregate rates would let item difficulty and
per-item variance leak into the statistic; computing ``m_i`` first and then averaging
cancels the item, and the bootstrap resamples **items**, not recordings.

Also reported, because each answers a different question:

* ``P(change | sham)`` — the editing-pipeline artefact floor. If this is not near zero
  the whole design is uninterpretable, exactly as in the withdrawn build;
* per-cell accuracy, so a MIG can always be read next to the competence that produced
  it (a causal metric near chance is not interpretable);
* the equal-energy contrast: MIG under the low-audibility masker versus the
  high-audibility one. This is the "bonus" criterion — whether audibility *drives*
  behaviour — as opposed to the gates, which only establish that the manipulation is
  real.

Usage::

    python scripts/ctl_analyse.py --dir outputs/ctl/sample30 --predictions preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ("original", "diff", "same", "sham")


def load_predictions(path: Path) -> dict[tuple[str, str, str], str | None]:
    """(source_pair_id, calibration, masker) is not unique per condition, so key by pair_id."""

    out: dict[tuple[str, str, str], str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[(rec["pair_id"], rec["condition"], "")] = rec.get("parsed_choice")
    return out


def cell_key(row: dict) -> str:
    return f"{row['calibration']}__{row['masker']}"


def collect(manifest_path: Path, choices: dict) -> dict:
    """Group choices into per-item records indexed by (cell, condition)."""

    items: dict[str, dict[str, dict[str, str | None]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, dict] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row["item_id"]
        cell = cell_key(row)
        choice = choices.get((row["pair_id"], row["condition"], ""))
        items[item_id][cell][row["condition"]] = choice
        entry = meta.setdefault(
            item_id,
            {
                "class": row["query_class"],
                "donor_class": row["absent_class"],
                "split": row.get("split"),
                "audibility_db": {},
            },
        )
        entry["audibility_db"][cell] = row["audibility_db"]
    return {"items": dict(items), "meta": meta}


def item_stat(record: dict[str, str | None]) -> dict[str, float] | None:
    """Per-item MIG and the raw change indicators for one cell."""

    original = record.get("original")
    if original is None:
        return None
    changed_diff = record.get("diff") is not None and record["diff"] != original
    changed_same = record.get("same") is not None and record["same"] != original
    changed_sham = record.get("sham") is not None and record["sham"] != original
    return {
        "m": float(changed_diff) - float(changed_same),
        "changed_diff": float(changed_diff),
        "changed_same": float(changed_same),
        "changed_sham": float(changed_sham),
        "answered": float(record.get("diff") is not None and record.get("same") is not None),
    }


def bootstrap_ci(values: list[float], *, n: int = 5000, seed: int = 20260916) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [arr[rng.randrange(arr.size)] for _ in range(arr.size)]
        means.append(float(np.mean(sample)))
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def equal_energy_gap(entry: dict) -> float | None:
    """|audibility(low_pass) - audibility(event_matched)| for one item, equal-energy arm."""

    aud = entry.get("audibility_db", {})
    a, b = aud.get("equal_energy__low_pass_400"), aud.get("equal_energy__event_matched")
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def usable_items(collected: dict, min_gap_db: float) -> set[str]:
    """Items whose equal-energy manipulation is big enough to be worth measuring.

    Measured on the 60-item sample, only 58% of items carry a >=10 dB separation and
    the 10th percentile is 1.2 dB. Including near-zero-gap items adds noise without
    adding manipulation, so every headline number is reported for the full set and for
    this subset.
    """

    return {
        item_id
        for item_id, entry in collected["meta"].items()
        if (gap := equal_energy_gap(entry)) is not None and gap >= min_gap_db
    }


def analyse(collected: dict, cells: list[str], *, restrict_to: set[str] | None = None) -> dict:
    items = collected["items"]
    meta = collected["meta"]
    if restrict_to is not None:
        items = {k: v for k, v in items.items() if k in restrict_to}
        meta = {k: v for k, v in meta.items() if k in restrict_to}
    results: dict[str, object] = {"n_items": len(items), "cells": {}}
    for cell in cells:
        stats = [s for item in items.values() if (s := item_stat(item.get(cell, {}))) is not None]
        if not stats:
            continue
        migs = [s["m"] for s in stats]
        mean, lo, hi = bootstrap_ci(migs)
        results["cells"][cell] = {  # type: ignore[index]
            "n_items": len(stats),
            "MIG": mean,
            "MIG_ci95": [lo, hi],
            "P_change_diff": float(np.mean([s["changed_diff"] for s in stats])),
            "P_change_same": float(np.mean([s["changed_same"] for s in stats])),
            "P_change_sham": float(np.mean([s["changed_sham"] for s in stats])),
            "answer_rate": float(np.mean([s["answered"] for s in stats])),
        }

    # per-cell accuracy, computed from the manifest ground truth via choices
    acc: dict[str, list[float]] = defaultdict(list)
    for item in items.values():
        for cell, record in item.items():
            for condition in CONDITIONS:
                choice = record.get(condition)
                if choice is None:
                    continue
                truth = "B" if condition == "diff" else "A"
                acc[f"{cell}__{condition}"].append(float(choice == truth))
    results["accuracy"] = {k: float(np.mean(v)) for k, v in sorted(acc.items())}  # type: ignore[assignment]

    # the bonus criterion: MIG under the low- versus high-audibility masker at equal energy
    ee = results["cells"].get("equal_energy__low_pass_400"), results["cells"].get("equal_energy__event_matched")  # type: ignore[index]
    if all(isinstance(x, dict) for x in ee):
        low, high = ee
        results["equal_energy_mig_contrast"] = {
            "mig_low_audibility_masker": low["MIG"],
            "mig_high_audibility_masker": high["MIG"],
            "difference": low["MIG"] - high["MIG"],
        }
    results["mean_equal_energy_gap_db"] = float(
        np.mean([g for e in meta.values() if (g := equal_energy_gap(e)) is not None] or [float("nan")])
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="built CTL directory")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-audibility-gap", type=float, default=10.0,
                        help="report a second analysis restricted to items at or above this gap")
    args = parser.parse_args()

    manifest = args.dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest}")
        return 2
    choices = load_predictions(args.predictions)
    collected = collect(manifest, choices)
    cells = sorted({cell_key(json.loads(l)) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()})
    results = analyse(collected, cells)
    subset_ids = usable_items(collected, args.min_audibility_gap)
    subset = analyse(collected, cells, restrict_to=subset_ids) if subset_ids else None
    if subset is not None:
        results["subset_analysis"] = {
            "min_audibility_gap_db": args.min_audibility_gap,
            "n_items": subset["n_items"],
            "cells": subset["cells"],
            "equal_energy_mig_contrast": subset.get("equal_energy_mig_contrast"),
        }

    out = args.output or (args.dir / "pilot_analysis.json")
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    print(f"items: {results['n_items']}")
    print(f"{'cell':44} {'n':>4} {'MIG':>8} {'95% CI':>18} {'sham':>7} {'answered':>9}")
    for cell, block in results["cells"].items():  # type: ignore[union-attr]
        lo, hi = block["MIG_ci95"]
        print(f"{cell:44} {block['n_items']:4d} {block['MIG']:+8.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{block['P_change_sham']:7.3f} {block['answer_rate']:9.3f}")
    if "equal_energy_mig_contrast" in results:
        c = results["equal_energy_mig_contrast"]
        print(f"\nequal-energy MIG contrast (bonus criterion):")
        print(f"  low-audibility masker  MIG = {c['mig_low_audibility_masker']:+.3f}")
        print(f"  high-audibility masker MIG = {c['mig_high_audibility_masker']:+.3f}")
        print(f"  difference                 = {c['difference']:+.3f}")
    if subset is not None and subset.get("equal_energy_mig_contrast"):
        c = subset["equal_energy_mig_contrast"]
        print(f"\nrestricted to items with a >= {args.min_audibility_gap:.0f} dB equal-energy gap "
              f"(n={subset['n_items']}):")
        for cell, block in subset["cells"].items():  # type: ignore[union-attr]
            lo, hi = block["MIG_ci95"]
            print(f"  {cell:42} MIG {block['MIG']:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  contrast difference = {c['difference']:+.3f}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
