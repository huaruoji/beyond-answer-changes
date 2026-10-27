"""v3 probe step 1: complete arithmetic of the corrected sample60_v2 run.

Reads only ``outputs/ctl/sample60_v2/{manifest.jsonl,pred_v2_3b.jsonl,pred_v2_7b.jsonl}``
and writes ``outputs/ctl/v3_probe/arithmetic.json`` with, for every model x cell:

* P(change | diff), P(change | same), P(change | sham) with the item's own
  ``original`` answer as the anchor (paired, exactly ctl_analyse's definition);
* accuracy per condition scored against the manifest ``ground_truth``;
* the full paired (original, diff, same, sham) answer contingency table;
* MIG with a 5000-draw item bootstrap CI;
* the exact McNemar-style split of the diff/same discordant pairs;
* the per-item m_i = 1[changed|diff] - 1[changed|same] distribution (+1/0/-1).

Nothing here is recomputed from audio and no existing script is modified.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("original", "diff", "same", "sham")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260917


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cell_of(source_pair_id: str) -> str:
    """'<item>__<calibration>__<masker>' -> '<calibration>__<masker>'."""

    parts = source_pair_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"unexpected source_pair_id {source_pair_id!r}")
    return "__".join(parts[-2:])


def collect(manifest_path: Path, pred_path: Path) -> dict:
    manifest = {row["pair_id"]: row for row in load_jsonl(manifest_path)}
    preds = load_jsonl(pred_path)

    cells: dict[str, dict[str, dict[str, str]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    truth: dict[str, dict[str, str]] = collections.defaultdict(dict)
    answers: dict[str, dict[str, str]] = collections.defaultdict(dict)
    failures = 0
    for row in preds:
        choice = row.get("parsed_choice")
        if row.get("error") or choice is None:
            failures += 1
            continue
        cell = cell_of(row["source_pair_id"])
        cells[cell][row["source_pair_id"]][row["condition"]] = choice
        base = manifest[row["pair_id"]]
        truth[cell][f"{row['source_pair_id']}__{row['condition']}"] = base["ground_truth"]
        answers[row["source_pair_id"]] = {}  # placeholder, unused
    return {"cells": cells, "truth": truth, "failures": failures, "n_records": len(preds)}


def bootstrap_ci(values: list[float], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return None, None, None
    idx = rng.integers(0, n, size=(draws, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign test p-value for k successes in n discordant pairs."""

    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def analyse_cell(records: dict[str, dict[str, str]], truth: dict[str, str]) -> dict:
    items = sorted(records)
    n = len(items)
    per_item = []
    for item in items:
        rec = records[item]
        if any(c not in rec for c in CONDITIONS):
            continue
        original = rec["original"]
        changed = {c: rec[c] != original for c in ("diff", "same", "sham")}
        correct = {c: rec[c] == truth[f"{item}__{c}"] for c in CONDITIONS}
        per_item.append(
            {
                "item": item,
                "answers": {c: rec[c] for c in CONDITIONS},
                "changed": {c: bool(changed[c]) for c in ("diff", "same", "sham")},
                "correct": {c: bool(correct[c]) for c in CONDITIONS},
                "m": float(changed["diff"]) - float(changed["same"]),
            }
        )
    if not per_item:
        return {"n_items": 0}

    migs = [row["m"] for row in per_item]
    mig, lo, hi = bootstrap_ci(migs)
    table = collections.Counter(
        tuple(row["answers"][c] for c in CONDITIONS) for row in per_item
    )
    discordant = {
        "changed_diff_only": sum(1 for r in per_item if r["changed"]["diff"] and not r["changed"]["same"]),
        "changed_same_only": sum(1 for r in per_item if r["changed"]["same"] and not r["changed"]["diff"]),
        "changed_both": sum(1 for r in per_item if r["changed"]["diff"] and r["changed"]["same"]),
        "changed_neither": sum(1 for r in per_item if not r["changed"]["diff"] and not r["changed"]["same"]),
    }
    p_change = {c: float(np.mean([r["changed"][c] for r in per_item])) for c in ("diff", "same", "sham")}
    accuracy = {c: float(np.mean([r["correct"][c] for r in per_item])) for c in CONDITIONS}
    answer_counts = {
        c: dict(collections.Counter(r["answers"][c] for r in per_item)) for c in CONDITIONS
    }
    m_dist = dict(collections.Counter(int(r["m"]) for r in per_item))
    return {
        "n_items": n,
        "MIG": mig,
        "MIG_ci95": [lo, hi],
        "P_change": p_change,
        "P_change_diff_minus_same_exact_p": binomial_two_sided(
            discordant["changed_diff_only"], discordant["changed_diff_only"] + discordant["changed_same_only"]
        ),
        "accuracy_vs_ground_truth": accuracy,
        "diff_same_discordance": discordant,
        "m_distribution": {"plus1": m_dist.get(1, 0), "zero": m_dist.get(0, 0), "minus1": m_dist.get(-1, 0)},
        "contingency_original_diff_same_sham": {
            "".join(k): v for k, v in sorted(table.items())
        },
        "answer_counts": answer_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/ctl/sample60_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ctl/v3_probe/arithmetic.json"))
    args = parser.parse_args()

    models = {"3b": args.run_dir / "pred_v2_3b.jsonl", "7b": args.run_dir / "pred_v2_7b.jsonl"}
    out: dict[str, object] = {
        "source": {
            "manifest": str(args.run_dir / "manifest.jsonl"),
            "predictions": {k: str(v) for k, v in models.items()},
            "n_expected_records_per_model": 960,
            "MIG_definition": "mean_i( 1[ans_diff_i != ans_original_i] - 1[ans_same_i != ans_original_i] )",
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "item", "ci": "percentile 2.5/97.5"},
        },
        "models": {},
    }
    for name, path in models.items():
        collected = collect(args.run_dir / "manifest.jsonl", path)
        result = {
            "n_records": collected["n_records"],
            "n_errors_or_unparsed": collected["failures"],
            "cells": {},
        }
        for cell in sorted(collected["cells"]):
            result["cells"][cell] = analyse_cell(collected["cells"][cell], collected["truth"][cell])
        out["models"][name] = result
        print(f"\n=== {name}: {collected['n_records']} records, {collected['failures']} errors/unparsed")
        print(f"{'cell':34} {'n':>3} {'MIG':>7} {'95% CI':>16} {'P(diff)':>8} {'P(same)':>8} {'P(sham)':>8} "
              f"{'+1':>3} {'0':>3} {'-1':>3}")
        for cell, block in result["cells"].items():
            if not block.get("n_items"):
                continue
            lo, hi = block["MIG_ci95"]
            md = block["m_distribution"]
            print(
                f"{cell:34} {block['n_items']:3d} {block['MIG']:+7.3f} [{lo:+.3f},{hi:+.3f}] "
                f"{block['P_change']['diff']:8.3f} {block['P_change']['same']:8.3f} {block['P_change']['sham']:8.3f} "
                f"{md['plus1']:3d} {md['zero']:3d} {md['minus1']:3d}"
            )
        print(f"{'cell':34} {'acc orig':>9} {'acc diff':>9} {'acc same':>9} {'acc sham':>9}")
        for cell, block in result["cells"].items():
            if not block.get("n_items"):
                continue
            a = block["accuracy_vs_ground_truth"]
            print(f"{cell:34} {a['original']:9.3f} {a['diff']:9.3f} {a['same']:9.3f} {a['sham']:9.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
