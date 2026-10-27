"""v3 probe analysis: does the model still know WHICH sound is present?

Reads ``outputs/ctl/v3_probe/probe/{manifest.jsonl,build_audit.json}`` and the probe
prediction files, and writes ``outputs/ctl/v3_probe/probe_results.json``.

The three questions per (item, arm):

* ``2afc``  the existing two-alternative question (A = queried class, B = donor class);
* ``presq`` "Is '<queried class>' present?" (A = yes, B = no);
* ``presa`` "Is '<donor class>' present?"   (A = yes, B = no).

Reading (A) IDENTITY predicts, on the magnitude-matched ``same`` arm (queried class
still present): presq = yes, presa = no, 2afc = A -- i.e. the *same* answer as on
``original``, even though the span was rewritten just as hard as in ``diff``.

Reading (B) DETECTION predicts the model cannot say which class is there: the answer
is driven by the edit, not by the named class, so ``presq`` and ``presa`` should agree
with each other on the same audio, and a magnitude-matched same-class edit should flip
the answer as often as the different-class one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ("original", "diff", "same", "same_loud", "diff_loud")
KINDS = ("2afc", "presq", "presa")
YES = "A"  # option A is 'yes' in both presence questions


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def mcnemar_exact(k_diff_only: int, k_same_only: int) -> float:
    from math import comb

    n = k_diff_only + k_same_only
    if n == 0:
        return 1.0
    k = min(k_diff_only, k_same_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=Path("outputs/ctl/v3_probe/probe"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ctl/v3_probe/probe_results.json"))
    parser.add_argument("--models", default="3b=outputs/ctl/v3_probe/probe_pred_3b.jsonl,"
                                          "7b=outputs/ctl/v3_probe/probe_pred_7b.jsonl")
    args = parser.parse_args()

    manifest = {row["pair_id"]: row for row in load_jsonl(args.probe_dir / "manifest.jsonl")}
    audit = {entry["item_id"]: entry for entry in json.loads((args.probe_dir / "build_audit.json").read_text())["items"]}
    matched_items = {item for item, entry in audit.items() if entry["same_arm_is_magnitude_matched"]}

    models = dict(part.split("=", 1) for part in args.models.split(","))
    out: dict[str, object] = {
        "design": {
            "cell": "equal_energy__low_pass_400",
            "n_items": len(audit),
            "n_items_magnitude_matched_same_arm": len(matched_items),
            "arms": list(ARMS),
            "question_kinds": list(KINDS),
            "presence_answer_encoding": "Option A = 'yes', Option B = 'no' (parser-compatible); the question names one class only",
            "loud_arm_gain_db": 10.0,
        },
        "models": {},
    }

    for name, path in models.items():
        preds = load_jsonl(Path(path))
        answer: dict[tuple[str, str, str], str | None] = {}
        item_of: dict[str, str] = {}
        errors = 0
        for row in preds:
            kind = row["pair_id"].rsplit("__", 1)[1]
            arm = row["condition"]
            item_id = row["source_pair_id"].split("__")[0]
            item_of[row["source_pair_id"]] = item_id
            if row.get("error") or row.get("parsed_choice") is None:
                errors += 1
                continue
            answer[(item_id, arm, kind)] = row["parsed_choice"]

        items = sorted({item for (item, _arm, _kind) in answer})
        block: dict[str, object] = {
            "n_records": len(preds),
            "n_errors_or_unparsed": errors,
            "n_items_scored": len(items),
            "accuracy_by_arm_question": {},
            "presence_by_arm": {},
            "consistency_by_arm": {},
            "mig_by_question": {},
            "magnitude_matched_subset": {},
        }

        # accuracy vs the manifest ground truth
        for arm in ARMS:
            for kind in KINDS:
                hits = total = 0
                for item in items:
                    row = manifest.get(f"{item}__equal_energy__low_pass_400__{arm}__{kind}")
                    if row is None:
                        continue
                    got = answer.get((item, arm, kind))
                    if got is None:
                        continue
                    total += 1
                    hits += got == row["ground_truth"]
                block["accuracy_by_arm_question"][f"{arm}__{kind}"] = {
                    "n": total,
                    "accuracy": hits / total if total else None,
                }

        # presence readouts
        for arm in ARMS:
            yes_q = sum(1 for item in items if answer.get((item, arm, "presq")) == YES)
            no_q = sum(1 for item in items if answer.get((item, arm, "presq")) == "B")
            yes_a = sum(1 for item in items if answer.get((item, arm, "presa")) == YES)
            no_a = sum(1 for item in items if answer.get((item, arm, "presa")) == "B")
            n = len(items)
            block["presence_by_arm"][arm] = {
                "n": n,
                "P_presq_yes_query_class_present": yes_q / n,
                "P_presq_yes_ci95": list(wilson(yes_q, n)),
                "P_presa_yes_donor_class_present": yes_a / n,
                "P_presa_yes_ci95": list(wilson(yes_a, n)),
                "cross_question_same_answer_rate": (sum(
                    1 for item in items
                    if answer.get((item, arm, "presq")) == answer.get((item, arm, "presa"))
                ) / n),
                "identity_pattern_query_yes_donor_no": (
                    sum(1 for item in items if answer.get((item, arm, "presq")) == YES and answer.get((item, arm, "presa")) == "B") / n
                ),
                "swap_pattern_query_no_donor_yes": (
                    sum(1 for item in items if answer.get((item, arm, "presq")) == "B" and answer.get((item, arm, "presa")) == YES) / n
                ),
                "counts": {"presq": dict(Counter(answer.get((item, arm, "presq")) for item in items)),
                           "presa": dict(Counter(answer.get((item, arm, "presa")) for item in items))},
                "no_q_no_a": no_q, "no_a_no_q": no_a,
            }

        # the MIG analogue per question: change relative to the item's own `original` answer
        for kind in KINDS:
            def changed(arm: str, subset: set[str] | None = None) -> tuple[int, int]:
                hits = total = 0
                for item in items:
                    if subset is not None and item not in subset:
                        continue
                    base = answer.get((item, "original", kind))
                    got = answer.get((item, arm, kind))
                    if base is None or got is None:
                        continue
                    total += 1
                    hits += got != base
                return hits, total

            entry = {}
            for arm in ("diff", "same", "same_loud", "diff_loud"):
                hits, total = changed(arm)
                entry[f"P_change_{arm}"] = hits / total if total else None
            d_only, s_only = 0, 0
            for item in items:
                base = answer.get((item, "original", kind))
                d = answer.get((item, "diff", kind))
                s = answer.get((item, "same", kind))
                if None in (base, d, s):
                    continue
                d_only += (d != base) and (s == base)
                s_only += (s != base) and (d == base)
            entry["MIG"] = (
                entry["P_change_diff"] - entry["P_change_same"]
                if entry["P_change_diff"] is not None and entry["P_change_same"] is not None else None
            )
            entry["mcnemar_exact_p"] = mcnemar_exact(d_only, s_only)
            entry["diff_only_changed"] = d_only
            entry["same_only_changed"] = s_only
            entry["MIG_loud"] = (
                entry["P_change_diff_loud"] - entry["P_change_same_loud"]
                if entry["P_change_diff_loud"] is not None and entry["P_change_same_loud"] is not None else None
            )
            block["mig_by_question"][kind] = entry

        # the magnitude-matched subset: same-class edit at least as large as the diff one
        for kind in KINDS:
            hits_d = tot = hits_s = 0
            d_only = s_only = 0
            for item in items:
                if item not in matched_items:
                    continue
                base = answer.get((item, "original", kind))
                d = answer.get((item, "diff", kind))
                s = answer.get((item, "same", kind))
                if None in (base, d, s):
                    continue
                tot += 1
                hits_d += d != base
                hits_s += s != base
                d_only += (d != base) and (s == base)
                s_only += (s != base) and (d == base)
            block["magnitude_matched_subset"][kind] = {
                "n": tot,
                "P_change_diff": hits_d / tot if tot else None,
                "P_change_same": hits_s / tot if tot else None,
                "MIG": (hits_d - hits_s) / tot if tot else None,
                "mcnemar_exact_p": mcnemar_exact(d_only, s_only),
            }
        out["models"][name] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, block in out["models"].items():
        print(f"\n=== model {name}  records={block['n_records']} unparsed={block['n_errors_or_unparsed']} items={block['n_items_scored']}")
        print("accuracy:", {k: round(v['accuracy'], 3) for k, v in block["accuracy_by_arm_question"].items()})
        for arm in ARMS:
            p = block["presence_by_arm"][arm]
            print(f"  {arm:10} presq(yes)={p['P_presq_yes_query_class_present']:.3f} "
                  f"presa(yes)={p['P_presa_yes_donor_class_present']:.3f} "
                  f"same-answer={p['cross_question_same_answer_rate']:.3f} "
                  f"(q=yes,a=no)={p['identity_pattern_query_yes_donor_no']:.3f} "
                  f"(q=no,a=yes)={p['swap_pattern_query_no_donor_yes']:.3f}")
        for kind, entry in block["mig_by_question"].items():
            print(f"  {kind:6} P(diff)={entry['P_change_diff']:.3f} P(same)={entry['P_change_same']:.3f} "
                  f"MIG={entry['MIG']:+.3f} p={entry['mcnemar_exact_p']:.4f} | "
                  f"loud: P(diff)={entry['P_change_diff_loud']:.3f} P(same)={entry['P_change_same_loud']:.3f} "
                  f"MIG_loud={entry['MIG_loud']:+.3f}")
        print("  magnitude-matched subset (same-edit LSD >= diff-edit LSD):")
        for kind, entry in block["magnitude_matched_subset"].items():
            print(f"    {kind:6} n={entry['n']} P(diff)={entry['P_change_diff']:.3f} "
                  f"P(same)={entry['P_change_same']:.3f} MIG={entry['MIG']:+.3f} p={entry['mcnemar_exact_p']:.4f}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
