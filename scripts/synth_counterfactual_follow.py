#!/usr/bin/env python3
"""Capability-conditioned counterfactual following, and the null model it corrects.

WHY THIS EXISTS
---------------
The project's headline metric has been the change-rate selectivity

    selectivity = P(answer changes | different-class replacement)
                - P(answer changes | same-class replacement)

An independent audit pointed out that this quantity is largely produced by the
metric's own construction, not by evidence use. Take a two-alternative question,
two independent inferences, and a model whose accuracy on the queried class is ``a``:

  * ``same``: both readings have ground truth A, so a disagreement happens whenever
    exactly one of the two is wrong:  P(change | same) = 2a(1-a)
  * ``diff``: the truth flips A -> B, so a change happens whenever the model is right
    twice or wrong twice:              P(change | diff) = a^2 + (1-a)^2
  * therefore                           selectivity_null = (2a - 1)^2

At the measured accuracies that null already predicts most of the observed effect
(see ``--null-check``), so the raw change rate cannot be read as evidence use.

WHAT THIS SCRIPT MEASURES, AND WHAT IT IS NOT
---------------------------------------------
``follow(diff)`` is defined as "the answer landed on the donor's class after the
different-class replacement". The manifest sets that condition's ground truth to
the donor class, so it is *identically* the per-condition accuracy:

    follow(diff)    == accuracy(diff)
    preserve(same)  == accuracy(same)

The script asserts this identity at run time and records it in the output under
``metric_identity``. It is kept as a named column because it reads directionally,
but presenting ``follow`` and ``accuracy_diff`` as two independent numbers invites
the reader to treat a renamed accuracy as a new estimand. They are one quantity.
What is genuinely new here is (a) not using the raw change rate, and (b)
conditioning on demonstrated capability (E1/E2/E3).

AGGREGATION
-----------
The manifest renders every source pair twice, once per ``surface_order`` (A/B
counterbalanced). An earlier version of this script keyed views by
``source_pair_id``, which silently kept only whichever order was read last: it
discarded half the rows and selected an order arbitrarily -- and in these runs,
the flattering one. This version keys by ``pair_id`` and reports four aggregations:

    pooled_both_orders      every view, both orders
    order0 / order1         one surface order each
    single_order_balanced   one order per source pair, exactly balanced

``single_order_balanced`` avoids the double-counting: the two orders of a pair
share the same waveform, so pooling them treats one clip as two independent
observations. Note what it does *not* do. It is balanced in count (300/300 on the
600 confirm pairs), but it still samples the order effect once per pair rather
than averaging it out, so it carries a residual bias of up to 0.66 pt against a
per-pair mean over both orders, with inconsistent sign. It is therefore a
balanced one-order-per-pair sample, not an order-effect-free estimator; the
per-pair mean is the estimator that removes the order effect. All confidence
intervals resample *source pairs* rather than views, which is what keeps the
dependence from inflating them.

NULL MODEL
----------
Two variants are reported, because they do not agree and the disagreement is
informative:

  * ``single_a``: uses ``a = accuracy(original)`` for both conditions. This is the
    textbook version but it assumes the model is equally accurate in all
    conditions, which these models are not (post-edit accuracy is several points
    higher than original accuracy for the Qwen models).
  * ``per_condition``: uses each condition's own accuracy. This is the better
    specified null, and it absorbs more of the observed selectivity.

The independence premise is also tested directly: under the null, the original and
same-class readings are independent draws, so their agreement rate should equal
the product of their marginals. ``independence_diagnostic`` reports observed vs
predicted agreement **in class-correctness space**, which is the space the null
is stated in. It does not hold (models agree far more than chance), so
``2a(1-a)`` is an upper bound on P(change|same), not a calibrated prediction. Both
facts are reported rather than smoothed over.

Do not compute this diagnostic in option-letter space. Because order 0 always
puts the queried class at A and order 1 always puts it at B, pooled letter
marginals sit near 0.5 and the predicted agreement collapses toward chance,
inflating the apparent violation roughly threefold. The letter-space figures are
still recorded, explicitly labelled, for continuity with earlier reports.

USAGE
    python scripts/synth_counterfactual_follow.py \
        outputs/round2/r2_qwen3b.jsonl outputs/round2/r2_qwen7b.jsonl \
        --output outputs/round2/counterfactual_follow.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DIFF = "target_replaced_diff"
SAME = "target_replaced_same"
ORIG = "original"
SHAM = "sham"
NEEDED = (ORIG, DIFF, SAME, SHAM)

SUBSET_ORDER = ("all", "E1_orig_correct", "E2_orig_and_same", "E3_all_three")


def load_views(path: Path, split: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Group rows by ``pair_id``: one view per (source pair x surface order).

    Keyed by ``pair_id``, NOT ``source_pair_id``. The manifest renders each source
    pair under both option orders; keying by the source pair makes the second order
    overwrite the first, which loses half the data and picks an order arbitrarily.
    """

    views: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split and row.get("split") != split:
            continue
        views[row["pair_id"]][row["condition"]] = row
    return {
        view: conds
        for view, conds in views.items()
        if all(c in conds for c in NEEDED)
    }


def item_record(conds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-view booleans: correctness at each condition, and did it follow the edit."""

    orig = conds[ORIG]
    diff = conds[DIFF]
    same = conds[SAME]
    sham = conds[SHAM]

    donor_class = diff.get("absent_class")
    query_class = orig.get("query_class")

    def choice(row: dict[str, Any]) -> str | None:
        return row.get("parsed_choice")

    def option_of(row: dict[str, Any], klass: str | None) -> str | None:
        """Which option letter names ``klass`` for this row (A/B is counterbalanced)."""

        if klass is None:
            return None
        for letter, name in (row.get("option_events") or {}).items():
            if name == klass:
                return letter
        return None

    orig_choice = choice(orig)
    diff_choice = choice(diff)
    same_choice = choice(same)
    sham_choice = choice(sham)

    # Directional outcomes. Because the diff row's ground truth IS the donor class,
    # follow_diff is identically correct_diff; likewise preserve_same/correct_same.
    follow = diff_choice is not None and diff_choice == option_of(diff, donor_class)
    preserve = same_choice is not None and same_choice == option_of(same, query_class)

    return {
        "view_id": orig.get("pair_id"),
        "source_pair_id": orig.get("source_pair_id"),
        "surface_order": orig.get("surface_order"),
        "class": query_class,
        "donor_class": donor_class,
        "choice_original": orig_choice,
        "choice_diff": diff_choice,
        "choice_same": same_choice,
        "correct_original": bool(orig.get("correct")),
        "correct_diff": bool(diff.get("correct")),
        "correct_same": bool(same.get("correct")),
        "correct_sham": bool(sham.get("correct")),
        "changed_diff": orig_choice is not None and diff_choice != orig_choice,
        "changed_same": orig_choice is not None and same_choice != orig_choice,
        "changed_sham": orig_choice is not None and sham_choice != orig_choice,
        "follow_diff": bool(follow),
        "preserve_same": bool(preserve),
    }


def rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if r[key]) / len(records)


def clustered_ci(
    records: list[dict[str, Any]],
    key: str,
    *,
    samples: int = 2000,
    seed: int = 20260918,
) -> tuple[float, float]:
    """Bootstrap CI resampling *source pairs*, not views.

    The two surface orders of a pair share a waveform, so views are not independent;
    resampling pairs keeps the dependence intact.
    """

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_pair[r["source_pair_id"]].append(r)
    if not by_pair:
        return (float("nan"), float("nan"))

    pairs = sorted(by_pair)
    sums = [sum(1 for r in by_pair[p] if r[key]) for p in pairs]
    sizes = [len(by_pair[p]) for p in pairs]
    n = len(pairs)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        num = 0
        den = 0
        for _ in range(n):
            i = rng.randrange(n)
            num += sums[i]
            den += sizes[i]
        means.append(num / den)
    means.sort()
    return (means[int(0.025 * samples)], means[int(0.975 * samples)])


def subsets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    e1 = [r for r in records if r["correct_original"]]
    e2 = [r for r in e1 if r["correct_same"]]
    e3 = [r for r in e2 if r["correct_sham"]]
    return {"all": records, "E1_orig_correct": e1, "E2_orig_and_same": e2, "E3_all_three": e3}


def null_prediction(accuracy: float) -> float:
    """(2a-1)^2: the selectivity a two-alternative model with NO evidence use produces."""

    return (2.0 * accuracy - 1.0) ** 2


def independence_diagnostic(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Under the null, original and same-class readings are independent draws.

    Compares observed agreement against the product of the marginals. If the model
    agrees far more often than independence predicts, ``2a(1-a)`` overstates the
    change rate the null can generate.

    This is computed in **class-correctness space**, because that is the space the
    null model is stated in: ``a`` is an accuracy. An earlier revision compared
    *option-letter* agreement instead. That is a different statistic and it
    overstates the violation roughly threefold on the pooled views, because the
    two surface orders swap which letter names the queried class (order 0 always
    puts it at A, order 1 always at B). Pooled letter marginals are therefore
    ~0.50 by construction, so the "predicted" agreement collapses toward chance
    even when the readings are strongly correlated. Letter agreement is still
    reported, explicitly labelled, and within a single order the two coincide.
    """

    n = len(records)
    if not n:
        return {"n": 0}

    # class-correctness space: the space the (2a-1)^2 null is derived in
    both_correct = sum(1 for r in records if r["correct_original"] and r["correct_same"])
    both_wrong = sum(
        1 for r in records if not r["correct_original"] and not r["correct_same"]
    )
    class_agree = (both_correct + both_wrong) / n
    a_orig = sum(1 for r in records if r["correct_original"]) / n
    a_same = sum(1 for r in records if r["correct_same"]) / n
    class_predicted = a_orig * a_same + (1 - a_orig) * (1 - a_same)

    # option-letter space, kept only for continuity with earlier reports
    tab: Counter[tuple[str, str]] = Counter()
    for r in records:
        o, s = r["choice_original"], r["choice_same"]
        if o and s:
            tab[(o, s)] += 1
    m = sum(tab.values())
    letter_obs = letter_pred = float("nan")
    if m:
        letter_obs = (tab[("A", "A")] + tab[("B", "B")]) / m
        p_a_orig = sum(v for (o, _), v in tab.items() if o == "A") / m
        p_a_same = sum(v for (_, s), v in tab.items() if s == "A") / m
        letter_pred = p_a_orig * p_a_same + (1 - p_a_orig) * (1 - p_a_same)

    return {
        "n": n,
        "space": "class_correctness",
        "observed_agreement": class_agree,
        "independence_predicted_agreement": class_predicted,
        "gap": class_agree - class_predicted,
        "independent": abs(class_agree - class_predicted) < 0.03,
        "letter_space_agreement": letter_obs,
        "letter_space_predicted_agreement": letter_pred,
        "letter_space_note": (
            "Letter agreement is not the null's space: the two surface orders swap "
            "which letter names the queried class, so pooled letter marginals sit "
            "near 0.5 and the predicted value collapses toward chance. Quote the "
            "class-space gap, not this one."
        ),
        "note": (
            "Agreement far above the independence prediction means the two readings "
            "are positively correlated, so 2a(1-a) is an upper bound on "
            "P(change|same), not a calibrated null. The 0.03 threshold is a "
            "convention, not a test."
        ),
    }


def analyse(records: list[dict[str, Any]], *, samples: int = 2000, seed: int = 20260918) -> dict[str, Any]:
    n = len(records)
    acc_orig = rate(records, "correct_original")
    acc_same = rate(records, "correct_same")
    acc_diff = rate(records, "correct_diff")
    changed_diff = rate(records, "changed_diff")
    changed_same = rate(records, "changed_same")
    observed = changed_diff - changed_same

    null_single = null_prediction(acc_orig)
    null_per_condition = (acc_diff**2 + (1 - acc_diff) ** 2) - 2 * acc_same * (1 - acc_same)

    out: dict[str, Any] = {
        "n_views": n,
        "n_source_pairs": len({r["source_pair_id"] for r in records}),
        "accuracy_original": acc_orig,
        "raw_change_rate": {
            "P_change_diff": changed_diff,
            "P_change_same": changed_same,
            "selectivity": observed,
        },
        "sham_floor": {
            "P_change_sham": rate(records, "changed_sham"),
            "note": "no-op rewrite; a deterministic model should not move here",
        },
        "null_check": {
            "single_a": {
                "a": acc_orig,
                "selectivity_null_2a_minus_1_squared": null_single,
                "fraction_of_observed_explained": null_single / observed if observed else float("nan"),
            },
            "per_condition": {
                "a_same": acc_same,
                "a_diff": acc_diff,
                "P_change_same_null_2a1_minus_a": 2 * acc_same * (1 - acc_same),
                "P_change_diff_null_a2_plus_1_minus_a2": acc_diff**2 + (1 - acc_diff) ** 2,
                "selectivity_null": null_per_condition,
                "fraction_of_observed_explained": (
                    null_per_condition / observed if observed else float("nan")
                ),
            },
            "observed_selectivity": observed,
        },
        "independence_diagnostic": independence_diagnostic(records),
    }

    # Identity is checked PER VIEW, not on the aggregate: two views that violate
    # it in opposite directions would cancel in a rate and pass unnoticed.
    follow_violations = [r for r in records if r["follow_diff"] != r["correct_diff"]]
    preserve_violations = [r for r in records if r["preserve_same"] != r["correct_same"]]

    for name, sub in subsets(records).items():
        f = rate(sub, "follow_diff")
        p = rate(sub, "preserve_same")
        ad = rate(sub, "correct_diff")
        as_ = rate(sub, "correct_same")
        lo_f, hi_f = clustered_ci(sub, "follow_diff", samples=samples, seed=seed)
        lo_p, hi_p = clustered_ci(sub, "preserve_same", samples=samples, seed=seed)
        out[name] = {
            "n": len(sub),
            "follow_diff": {"rate": f, "ci95": [lo_f, hi_f]},
            "preserve_same": {"rate": p, "ci95": [lo_p, hi_p]},
            "accuracy_diff": ad,
            "accuracy_same": as_,
            "accuracy_original": rate(sub, "correct_original"),
        }

    out["metric_identity"] = {
        "follow_diff_equals_accuracy_diff": not follow_violations,
        "preserve_same_equals_accuracy_same": not preserve_violations,
        "holds": not follow_violations and not preserve_violations,
        "n_follow_violations": len(follow_violations),
        "n_preserve_violations": len(preserve_violations),
        "checked": "per view, not on the aggregate rate",
        "explanation": (
            "The different-class condition's ground truth is the donor class, so "
            "'did the answer follow the edit' and 'was the answer correct' are the "
            "same event. follow/preserve are directional names for per-condition "
            "accuracy, not an independent estimand."
        ),
    }

    out["notes"] = {
        "preserve_same_on_E2_E3": (
            "1.000 by construction: E2/E3 require the same-class condition to be "
            "correct, and its correct answer is the original class. Do not read it "
            "as a finding."
        ),
        "informative_on_conditioned_subsets": "follow_diff (== accuracy_diff)",
    }

    e2 = subsets(records)["E2_orig_and_same"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in e2:
        by_class[r["class"]].append(r)
    out["per_class_E2"] = {
        klass: {
            "n": len(rs),
            "follow_diff": rate(rs, "follow_diff"),
            "preserve_same": rate(rs, "preserve_same"),
        }
        for klass, rs in sorted(by_class.items())
        if len(rs) >= 5
    }
    return out


def order_assignment(pair_ids: list[str]) -> dict[str, int]:
    """Assign each source pair one surface order, exactly balanced.

    Hashing each pair independently and taking the parity gives an *unbalanced*
    split in practice (274/326 on the 600 confirm pairs, a 52-pair skew), which
    would put the order effect back into the "balanced" aggregate. Sorting the
    pairs by a stable hash and then alternating makes the split balanced by
    construction while staying deterministic across runs and independent of the
    pair's identity.
    """

    ordered = sorted(pair_ids, key=lambda p: (hashlib.sha256(p.encode()).hexdigest(), p))
    return {pair: index % 2 for index, pair in enumerate(ordered)}


def aggregations(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pair_ids = sorted({r["source_pair_id"] for r in records})
    assignment = order_assignment(pair_ids)
    return {
        "pooled_both_orders": records,
        "order0": [r for r in records if r["surface_order"] == 0],
        "order1": [r for r in records if r["surface_order"] == 1],
        "single_order_balanced": [
            r
            for r in records
            if r["surface_order"] == assignment[r["source_pair_id"]]
        ],
    }


def pair_mean_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average within a source pair first, then over pairs.

    This is the estimator that actually removes the surface-order effect: both
    orders of a pair contribute, so the order contrast is averaged out rather
    than sampled once. ``single_order_balanced`` only avoids double-counting; it
    still samples one order per pair and therefore carries a residual bias of up
    to 0.66 pt against this estimator.

    Subsets are still defined per view (a view qualifies or not), then averaged
    within its pair; a pair with no qualifying view drops out of that subset.
    """

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_pair[r["source_pair_id"]].append(r)

    def subset_cond(name: str):
        if name == "all":
            return lambda r: True
        if name == "E1_orig_correct":
            return lambda r: r["correct_original"]
        if name == "E2_orig_and_same":
            return lambda r: r["correct_original"] and r["correct_same"]
        return lambda r: r["correct_original"] and r["correct_same"] and r["correct_sham"]

    out: dict[str, Any] = {"n_source_pairs": len(by_pair), "estimator": "pair mean"}
    for name in SUBSET_ORDER:
        cond = subset_cond(name)
        per_pair_follow: list[float] = []
        per_pair_preserve: list[float] = []
        n_views = 0
        for views in by_pair.values():
            qualifying = [r for r in views if cond(r)]
            if not qualifying:
                continue
            n_views += len(qualifying)
            per_pair_follow.append(rate(qualifying, "follow_diff"))
            per_pair_preserve.append(rate(qualifying, "preserve_same"))
        if not per_pair_follow:
            out[name] = {"n_source_pairs": 0, "n_views": 0}
            continue
        out[name] = {
            "n_source_pairs": len(per_pair_follow),
            "n_views": n_views,
            "follow_diff": sum(per_pair_follow) / len(per_pair_follow),
            "preserve_same": sum(per_pair_preserve) / len(per_pair_preserve),
        }
    return out


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so the output is strict JSON.

    An empty subset yields ``float("nan")`` rates. ``json.dumps`` would emit the
    bare token ``NaN``, which is valid Python but rejected by strict JSON parsers
    (``jq``, JavaScript's ``JSON.parse``, most other languages). Emitting ``null``
    keeps the artifact machine-readable everywhere.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--split", default="confirm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help=(
            "Model key to record for each input, in the same order. Defaults to "
            "the input filename stem, which embeds the run name (e.g. "
            "'r2v2_qwen3b') rather than the model identity. Pass explicit keys "
            "when the artifact is consumed by another script that keys models "
            "by identity (see consolidate_evidence.py MODELS)."
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

    report: dict[str, Any] = {"split": args.split, "models": {}}
    for path, key in zip(args.predictions, keys):
        views = load_views(path, args.split)
        records = [item_record(c) for c in views.values()]
        if not records:
            print(f"no usable views in {path}")
            continue

        aggs = aggregations(records)
        block: dict[str, Any] = {
            "n_views": len(records),
            "n_source_pairs": len({r["source_pair_id"] for r in records}),
            "by_aggregation": {
                name: analyse(rs, samples=args.samples)
                for name, rs in aggs.items()
                if rs
            },
            "pair_mean": pair_mean_analysis(records),
        }
        report["models"][key] = block

        for name, sub in block["by_aggregation"].items():
            if not sub["metric_identity"]["holds"]:
                raise SystemExit(
                    f"metric identity violated for {key}/{name}: "
                    "follow_diff != accuracy_diff"
                )

        print(f"\n===== {key}  ({block['n_views']} views, "
              f"{block['n_source_pairs']} source pairs) =====")
        for name, sub in block["by_aggregation"].items():
            raw = sub["raw_change_rate"]
            null = sub["null_check"]
            print(f"  [{name}]  n={sub['n_views']}")
            print(f"    raw change rate : P(diff)={raw['P_change_diff']:.3f}  "
                  f"P(same)={raw['P_change_same']:.3f}  selectivity={raw['selectivity']:+.3f}"
                  f"   sham={sub['sham_floor']['P_change_sham']:.4f}")
            print(f"    null (single a={null['single_a']['a']:.3f}) : "
                  f"{null['single_a']['selectivity_null_2a_minus_1_squared']:.3f}  "
                  f"= {100 * null['single_a']['fraction_of_observed_explained']:.0f}% of observed")
            print(f"    null (per-condition)   : "
                  f"{null['per_condition']['selectivity_null']:.3f}  "
                  f"= {100 * null['per_condition']['fraction_of_observed_explained']:.0f}% of observed")
            ind = sub["independence_diagnostic"]
            if ind.get("n"):
                print(f"    independence    : agreement {ind['observed_agreement']:.3f} vs "
                      f"predicted {ind['independence_predicted_agreement']:.3f} "
                      f"-> {'independent' if ind['independent'] else 'NOT independent'}")
            print(f"    {'subset':22} {'n':>5} {'follow(diff)':>13} {'preserve(same)':>15}")
            for sname in SUBSET_ORDER:
                b = sub[sname]
                note = "" if sname == "all" else "  (preserve=1.0 by definition)"
                print(f"    {sname:22} {b['n']:5d} "
                      f"{b['follow_diff']['rate']:13.3f} {b['preserve_same']['rate']:15.3f}{note}")

        pm = block["pair_mean"]
        print("  [pair_mean]  the estimator that removes the order effect")
        for sname in SUBSET_ORDER:
            b = pm[sname]
            if not b.get("n_source_pairs"):
                continue
            note = "" if sname == "all" else "  (preserve=1.0 by definition)"
            print(f"    {sname:22} {b['n_source_pairs']:5d} "
                  f"{b['follow_diff']:13.3f} {b['preserve_same']:15.3f}{note}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report["bootstrap"] = {
            "samples": args.samples,
            "seed": 20260918,
            "unit": "source pair (cluster bootstrap, ratio estimator)",
            "note": (
                "seed is the analyse() default; recorded here so the intervals in "
                "this artifact are reproducible without reading the source"
            ),
        }
        args.output.write_text(
            json.dumps(_json_safe(report), indent=2, default=float, allow_nan=False),
            encoding="utf-8",
        )
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
