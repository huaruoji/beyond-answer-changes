"""Regression tests for the counterfactual-follow analysis.

Three defects motivated these tests, all found by independent audit:

1. ``load_items`` keyed views by ``source_pair_id`` while the manifest renders each
   source pair under both ``surface_order`` values. The second order overwrote the
   first, silently discarding half the rows and leaving whichever order happened to
   be read last -- which was the more favourable one. The loader must key by
   ``pair_id`` and keep every view.

2. ``follow(diff)`` and ``accuracy(diff)`` are the same event, because the
   different-class condition's ground truth is the donor class. They were printed
   as two columns, which reads as two findings. The identity is now asserted.

3. The null model's independence premise was never checked. Under the null the
   original and same-class readings are independent draws; if the observed
   agreement is far above the independence prediction, ``2a(1-a)`` is an upper
   bound rather than a calibrated prediction. The diagnostic must be present.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "synth_counterfactual_follow.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("synth_counterfactual_follow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _row(pair_id, source_pair_id, order, condition, choice, *, donor="B", query="A",
         events=None, correct=None):
    events = events or {"A": query, "B": donor}
    if correct is None:
        correct = choice == ("B" if condition == "target_replaced_diff" else "A")
    return {
        "pair_id": pair_id,
        "source_pair_id": source_pair_id,
        "surface_order": order,
        "condition": condition,
        "parsed_choice": choice,
        "correct": correct,
        "option_events": events,
        "query_class": query,
        "absent_class": donor,
        "split": "confirm",
    }


def _write_manifest(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_loader_keeps_both_surface_orders(mod, tmp_path):
    """Both orders of a source pair must survive as separate views."""

    rows = []
    for order, choice in ((0, "A"), (1, "B")):
        pid = f"p__o{order}"
        for cond in ("original", "target_replaced_diff", "target_replaced_same", "sham"):
            rows.append(_row(pid, "p", order, cond, choice))
    path = tmp_path / "preds.jsonl"
    _write_manifest(path, rows)

    views = mod.load_views(path, "confirm")
    assert len(views) == 2, "the second surface order overwrote the first"
    orders = sorted(v["original"]["surface_order"] for v in views.values())
    assert orders == [0, 1]


def test_loader_does_not_drop_rows(mod, tmp_path):
    """N source pairs x 2 orders x 4 conditions must yield 2N views, not N."""

    rows = []
    for pair in range(5):
        for order in (0, 1):
            pid = f"pair{pair}__o{order}"
            for cond in ("original", "target_replaced_diff", "target_replaced_same", "sham"):
                rows.append(_row(pid, f"pair{pair}", order, cond, "A"))
    path = tmp_path / "preds.jsonl"
    _write_manifest(path, rows)

    views = mod.load_views(path, "confirm")
    assert len(views) == 10


def test_follow_diff_is_identically_accuracy_diff(mod):
    """The two names denote one event; the analysis must record the identity."""

    records = []
    for i in range(40):
        good = i % 2 == 0
        records.append({
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": i % 2,
            "class": "A", "donor_class": "B",
            "choice_original": "A", "choice_diff": "B" if good else "A",
            "choice_same": "A",
            "correct_original": True, "correct_diff": good,
            "correct_same": True, "correct_sham": True,
            "changed_diff": good, "changed_same": False, "changed_sham": False,
            "follow_diff": good, "preserve_same": True,
        })
    out = mod.analyse(records, samples=50)
    assert out["metric_identity"]["holds"] is True
    assert out["metric_identity"]["follow_diff_equals_accuracy_diff"] is True
    for name in mod.SUBSET_ORDER:
        sub = out[name]
        assert sub["follow_diff"]["rate"] == pytest.approx(sub["accuracy_diff"])
        assert sub["preserve_same"]["rate"] == pytest.approx(sub["accuracy_same"])


def test_analysis_flags_broken_identity(mod):
    """If the two ever diverge, the analysis must say so rather than hide it."""

    records = []
    for i in range(10):
        records.append({
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": 0,
            "class": "A", "donor_class": "B",
            "choice_original": "A", "choice_diff": "B", "choice_same": "A",
            "correct_original": True,
            "correct_diff": False,          # deliberately inconsistent with follow
            "correct_same": True, "correct_sham": True,
            "changed_diff": True, "changed_same": False, "changed_sham": False,
            "follow_diff": True, "preserve_same": True,
        })
    out = mod.analyse(records, samples=20)
    assert out["metric_identity"]["holds"] is False
    assert out["metric_identity"]["n_follow_violations"] == 10


def test_identity_check_is_per_view_not_aggregate(mod):
    """Compensating violations must not cancel out.

    Two views that violate the identity in opposite directions leave the
    aggregate follow rate equal to the aggregate accuracy, so a rate-level check
    would report the identity as holding. The check is per view.
    """

    def rec(i, follow, correct):
        return {
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": 0,
            "class": "A", "donor_class": "B",
            "choice_original": "A", "choice_diff": "B", "choice_same": "A",
            "correct_original": True, "correct_diff": correct,
            "correct_same": True, "correct_sham": True,
            "changed_diff": True, "changed_same": False, "changed_sham": False,
            "follow_diff": follow, "preserve_same": True,
        }

    # each violates; the two rates are both 0.5 and therefore equal
    records = [rec(0, True, False), rec(1, False, True)]
    out = mod.analyse(records, samples=20)
    assert out["all"]["follow_diff"]["rate"] == pytest.approx(
        out["all"]["accuracy_diff"]
    ), "the aggregate rates coincide, which is exactly the trap"
    assert out["metric_identity"]["holds"] is False
    assert out["metric_identity"]["n_follow_violations"] == 2


def test_preserve_and_follow_flags_are_independent(mod):
    """A preserve-only violation must not also condemn the follow identity."""

    records = [{
        "view_id": "v0", "source_pair_id": "p0", "surface_order": 0,
        "class": "A", "donor_class": "B",
        "choice_original": "A", "choice_diff": "B", "choice_same": "A",
        "correct_original": True, "correct_diff": True,
        "correct_same": False,          # preserve violates, follow does not
        "correct_sham": True,
        "changed_diff": True, "changed_same": False, "changed_sham": False,
        "follow_diff": True, "preserve_same": True,
    }]
    ident = mod.analyse(records, samples=10)["metric_identity"]
    assert ident["follow_diff_equals_accuracy_diff"] is True
    assert ident["preserve_same_equals_accuracy_same"] is False
    assert ident["holds"] is False


def test_independence_diagnostic_detects_correlation(mod):
    """Perfectly correlated readings must be reported as not independent.

    The fixture is half "reliably right" and half "reliably wrong". Each reading
    is then 50% accurate, so independence predicts agreement 0.5 while the
    observed agreement is 1.0 -- a clear violation. Note this is a *class-space*
    statistic: a model that is always correct is perfectly predicted by
    independence, which is the correct answer rather than a failure to detect.
    """

    records = []
    for i in range(30):
        good = i < 15
        records.append({
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": 0,
            "class": "A", "donor_class": "B",
            "choice_original": "A" if good else "B",
            "choice_diff": "B" if good else "A",
            "choice_same": "A" if good else "B",
            "correct_original": good, "correct_diff": good,
            "correct_same": good, "correct_sham": good,
            "changed_diff": good, "changed_same": False, "changed_sham": False,
            "follow_diff": good, "preserve_same": good,
        })
    diag = mod.analyse(records, samples=20)["independence_diagnostic"]
    assert diag["n"] == 30
    assert diag["space"] == "class_correctness"
    assert diag["observed_agreement"] == pytest.approx(1.0)
    assert diag["independence_predicted_agreement"] == pytest.approx(0.5)
    assert diag["independent"] is False


def test_independence_diagnostic_uses_class_space_not_letter_space(mod):
    """The verdict must come from class correctness, not pooled letter agreement.

    The fixture is chosen so the two spaces *disagree*, which is what makes this a
    real guard. Every view is correct, and both orders are present:

      * class space: both readings are always correct, so observed agreement 1.0
        and predicted 1.0 -> gap 0 -> independent (correctly).
      * letter space: order 0 answers A and order 1 answers B, so the pooled
        marginals are both 0.5 and the prediction collapses to 0.5, while the
        observed agreement stays 1.0 -> gap +0.5, which would read as a large
        violation.

    An implementation that used letter agreement would therefore report
    ``independent: False`` with a +0.5 gap here. Asserting the class-space values
    fails if the diagnostic is ever switched back to letter space.
    """

    records = []
    for i in range(40):
        order = i % 2
        correct_letter = "A" if order == 0 else "B"
        records.append({
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": order,
            "class": "A", "donor_class": "B",
            "choice_original": correct_letter,
            "choice_diff": "B" if order == 0 else "A",
            "choice_same": correct_letter,
            "correct_original": True, "correct_diff": True,
            "correct_same": True, "correct_sham": True,
            "changed_diff": True, "changed_same": False, "changed_sham": False,
            "follow_diff": True, "preserve_same": True,
        })
    diag = mod.analyse(records, samples=20)["independence_diagnostic"]

    # class space is primary and correct: no violation at all
    assert diag["space"] == "class_correctness"
    assert diag["observed_agreement"] == pytest.approx(1.0)
    assert diag["independence_predicted_agreement"] == pytest.approx(1.0)
    assert diag["gap"] == pytest.approx(0.0)
    assert diag["independent"] is True

    # letter space is retained, and would have cried violation on the same data
    assert diag["letter_space_agreement"] == pytest.approx(1.0)
    assert diag["letter_space_predicted_agreement"] == pytest.approx(0.5)
    letter_gap = diag["letter_space_agreement"] - diag["letter_space_predicted_agreement"]
    assert letter_gap == pytest.approx(0.5)
    assert letter_gap != pytest.approx(diag["gap"]), (
        "the two spaces must disagree on this fixture, or the test guards nothing"
    )


def test_independence_diagnostic_class_gap_is_exact(mod):
    """Pin the class-space gap to an independently computed value."""

    # 60 views, 30 pairs. Each pair: order 0 always correct; order 1 correct for
    # the first 15 pairs, wrong for the last 15. So a_orig = a_same = 0.75.
    records = []
    for pair in range(30):
        for order in (0, 1):
            good = order == 0 or pair < 15
            records.append({
                "view_id": f"v{pair}__o{order}", "source_pair_id": f"p{pair}",
                "surface_order": order, "class": "A", "donor_class": "B",
                "choice_original": "A" if good else "B",
                "choice_diff": "B" if good else "A",
                "choice_same": "A" if good else "B",
                "correct_original": good, "correct_diff": good,
                "correct_same": good, "correct_sham": good,
                "changed_diff": good, "changed_same": False, "changed_sham": False,
                "follow_diff": good, "preserve_same": good,
            })

    diag = mod.analyse(records, samples=20)["independence_diagnostic"]
    # both-correct pairs: 15 (order1 correct) ; both-wrong: 15 (order1 wrong)
    # plus order0 is always correct, so "both correct" means order1 also correct
    both_correct = 15 * 2
    both_wrong = 15 * 2
    n = 60
    assert diag["observed_agreement"] == pytest.approx((both_correct + both_wrong) / n)
    assert diag["independence_predicted_agreement"] == pytest.approx(0.75 * 0.75 + 0.25 * 0.25)
    assert diag["gap"] == pytest.approx(
        diag["observed_agreement"] - diag["independence_predicted_agreement"]
    )


def test_null_reported_under_both_parameterisations(mod):
    records = []
    for i in range(20):
        records.append({
            "view_id": f"v{i}", "source_pair_id": f"p{i}", "surface_order": i % 2,
            "class": "A", "donor_class": "B",
            "choice_original": "A", "choice_diff": "B", "choice_same": "A",
            "correct_original": True, "correct_diff": True,
            "correct_same": True, "correct_sham": True,
            "changed_diff": True, "changed_same": False, "changed_sham": False,
            "follow_diff": True, "preserve_same": True,
        })
    out = mod.analyse(records, samples=20)
    null = out["null_check"]
    assert "single_a" in null and "per_condition" in null
    assert "fraction_of_observed_explained" in null["single_a"]
    assert "fraction_of_observed_explained" in null["per_condition"]


def test_order_assignment_is_deterministic_and_balanced(mod):
    """The balanced single-order choice must be stable AND actually balanced.

    Hashing each pair independently and taking the parity gives 274/326 on the
    600 confirm pairs, which would leave the order effect in the "balanced"
    aggregate. The assignment sorts by hash and alternates instead.
    """

    pairs = [f"confirm-{i:05d}" for i in range(600)]
    a = mod.order_assignment(pairs)
    b = mod.order_assignment(pairs)
    assert a == b, "assignment must be deterministic across runs"
    counts = [sum(1 for v in a.values() if v == o) for o in (0, 1)]
    assert counts == [300, 300], f"expected an exactly balanced split, got {counts}"
    # an odd number of pairs can only be off by one
    odd = mod.order_assignment(pairs + ["extra"])
    odd_counts = [sum(1 for v in odd.values() if v == o) for o in (0, 1)]
    assert abs(odd_counts[0] - odd_counts[1]) == 1


def test_clustered_ci_resamples_pairs_not_views(mod):
    """Two orders of one pair must not count as two independent observations."""

    records = []
    for i in range(10):
        for order in (0, 1):
            records.append({
                "view_id": f"v{i}__o{order}", "source_pair_id": f"p{i}",
                "surface_order": order, "class": "A", "donor_class": "B",
                "choice_original": "A", "choice_diff": "B", "choice_same": "A",
                "correct_original": True, "correct_diff": True,
                "correct_same": True, "correct_sham": True,
                "changed_diff": True, "changed_same": False, "changed_sham": False,
                "follow_diff": True, "preserve_same": True,
            })
    lo, hi = mod.clustered_ci(records, "follow_diff", samples=100)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_pair_mean_removes_the_order_effect(mod):
    """The pair mean must average the two orders, not pick one.

    Construct pairs whose two orders disagree completely: order 0 is always
    correct, order 1 always wrong. A single-order estimator returns either 1.0 or
    0.0 depending on which order it happened to sample; the pair mean returns
    0.5 regardless.
    """

    records = []
    for i in range(20):
        for order in (0, 1):
            good = order == 0
            records.append({
                "view_id": f"v{i}__o{order}", "source_pair_id": f"p{i}",
                "surface_order": order, "class": "A", "donor_class": "B",
                "choice_original": "A" if good else "B",
                "choice_diff": "B" if good else "A",
                "choice_same": "A" if good else "B",
                "correct_original": good, "correct_diff": good,
                "correct_same": good, "correct_sham": good,
                "changed_diff": good, "changed_same": False, "changed_sham": False,
                "follow_diff": good, "preserve_same": good,
            })

    pm = mod.pair_mean_analysis(records)
    assert pm["n_source_pairs"] == 20
    assert pm["all"]["follow_diff"] == pytest.approx(0.5)
    assert pm["all"]["preserve_same"] == pytest.approx(0.5)

    # a single-order estimator lands on an extreme instead
    aggs = mod.aggregations(records)
    assert aggs["order0"] and aggs["order1"]
    assert mod.rate(aggs["order0"], "follow_diff") == pytest.approx(1.0)
    assert mod.rate(aggs["order1"], "follow_diff") == pytest.approx(0.0)


def test_pair_mean_subsets_drop_pairs_without_qualifying_views(mod):
    """A pair with no qualifying view must leave that subset, not count as zero."""

    records = []
    for i in range(4):
        for order in (0, 1):
            good = i < 2
            records.append({
                "view_id": f"v{i}__o{order}", "source_pair_id": f"p{i}",
                "surface_order": order, "class": "A", "donor_class": "B",
                "choice_original": "A" if good else "B",
                "choice_diff": "B", "choice_same": "A",
                "correct_original": good, "correct_diff": True,
                "correct_same": good, "correct_sham": True,
                "changed_diff": True, "changed_same": False, "changed_sham": False,
                "follow_diff": True, "preserve_same": True,
            })
    pm = mod.pair_mean_analysis(records)
    assert pm["all"]["n_source_pairs"] == 4
    # only the 2 "good" pairs have any view with original+same correct
    assert pm["E2_orig_and_same"]["n_source_pairs"] == 2


def test_committed_artifact_schema_is_stable(mod, tmp_path):
    """The docs tell collaborators which JSON paths to read; guard them.

    The reproduce recipe in `README.md` and the analysis scripts name these
    paths, and a schema change silently breaks that
    instruction. It has happened twice: `per_class_E2` moved under
    `by_aggregation.pooled_both_orders`, and the model key drifted from
    `r2_qwen3b` to `r2v2_qwen3b` because the generator keyed on the input
    filename stem. Both are now guarded -- the key by `--keys`, this by the
    assertions below.
    """

    import json
    from pathlib import Path

    artifact = (
        Path(__file__).resolve().parents[1]
        / "outputs" / "p0_fun_audio_chat" / "counterfactual_follow_3models.json"
    )
    if not artifact.exists():
        pytest.skip("committed artifact not present")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert "bootstrap" in payload, "seed/samples provenance must be recorded"
    assert payload["bootstrap"]["unit"].startswith("source pair")

    model = payload["models"]["r2_qwen3b"]
    assert set(model["by_aggregation"]) == {
        "pooled_both_orders", "order0", "order1", "single_order_balanced"
    }
    assert "pair_mean" in model

    pooled = model["by_aggregation"]["pooled_both_orders"]
    for key in ("all", "E1_orig_correct", "E2_orig_and_same", "E3_all_three"):
        assert key in pooled
        assert "follow_diff" in pooled[key] and "ci95" in pooled[key]["follow_diff"]
    for key in ("single_a", "per_condition", "observed_selectivity"):
        assert key in pooled["null_check"]
    assert pooled["independence_diagnostic"]["space"] == "class_correctness"
    assert "per_class_E2" in pooled
    assert pooled["metric_identity"]["holds"] is True


def test_empty_subset_serialises_to_null_not_nan(mod):
    """Non-finite rates must not reach the JSON as literal NaN.

    `json.dumps` emits the bare token `NaN` for a float nan, which Python accepts
    but strict parsers (`JSON.parse`, `jq`) reject. An empty E3 subset produces
    exactly that, so the writer runs the report through `_json_safe` and passes
    `allow_nan=False`. Without both, an artifact becomes unparseable downstream.
    """

    import json

    # one view, whose sham answer is wrong -> E3 is empty
    records = [{
        "view_id": "v0", "source_pair_id": "p0", "surface_order": 0,
        "class": "A", "donor_class": "B",
        "choice_original": "A", "choice_diff": "B", "choice_same": "A",
        "choice_sham": "A",
        "correct_original": True, "correct_diff": True,
        "correct_same": True, "correct_sham": False,
        "changed_diff": True, "changed_same": False, "changed_sham": False,
        "follow_diff": True, "preserve_same": True,
    }]
    out = mod.analyse(records, samples=10)
    assert out["E3_all_three"]["n"] == 0
    assert out["E3_all_three"]["follow_diff"]["rate"] != out["E3_all_three"]["follow_diff"]["rate"], \
        "the fixture must produce a nan rate for this test to mean anything"

    raw = json.dumps(out, default=float)
    assert "NaN" in raw, "sanity: the unguarded dump really does emit NaN"

    safe = mod._json_safe(out)
    serialised = json.dumps(safe, default=float, allow_nan=False)
    assert "NaN" not in serialised
    assert "Infinity" not in serialised
    # and it survives a strict parse
    reloaded = json.loads(serialised)
    assert reloaded["E3_all_three"]["follow_diff"]["rate"] is None


def test_json_safe_recurses_into_nested_structures(mod):
    """The sanitiser must reach rates nested inside lists and dicts."""

    import json

    payload = {
        "a": float("nan"),
        "b": [float("inf"), float("-inf"), 1.0],
        "c": {"d": {"e": float("nan")}},
        "f": "text",
        "g": 3,
    }
    safe = mod._json_safe(payload)
    assert safe["a"] is None
    assert safe["b"] == [None, None, 1.0]
    assert safe["c"]["d"]["e"] is None
    assert safe["f"] == "text" and safe["g"] == 3
    json.dumps(safe, allow_nan=False)  # must not raise


def test_cli_output_is_strict_json_end_to_end(tmp_path):
    """Run the real CLI on a manifest whose E3 subset is empty.

    This is the test that actually guards the writer: the unit tests above call
    `_json_safe` and `analyse` directly, so they would still pass if `main()`
    stopped applying the sanitiser. This drives the script as a subprocess and
    strict-parses whatever it wrote.
    """

    import json
    import subprocess
    import sys
    from pathlib import Path

    rows = []
    for pair in range(3):
        for order in (0, 1):
            pid = f"x{pair}__o{order}"
            # The real manifest counterbalances the surface order: order 0 names
            # the queried class "alpha" at A, order 1 names it at B. Mirror that
            # or the option letters and the class labels disagree.
            if order == 0:
                events = {"A": "alpha", "B": "beta"}
                query_letter, donor_letter = "A", "B"
            else:
                events = {"A": "beta", "B": "alpha"}
                query_letter, donor_letter = "B", "A"
            for cond in ("original", "target_replaced_diff",
                         "target_replaced_same", "sham"):
                # The diff row's ground truth is the donor class, so "correct"
                # and "followed the edit" coincide by construction; keep the
                # fixture consistent with that or the identity guard fires.
                if cond in ("target_replaced_diff", "sham"):
                    letter = donor_letter
                else:
                    letter = query_letter
                rows.append({
                    "pair_id": pid, "source_pair_id": f"x{pair}",
                    "surface_order": order, "condition": cond,
                    "parsed_choice": letter, "split": "confirm",
                    "correct": cond != "sham",
                    "option_events": events,
                    "query_class": "alpha", "absent_class": "beta",
                })
    manifest = tmp_path / "preds.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "follow.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "synth_counterfactual_follow.py"

    proc = subprocess.run(
        [sys.executable, str(script), str(manifest),
         "--split", "confirm", "--output", str(out), "--samples", "50"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text, "the CLI wrote a literal NaN token"
    assert "Infinity" not in text
    # strict parse: this is what a downstream consumer does
    payload = json.loads(text)
    assert payload["models"]["preds"]["by_aggregation"]["pooled_both_orders"][
        "E3_all_three"
    ]["n"] == 0
    assert "bootstrap" in payload
