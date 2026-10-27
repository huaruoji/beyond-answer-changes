"""Tests for the synthetic attribution estimator's decision logic.

The estimator decides whether a run is interpretable at all, so its gates are
worth pinning. Each test builds a change pattern with a known rate and checks
that the report recovers it and reaches the intended verdict.
"""

from __future__ import annotations

import pytest

from scripts.synth_attribution import (
    NATURAL_CONDITIONS,
    SYNTHETIC_CONDITIONS,
    _aggregate,
    _contrasts,
    _gates,
    _verdict,
)

DIFF = "target_replaced_diff"
SAME = "target_replaced_same"
CONTROLS = [
    "nuisance_replaced",
    "nuisance_removed",
    "background_removed",
    "sham",
    "silence",
]


def _full(views):
    """Aggregate plus contrasts, the way main() reports them."""

    metrics = _aggregate(views)
    return metrics | _contrasts(metrics, DIFF, SAME, CONTROLS)


def _views(
    *,
    n: int = 200,
    diff: float = 0.0,
    same: float = 0.0,
    nuis: float = 0.0,
    background: float = 0.0,
    sham: float = 0.0,
    silence: float = 0.0,
    accuracy: float = 0.9,
) -> tuple[list[dict[str, bool]], dict[str, float]]:
    """Synthesise views with exact change rates, plus the accuracy to report."""

    rates = {
        "target_replaced_diff": diff,
        "target_replaced_same": same,
        "nuisance_removed": nuis,
        "background_removed": background,
        "sham": sham,
        "silence": silence,
    }
    views: list[dict[str, bool]] = []
    for index in range(n):
        # Deterministic, exactly `rate` of the views flagged as changed.
        view = {
            condition: (index % 100) < round(rate * 100)
            for condition, rate in rates.items()
        }
        views.append(view)
    return views, {"original": accuracy}


def _report(views, accuracy, **overrides):
    metrics = _full(views)
    accuracy = dict(accuracy)
    accuracy.update(overrides.pop("accuracy", {}))
    report = {
        "metrics": metrics,
        "accuracy_by_condition": accuracy,
        "bootstrap_95": overrides.pop("bootstrap_95", {}),
    }
    report["gates"] = _gates(accuracy, metrics)
    report["verdict"] = _verdict(report)
    return report


def test_aggregate_recovers_known_rates() -> None:
    views, _ = _views(diff=0.80, same=0.10, nuis=0.05, background=0.04, sham=0.02)
    metrics = _full(views)
    assert metrics["A_diff"] == pytest.approx(0.80, abs=0.02)
    assert metrics["A_same"] == pytest.approx(0.10, abs=0.02)
    assert metrics["A_nuisance_removed"] == pytest.approx(0.05, abs=0.02)
    assert metrics["A_background_removed"] == pytest.approx(0.04, abs=0.02)
    assert metrics["A_sham"] == pytest.approx(0.02, abs=0.02)
    assert metrics["selectivity"] == pytest.approx(0.70, abs=0.03)
    assert metrics["specificity"] == pytest.approx(0.08, abs=0.03)
    assert metrics["perturbation"] == pytest.approx(0.05, abs=0.02)


def test_zero_rates_give_zero_selectivity() -> None:
    views, _ = _views()
    metrics = _full(views)
    assert metrics["selectivity"] == 0.0
    assert metrics["specificity"] == 0.0


def test_sham_failure_outranks_everything() -> None:
    """A pipeline that perturbs audio invalidates a strong-looking result."""

    views, accuracy = _views(diff=0.80, same=0.10, sham=0.12)
    report = _report(views, accuracy)
    assert report["metrics"]["selectivity"] > 0.5
    assert report["gates"]["G_sham"]["passed"] is False
    assert report["verdict"].startswith("INSTRUMENT INVALID")


def test_competence_gate_blocks_reading_selectivity() -> None:
    views, accuracy = _views(diff=0.20, same=0.05, silence=0.05, accuracy=0.55)
    report = _report(views, accuracy)
    assert report["gates"]["G_competence"]["passed"] is False
    assert report["verdict"].startswith("NO CLAIM")


def test_low_dependency_alone_blocks_the_run() -> None:
    """High accuracy with no audio dependence is still not interpretable."""

    views, accuracy = _views(diff=0.80, same=0.10, silence=0.02, accuracy=0.92)
    report = _report(views, accuracy)
    assert report["gates"]["G_competence"]["passed"] is False
    assert report["verdict"].startswith("NO CLAIM")


def test_positive_selectivity_requires_an_interval_above_zero() -> None:
    views, accuracy = _views(diff=0.80, same=0.10, silence=0.60)
    report = _report(
        views, accuracy, bootstrap_95={"selectivity": [0.55, 0.85]}
    )
    assert report["gates"] == {
        "G_competence": {**report["gates"]["G_competence"], "passed": True},
        "G_sham": {**report["gates"]["G_sham"], "passed": True},
    }
    assert report["verdict"].startswith("SELECTIVITY POSITIVE")


def test_interval_spanning_zero_is_reported_as_no_dissociation() -> None:
    views, accuracy = _views(diff=0.30, same=0.28, silence=0.60)
    report = _report(views, accuracy)
    assert report["gates"]["G_competence"]["passed"] is True
    assert report["gates"]["G_sham"]["passed"] is True
    assert report["verdict"].startswith("NO DISSOCIATION")


def test_negative_interval_is_reported_as_negative() -> None:
    views, accuracy = _views(diff=0.10, same=0.40, silence=0.60)
    report = _report(
        views, accuracy, bootstrap_95={"selectivity": [-0.40, -0.20]}
    )
    assert report["verdict"].startswith("SELECTIVITY NEGATIVE")


def test_all_seven_conditions_are_tracked() -> None:
    """``original`` is the reference, so six conditions produce change rates."""

    views, _ = _views(diff=0.5, same=0.5, nuis=0.5, background=0.5, sham=0.5, silence=0.5)
    assert set(views[0]) == set(SYNTHETIC_CONDITIONS) - {"original"}


def test_natural_route_reports_no_selectivity_rather_than_a_substitute() -> None:
    """Without a same-span control the headline is undefined, not approximated.

    The natural route writes the same donor into a different span. That is a
    legitimate control but it is not the matched same-span contrast, so the
    report must say so instead of quietly substituting one number for another.
    """

    views = [
        {
            "target_replaced_diff": (i % 100) < 70,
            "nuisance_replaced": (i % 100) < 12,
            "nuisance_removed": (i % 100) < 8,
            "background_removed": (i % 100) < 3,
            "sham": (i % 100) < 2,
            "silence": (i % 100) < 60,
        }
        for i in range(200)
    ]
    metrics = _aggregate(views) | _contrasts(
        _aggregate(views), "target_replaced_diff", None, CONTROLS
    )
    assert metrics["selectivity"] is None
    assert metrics["A_same"] is None
    assert metrics["A_diff"] == pytest.approx(0.70, abs=0.02)
    assert metrics["controls"]["nuisance_replaced"] == pytest.approx(0.12, abs=0.03)
    report = {
        "metrics": metrics,
        "accuracy_by_condition": {"original": 0.85},
        "bootstrap_95": {},
    }
    report["gates"] = _gates(report["accuracy_by_condition"], metrics)
    report["verdict"] = _verdict(report)
    assert report["verdict"].startswith("NO MATCHED SAME-SPAN CONTROL")
    assert report["gates"]["G_sham"]["passed"] is True
