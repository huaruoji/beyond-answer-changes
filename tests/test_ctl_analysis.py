"""Test the CTL analysis against synthetic predictions with known properties.

The pilot's numbers come out of `scripts/ctl_analyse.py`. A bug there would not crash
anything — it would produce plausible MIG values that mean nothing — so the analysis is
validated the same way the instrument is: by feeding it behaviour whose answer is known
in advance and checking it is recovered.

Four behaviours are synthesised from the real manifest:

=====================  ==========================================================
``perfect``            follows the evidence; MIG must be 1.0 and sham 0
``always_a``           ignores the audio; MIG must be 0.0 — the statistic is
                       computable without competence but is not evidence of it
``audibility_driven``  follows the evidence only when audibility is high; the
                       equal-energy contrast must recover ≈ +0.5
``noisy``              answers at chance; MIG must be near 0. Measured with a 20% flip
                       rate the MIG was +0.43, because two independent 20% flips give the
                       two conditions different change rates — variance, not bias, and a
                       useful reminder that a single pilot MIG needs its interval
=====================  ==========================================================
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from scripts.ctl_analyse import (
    analyse,
    bootstrap_ci,
    collect,
    equal_energy_gap,
    item_stat,
    usable_items,
)

MANIFEST = Path("outputs/ctl/sample30/manifest.jsonl")

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="built CTL sample not present in this checkout"
)


def _synthesise(mode: str) -> dict:
    rows = [json.loads(ln) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(7)
    choices: dict[tuple[str, str, str], str | None] = {}
    for row in rows:
        truth = row["ground_truth"]
        if mode == "perfect":
            choice = truth
        elif mode == "always_a":
            choice = "A"
        elif mode == "audibility_driven":
            choice = truth if float(row["audibility_db"]) > -5.0 else "A"
        elif mode == "noisy":
            # chance-level answering: the audio is ignored entirely
            choice = rng.choice(["A", "B"])
        else:
            raise ValueError(mode)
        choices[(row["pair_id"], row["condition"], "")] = choice
    return collect(MANIFEST, choices)


def _cells() -> list[str]:
    rows = [json.loads(ln) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted({f"{r['calibration']}__{r['masker']}" for r in rows})


def test_evidence_follower_scores_a_perfect_mig() -> None:
    result = analyse(_synthesise("perfect"), _cells())
    for block in result["cells"].values():
        assert block["MIG"] == pytest.approx(1.0, abs=1e-9)
        assert block["P_change_sham"] == pytest.approx(0.0, abs=1e-9)
        assert block["P_change_diff"] == pytest.approx(1.0, abs=1e-9)
        assert block["P_change_same"] == pytest.approx(0.0, abs=1e-9)


def test_audio_ignoring_model_scores_zero_but_is_not_mistaken_for_evidence() -> None:
    """MIG = 0 is the same number for "ignores audio" and "cannot do the task".

    This is why the plan forbids describing the statistic as independent of
    competence: computable without it, yes; informative about it, no.
    """

    result = analyse(_synthesise("always_a"), _cells())
    for block in result["cells"].values():
        assert block["MIG"] == pytest.approx(0.0, abs=1e-9)
        assert block["P_change_diff"] == pytest.approx(0.0, abs=1e-9)
        assert block["P_change_same"] == pytest.approx(0.0, abs=1e-9)


def test_equal_energy_contrast_recovers_a_known_audibility_effect() -> None:
    """The bonus criterion must show up when the behaviour is built to have it."""

    result = analyse(_synthesise("audibility_driven"), _cells())
    contrast = result["equal_energy_mig_contrast"]
    assert contrast["mig_low_audibility_masker"] > contrast["mig_high_audibility_masker"]
    assert contrast["difference"] == pytest.approx(0.5, abs=1e-9)


def test_chance_level_answering_gives_a_near_zero_mig() -> None:
    result = analyse(_synthesise("noisy"), _cells())
    migs = [block["MIG"] for block in result["cells"].values()]
    assert max(abs(m) for m in migs) < 0.35, migs
    # a chance-level model also moves under the no-op edit, which is what the sham
    # floor is for: it separates "the pipeline perturbs audio" from "the model is noisy"
    assert min(block["P_change_sham"] for block in result["cells"].values()) > 0.1


def test_bootstrap_ci_is_degenerate_for_a_constant_sample() -> None:
    mean, low, high = bootstrap_ci([1.0] * 23)
    assert (mean, low, high) == (1.0, 1.0, 1.0)
    mean, low, high = bootstrap_ci([])
    assert mean != mean  # NaN, not a crash


def test_item_stat_flags_unanswered_items() -> None:
    assert item_stat({"original": "A", "diff": "B", "same": "A", "sham": "A"})["m"] == 1.0
    assert item_stat({"original": "A", "diff": "A", "same": "A", "sham": "A"})["m"] == 0.0
    assert item_stat({}) is None
    partial = item_stat({"original": "A", "diff": None, "same": "A", "sham": "A"})
    assert partial["answered"] == 0.0


def test_subset_selection_tracks_the_measured_gaps() -> None:
    """The subset must be exactly the items whose manipulation is large enough.

    Measured on the 60-item sample only 58% of items clear 10 dB, so the headline
    number is reported both ways; if the filter silently selected everything (or
    nothing) the two reports would look identical and nobody would notice.
    """

    collected = _synthesise("perfect")
    meta = collected["meta"]
    gaps = {k: equal_energy_gap(v) for k, v in meta.items()}
    assert all(g is not None for g in gaps.values())

    for threshold in (5.0, 10.0, 20.0):
        picked = usable_items(collected, threshold)
        assert picked == {k for k, g in gaps.items() if g >= threshold}
        assert picked, f"no items cleared {threshold} dB"

    # a threshold above every gap must select nothing rather than silently falling back
    assert usable_items(collected, max(gaps.values()) + 1.0) == set()


def test_restricted_analysis_uses_only_the_subset() -> None:
    collected = _synthesise("audibility_driven")
    cells = _cells()
    picked = sorted(usable_items(collected, 10.0))[:10]
    restricted = analyse(collected, cells, restrict_to=set(picked))
    assert restricted["n_items"] == len(picked)
    for block in restricted["cells"].values():
        assert block["n_items"] == len(picked)


def test_restricted_analysis_still_recovers_the_contrast() -> None:
    """Filtering to high-gap items must not destroy the known effect."""

    collected = _synthesise("audibility_driven")
    cells = _cells()
    subset = analyse(collected, cells, restrict_to=usable_items(collected, 10.0))
    contrast = subset["equal_energy_mig_contrast"]
    assert contrast["difference"] > 0.2, contrast


def test_verdict_rule_separates_the_three_branches() -> None:
    """The pilot is only worth GPU time if its decision rule can tell outcomes apart.

    Gates alone establish that the manipulation is real; the equal-energy contrast
    establishes whether the model responds to it. Those are different claims, and the
    three regimes below must land in three different branches.
    """

    from scripts.ctl_verdict_check import synthesise, verdict

    rows = [json.loads(ln) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cells = _cells()
    readings = {}
    for regime in ("evidence_user", "audio_ignoring", "audibility_gated"):
        collected = synthesise(rows, regime)
        result = analyse(collected, cells)
        subset = analyse(collected, cells, restrict_to=usable_items(collected, 10.0))
        result["subset_analysis"] = {
            "n_items": subset["n_items"],
            "cells": subset["cells"],
            "equal_energy_mig_contrast": subset.get("equal_energy_mig_contrast"),
        }
        readings[regime] = verdict(result, 10.0)["reading"]

    assert len(set(readings.values())) == 3, readings
    assert "bonus" in readings["audibility_gated"]
    assert "does not respond" in readings["audio_ignoring"]
    assert "without bonus" in readings["evidence_user"]
