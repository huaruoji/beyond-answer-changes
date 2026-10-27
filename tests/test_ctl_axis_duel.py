"""Tests for the axis duel: energy-calibrated vs audibility-calibrated difficulty.

The properties locked here are the ones whose absence has already cost this branch real
time, plus the ones the duel's whole design rests on:

* **both axes cover exactly the same item set** — the head-to-head is only a head-to-head
  if the two axes run over the same events, backgrounds and donors;
* **no two conditions of a cell are byte-identical** — the void-build bug that made every
  downstream number meaningless while the manifest gave the arms opposite ground truths;
* **nothing clips** — ``sample60_v2`` shipped 154/240 edited arms at full scale because
  the peak was checked before the splice;
* **the recorded calibration is the realised one** — an axis whose manifest number is not
  the number on disk is not calibrated;
* **the axes still dissociate** — a build where the two axes collapse onto one line cannot
  answer the question it exists for, so the separation is asserted, not merely reported.

Two layers: a self-contained mini-build (2 items x 2 levels x 2 axes) that exercises the
builder's real code paths in any checkout that has the corpus, and a check of the full
committed artifact under ``outputs/ctl/axis_duel`` when it is present.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import ctl_axis_duel_build as duel
from src.audibility import audibility_margin_db
from src.ctl import LevelTarget, build_item, render_item, write_manifest
from src.maskers import (
    MaskerSpec,
    normalise_event,
    shaped_noise,
    solve_masker_attenuation_db,
    solve_masker_for_energy_snr_db,
)
from src.synth import read_audio_mono

SR = 16000
FOREGROUND = Path("data/processed/synth/foreground_all")
BACKGROUND = Path("data/processed/synth/background")
VOCAB = Path("data/processed/synth/vocab_confirm.txt")
ARTIFACT = Path("outputs/ctl/axis_duel")

pytestmark = pytest.mark.skipif(
    not FOREGROUND.exists() or not BACKGROUND.exists(),
    reason="CTL corpus not present in this checkout",
)


def _corpus() -> tuple[list[str], list[Path]]:
    labels = [ln.strip() for ln in VOCAB.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(
        p for d in sorted(BACKGROUND.iterdir()) if d.is_dir() for p in d.glob("*.wav")
    )
    return labels, backgrounds


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _band_share(wave: np.ndarray, lo: float, hi: float) -> float:
    power = np.abs(np.fft.rfft(np.asarray(wave, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(wave.size, d=1.0 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    return float(power[sel].sum() / power.sum())


# --- masker conventions -------------------------------------------------------


def test_threshold_shaped_masker_is_loud_where_hearing_is_insensitive() -> None:
    """The literal reading, and the only one that separates the axes.

    PSD follows the threshold in quiet, so the masker spends its energy below 300 Hz —
    where a human listener cannot use it. Measured separation from white noise at matched
    energy is ~30 dB of audibility, which is what gives the duel any power at all. If a
    future edit "fixes" this shape into the equal-masking one, the two axes become the
    same knob (median 2 dB apart) and the experiment silently stops being able to answer
    its question.
    """

    shaped = shaped_noise(6.0, SR, MaskerSpec(kind="threshold_shaped", rms=0.05, seed=0))
    white = shaped_noise(6.0, SR, MaskerSpec(kind="white", rms=0.05, seed=0))
    assert _rms(shaped) == pytest.approx(0.05, rel=1e-6)
    assert _band_share(shaped, 0.0, 300.0) > 0.9, _band_share(shaped, 0.0, 300.0)
    assert _band_share(white, 0.0, 300.0) < 0.1
    assert _band_share(shaped, 1000.0, 5000.0) < _band_share(white, 1000.0, 5000.0)


def test_threshold_inverse_masker_is_loud_where_hearing_is_sensitive() -> None:
    """The audiological equal-masking variant, kept as the measured near-null control."""

    inverse = shaped_noise(6.0, SR, MaskerSpec(kind="threshold_inverse", rms=0.05, seed=0))
    white = shaped_noise(6.0, SR, MaskerSpec(kind="white", rms=0.05, seed=0))
    assert _rms(inverse) == pytest.approx(0.05, rel=1e-6)
    assert _band_share(inverse, 1000.0, 5000.0) > _band_share(white, 1000.0, 5000.0)
    assert _band_share(inverse, 0.0, 300.0) < 0.05


def test_energy_solver_hits_its_realised_target() -> None:
    event = normalise_event(shaped_noise(1.5, SR, MaskerSpec(kind="white", rms=0.05, seed=3)), 0.15)
    template = shaped_noise(6.0, SR, MaskerSpec(kind="white", rms=1.0, seed=4))
    sliced = template[: event.size]
    background = (shaped_noise(6.0, SR, MaskerSpec(kind="pink", rms=0.002, seed=5))[: event.size]).astype(np.float32)
    for target in (5.0, -5.0, -20.0):
        _, level = solve_masker_for_energy_snr_db(
            event, sliced, SR, reference_rms=1.0, target_snr_db=target, background=background
        )
        interferer = background + sliced * level
        achieved = 20 * np.log10(_rms(event) / _rms(interferer))
        assert achieved == pytest.approx(target, abs=0.1), (target, achieved)


def test_audibility_solver_hits_its_realised_target_with_background() -> None:
    """The solve must land on background+masker, not on the masker alone."""

    event = normalise_event(shaped_noise(1.5, SR, MaskerSpec(kind="white", rms=0.05, seed=6)), 0.15)
    template = shaped_noise(6.0, SR, MaskerSpec(kind="threshold_shaped", rms=1.0, seed=7))
    sliced = template[: event.size]
    background = (shaped_noise(6.0, SR, MaskerSpec(kind="pink", rms=0.01, seed=8))[: event.size]).astype(np.float32)
    baseline = audibility_margin_db(event, background, SR)
    for drop in (5.0, 15.0):
        _, level = solve_masker_attenuation_db(
            event,
            sliced,
            SR,
            reference_rms=1.0,
            target_margin_db=baseline - drop,
            background=background,
        )
        achieved = audibility_margin_db(event, (background + sliced * level).astype(np.float32), SR)
        assert achieved == pytest.approx(baseline - drop, abs=0.1), (drop, achieved)


def test_level_target_validation() -> None:
    with pytest.raises(ValueError):
        LevelTarget("x", "white", "loudness", 1.0)
    with pytest.raises(ValueError):
        LevelTarget("x", "white", "energy_snr_db", 1.0, relative_to_baseline=True)


# --- item-level construction --------------------------------------------------


@pytest.fixture(scope="module")
def duel_item():
    labels, backgrounds = _corpus()
    query, other = labels[0], labels[1]
    event_path = sorted((FOREGROUND / query).glob("*.wav"))[0]
    donor_path = sorted((FOREGROUND / other).glob("*.wav"))[0]
    targets = (
        LevelTarget("energy_L0", "white", "energy_snr_db", 0.0),
        LevelTarget("energy_L1", "white", "energy_snr_db", -12.0),
        LevelTarget("audibility_L0", "threshold_shaped", "audibility_margin_db", 15.0, True),
        LevelTarget("audibility_L1", "threshold_shaped", "audibility_margin_db", 30.0, True),
    )
    item = build_item(
        item_id="duel-test-00",
        query_class=query,
        donor_class=other,
        event_source=event_path,
        background_source=backgrounds[0],
        donor_source=donor_path,
        maskers=duel.MASKER_SPECS,
        seed=17,
        level_targets=targets,
    )
    donor, _ = read_audio_mono(donor_path, SR)
    return item, donor, targets


def test_every_level_gets_its_own_solved_cell(duel_item) -> None:
    item, _, _ = duel_item
    assert set(item.masker_rms) == {"energy_L0", "energy_L1", "audibility_L0", "audibility_L1"}
    for target in duel_item[2]:
        assert item.level_of(target.calibration, target.masker) > 0.0
    # the energy cells must land on their energy targets, the audibility cells on their
    # margins; each solved against background+masker, which is what the manifest records
    for target in duel_item[2]:
        if target.quantity == "energy_snr_db":
            assert item.energy_snr(target.calibration, target.masker) == pytest.approx(
                target.target, abs=0.25
            )
        else:
            assert item.audibility(target.calibration, target.masker) == pytest.approx(
                item.baseline_margin_db - target.target, abs=0.25
            )


def test_scaled_copy_preserves_the_axes_and_leaves_the_item_alone(duel_item) -> None:
    item, _, _ = duel_item
    calibration, masker = "energy_L0", "white"
    before_energy = item.energy_snr(calibration, masker)
    before_audibility = item.audibility(calibration, masker)
    before_event = item.event.copy()
    scaled = item.scaled_copy(0.25)
    assert np.array_equal(item.event, before_event), "the caller's item must not be mutated"
    assert scaled.event is not item.event
    assert _rms(scaled.event) == pytest.approx(0.25 * _rms(item.event), rel=1e-5)
    # energy SNR is exactly invariant under one gain on the whole mix
    assert scaled.energy_snr(calibration, masker) == pytest.approx(before_energy, abs=1e-3)
    # and the audibility margin is invariant while the masker, not the quiet threshold,
    # sets the threshold — which is the case for a masker loud enough to be calibrated
    assert abs(scaled.audibility(calibration, masker) - before_audibility) < 1.0


def test_render_item_records_the_calibration_of_the_audio_it_wrote(duel_item, tmp_path: Path) -> None:
    """The manifest row must describe the array after the shared headroom gain.

    The cell is deliberately a *weak* masker: with a masker loud enough to set the
    threshold the margin is scale-invariant, and the test would pass on a manifest built
    from the pre-gain arrays. Only when the threshold in quiet is what the event sits
    above does the gain move the recorded number, which is what makes this a test.
    """

    item, _, _ = duel_item
    calibration, masker = "energy_weak", "white"
    item.masker_rms[calibration] = {masker: 0.0005}
    item.event = (item.event * 8.0).astype(np.float32)  # force the mix over the ceiling
    unscaled_peak = float(np.abs(item.scene_for(calibration, masker)).max())
    out = render_item(item, tmp_path, calibration=calibration, masker=masker, conditions=("original",))
    gain = float(out["peak"]) / unscaled_peak  # type: ignore[arg-type]
    assert gain < 0.5, "the test needs a real headroom gain to be meaningful"
    scaled = item.scaled_copy(gain)
    row = out["rows"][0]  # type: ignore[index]
    assert abs(
        scaled.audibility(calibration, masker) - item.audibility(calibration, masker)
    ) > 1.0, "the gain must actually move the recorded quantity, else the test is vacuous"
    assert float(row["audibility_db"]) == pytest.approx(scaled.audibility(calibration, masker), abs=1e-4)
    assert float(row["energy_snr_db"]) == pytest.approx(scaled.energy_snr(calibration, masker), abs=1e-4)


def test_render_item_refuses_a_clipping_edit(duel_item, tmp_path: Path) -> None:
    """Post-splice guard: the pre-splice peak check cannot see an impulsive donor.

    ``sample60_v2`` shipped 154/240 edited arms at full scale for exactly this reason.
    """

    item, donor, _ = duel_item
    spike = np.zeros(donor.size, dtype=np.float32)
    spike[donor.size // 2] = 1.0  # RMS-matched to the span, so this becomes enormous
    with pytest.raises(AssertionError, match="peaks at"):
        render_item(
            item,
            tmp_path,
            calibration="energy_L0",
            masker="white",
            conditions=("original", "diff"),
            donor=spike,
            donor_span_samples=int(spike.size),
        )


# --- item selection and the donor ladder check --------------------------------


def _fake_admissible(classes: list[str], per_class: int) -> list[dict[str, object]]:
    return [
        {"item_id": f"{label}-{index:02d}", "query_class": label}
        for label in classes
        for index in range(per_class)
    ]


def test_select_admissible_covers_classes_round_robin() -> None:
    """A capped build must not be an alphabetical prefix of the corpus.

    The superseded build stopped screening once it had the requested items, so its 40 items
    came from the first 22 of 36 classes and 14 classes were absent. Selection is
    round-robin: every class contributes its first item before any class contributes a
    second, so a capped build still covers the vocabulary.
    """

    classes = [f"class-{i:02d}" for i in range(10)]
    pool = _fake_admissible(classes, 3)
    selected = duel.select_admissible(pool, 7)
    assert [entry["item_id"] for entry in selected] == [
        "class-00-00", "class-01-00", "class-02-00", "class-03-00", "class-04-00",
        "class-05-00", "class-06-00",
    ]
    assert len({entry["query_class"] for entry in selected}) == 7
    # and the alphabetical-prefix selection this replaced would have covered three classes
    prefix = pool[:7]
    assert len({entry["query_class"] for entry in prefix}) == 3
    # a pool smaller than the request is returned whole rather than truncated per class
    assert len(duel.select_admissible(pool, 999)) == len(pool)
    assert len(duel.select_admissible(pool, 30)) == len(pool)


def _ladder_rows(medians: dict[str, list[float]]) -> list[dict[str, object]]:
    rows = []
    for axis, values in medians.items():
        for level, value in enumerate(values):
            rows.append(
                {
                    "axis": axis,
                    "level": level,
                    "condition": "diff",
                    "span_audibility_db": value,
                }
            )
    return rows


def test_donor_ladder_check_rejects_a_flat_diff_arm() -> None:
    """The check must fail on the defect it exists for, not merely pass on the fix.

    Measured on the superseded build, the diff arm's own audibility moved by only about
    +-3 dB across the 30 dB ladder while the manifest's audibility_db described the event
    that had been overwritten. A check that passes that shape is decoration.
    """

    flat = duel.donor_ladder_check(
        _ladder_rows(
            {
                "energy": [1.5, -2.8, -4.5, -4.2, -4.3],
                "audibility": [11.8, 9.4, 7.1, 7.5, 7.2],
            }
        )
    )
    assert not flat["passed"], flat
    assert any("does not track the ladder" in failure for failure in flat["failures"])

    tracking = duel.donor_ladder_check(
        _ladder_rows(
            {
                "energy": [0.3, -7.2, -14.9, -21.8, -29.2],
                "audibility": [11.2, 6.2, 0.9, -5.0, -12.5],
            }
        )
    )
    assert tracking["passed"], tracking
    assert tracking["axes"]["energy"]["span_db"] == pytest.approx(29.5, abs=0.01)

    # a ladder that stops biting at the hard end is also caught
    stalled = duel.donor_ladder_check(
        _ladder_rows(
            {
                "energy": [0.3, -7.2, -14.9, -15.4, -14.8],
                "audibility": [11.2, 6.2, 0.9, -5.0, -12.5],
            }
        )
    )
    assert not stalled["passed"], stalled


# --- the mini build: the properties, end to end -------------------------------


def test_mini_duel_locks_the_build_properties(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build 2 items x 2 levels x 2 axes and assert the load-bearing properties."""

    # a two-level version of the real grid. The ladders have to be wide enough that the
    # donor's own audibility really moves: with 15/30 dB drops the first two items' donors
    # are already below the threshold in quiet at both levels and the ladder check — which
    # exists to catch exactly that flatness — fails on two items' worth of median noise.
    monkeypatch.setattr(
        duel,
        "AXES",
        (
            duel.AxisSpec("energy", "white", "energy_snr_db", (0.0, -25.0)),
            duel.AxisSpec(
                "audibility", "threshold_shaped", "audibility_margin_db", (5.0, 45.0), True
            ),
        ),
    )
    labels, backgrounds = _corpus()
    candidates = duel.candidate_items(labels, FOREGROUND, backgrounds, 20260918)[:16]
    admissible, skipped, stats = duel.assemble(candidates, 2, verbose=False)
    assert len(admissible) == 2, skipped[:3]
    # every candidate was screened, not just the prefix needed to reach the wanted count
    assert stats["candidates_screened"] == len(candidates) == 16
    assert stats["selected"] == 2 and stats["requested"] == 2
    assert stats["classes_covered"] == stats["classes_admissible"] == 2

    rows, entries, cells = duel.render_all(admissible, tmp_path)
    write_manifest(rows, tmp_path / "manifest.jsonl")
    audit = duel.run_audit(tmp_path, admissible, cells, rows, stats)
    assert audit["passed"], audit["failures"]

    assert len(rows) == 2 * 2 * 2 * 2
    assert len(entries) == 2 and len(cells) == 8
    assert audit["items_per_level"] == {
        "energy_L0": 2, "energy_L1": 2, "audibility_L0": 2, "audibility_L1": 2
    }

    # 1. both axes cover exactly the same items and the same levels
    per_axis: dict[str, set[str]] = defaultdict(set)
    levels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        per_axis[str(row["axis"])].add(str(row["item_id"]))
        levels[str(row["axis"])].add(int(row["level"]))
    assert per_axis["energy"] == per_axis["audibility"], per_axis
    assert len(per_axis["energy"]) == 2
    assert levels["energy"] == levels["audibility"] == {0, 1}

    # 2. no two conditions of the same cell are byte-identical
    per_cell: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in rows:
        key = f"{row['item_id']}__{row['calibration']}__{row['masker']}"
        per_cell[key][str(row["condition"])] = Path(str(row["audio"]))
    assert len(per_cell) == 8
    for key, paths in per_cell.items():
        assert set(paths) == {"original", "diff"}, key
        assert paths["original"].read_bytes() != paths["diff"].read_bytes(), key

    # 3. nothing clips
    for path in sorted((tmp_path / "audio").glob("*.wav")):
        wave, _ = sf.read(path, dtype="float32")
        assert float(np.abs(wave).max()) <= 1.0, path.name

    # 4. both calibrations, the baseline margin and the realised span level are per row
    for row in rows:
        for key in (
            "energy_snr_db",
            "audibility_db",
            "baseline_margin_db",
            "span_audibility_db",
            "span_energy_snr_db",
        ):
            assert np.isfinite(float(row[key])), (key, row["pair_id"])
        assert row["audibility_db_reference"] == "event"
        assert row["span_audibility_db_reference"] == "queried_span_cue"
        assert row["ground_truth"] == ("B" if row["condition"] == "diff" else "A")
        assert set(row["option_events"]) == {"A", "B"}

    # 5. the diff arm's own difficulty tracks the ladder (CTL-Plan §3.4i#3)
    ladder = duel.donor_ladder_check(rows)
    assert ladder["passed"], ladder
    assert audit["checks"]["donor_ladder_passed"] is True


# --- the committed artifact ---------------------------------------------------


@pytest.mark.skipif(
    not (ARTIFACT / "manifest.jsonl").exists(),
    reason="axis_duel build not present",
)
@pytest.mark.skipif(
    not (ARTIFACT / "audio").is_dir(),
    reason=(
        "axis_duel audio is not in the working tree: it is regenerable from "
        "scripts/ctl_axis_duel_build.py and was dropped to keep the checkout small. "
        "The manifest/audit JSON checks below still run when the JSON is present; this "
        "test additionally verifies audio bytes, so it needs the audio."
    ),
)
def test_committed_axis_duel_artifact() -> None:
    rows = [
        json.loads(line)
        for line in (ARTIFACT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    axis = json.loads((ARTIFACT / "axis.json").read_text(encoding="utf-8"))
    audit = json.loads((ARTIFACT / "build_audit.json").read_text(encoding="utf-8"))
    report = json.loads((ARTIFACT / "calibration_report.json").read_text(encoding="utf-8"))

    assert audit["passed"], audit["failures"]
    assert audit["checks"]["axis_item_sets_identical"] is True
    assert audit["checks"]["cells_with_byte_identical_conditions"] == 0
    assert audit["checks"]["files_clipping"] == 0
    assert audit["checks"]["worst_calibration_error_db"] <= audit["calibration_tolerance_db"]
    # the diff arm must carry the ladder, and every level must have the same n
    assert audit["checks"]["donor_ladder_passed"] is True, audit["checks"]["donor_ladder"]
    assert set(audit["items_per_level"].values()) == {100}, audit["items_per_level"]
    assert audit["screening"]["candidates_screened"] == audit["screening"]["candidates_total"]
    # every class that has an admissible candidate must be represented; a class with none
    # is a corpus property, a class dropped by the selection would be a coverage bias
    assert audit["screening"]["classes_covered"] == audit["screening"]["classes_admissible"]

    per_axis: dict[str, set[str]] = defaultdict(set)
    levels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        per_axis[str(row["axis"])].add(str(row["item_id"]))
        levels[str(row["axis"])].add(int(row["level"]))
    assert per_axis["energy"] == per_axis["audibility"]
    assert levels["energy"] == levels["audibility"] == {0, 1, 2, 3, 4}
    assert len(per_axis["energy"]) == len(axis["items"]) == 100

    # every row records the level of the cue that is actually in its audio, and on the
    # diff arm that is the replacement, not the event it replaced
    for row in rows:
        assert np.isfinite(float(row["span_audibility_db"])), row["pair_id"]
        assert np.isfinite(float(row["span_energy_snr_db"])), row["pair_id"]
        assert row["span_audibility_db_reference"] == "queried_span_cue"
        assert row["audibility_db_reference"] == "event"
    diff_rows = [row for row in rows if row["condition"] == "diff"]
    spread = max(float(r["span_audibility_db"]) for r in diff_rows) - min(
        float(r["span_audibility_db"]) for r in diff_rows
    )
    assert spread > 30.0, f"the diff arm's own cue only spans {spread:.1f} dB"

    # no two conditions identical, and nothing clips, re-derived from disk
    per_cell: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in rows:
        key = f"{row['item_id']}__{row['calibration']}__{row['masker']}"
        per_cell[key][str(row["condition"])] = Path(str(row["audio"]))
    assert len(per_cell) == 1000
    for key, paths in per_cell.items():
        assert paths["original"].read_bytes() != paths["diff"].read_bytes(), key
    peaks = [
        float(np.abs(sf.read(path, dtype="float32")[0]).max())
        for path in sorted((ARTIFACT / "audio").glob("*.wav"))
    ]
    assert len(peaks) == 2000
    assert max(peaks) <= 1.0

    # the design property the whole experiment rests on: the axes must separate
    separation = report["separation_at_matched_energy_db"]["median"]
    assert separation >= 5.0, (
        f"the two axes are only {separation:+.2f} dB apart at matched energy; a build this "
        f"collinear cannot tell energy from audibility"
    )
    for name in ("energy", "audibility"):
        slope = report["within_axis_slope_aud_vs_energy"][name]["median"]
        assert slope == pytest.approx(1.0, abs=0.15), (name, slope)
