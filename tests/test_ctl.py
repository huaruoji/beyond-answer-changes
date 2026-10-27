"""Tests for CTL item construction.

Every case here corresponds to a bug that actually shipped during round 4 and was
caught only by an audit or by a downstream failure, so the tests are written against
the failure mode rather than against the happy path:

* the event was added into a scene that already contained it (I7 saw a 27x full-scale
  disagreement between the file and the audited array);
* solving for the event gain produced a scene that clipped to death in PCM_16, so the
  rendered file no longer contained the event;
* the audibility threshold was computed from the scene span here and from the masker
  alone in the maskers module — two conventions, ~40 dB apart on the same item;
* the level solver bisected the wrong way and converged 6.5 dB off target.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audibility import audibility_margin_db
from src.ctl import (
    DEFAULT_CONDITIONS,
    audit_bit_identity,
    audit_sham,
    build_item,
    render_item,
    write_manifest,
)
from src.maskers import MaskerSpec, normalise_event
from src.synth import read_audio_mono

SR = 16000
FOREGROUND = Path("data/processed/synth/foreground_all")
BACKGROUND = Path("data/processed/synth/background")
VOCAB = Path("data/processed/synth/vocab_confirm.txt")

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


@pytest.fixture(scope="module")
def item_and_donor():
    labels, backgrounds = _corpus()
    if len(labels) < 2:
        pytest.skip("need at least two classes")
    rng = random.Random(11)
    query = labels[0]
    other = labels[1]
    event_path = sorted((FOREGROUND / query).glob("*.wav"))[0]
    donor_path = sorted((FOREGROUND / other).glob("*.wav"))[0]
    same_candidates = [f for f in sorted((FOREGROUND / query).glob("*.wav")) if f != event_path]
    if not same_candidates:
        pytest.skip(f"no second file under {query!r} to use as a same-class donor")
    same_path = same_candidates[0]
    item = build_item(
        item_id="t-00",
        query_class=query,
        donor_class=other,
        same_donor_class=query,
        event_source=event_path,
        background_source=rng.choice(backgrounds),
        donor_source=donor_path,
        seed=5,
    )
    donor, _ = read_audio_mono(donor_path, SR)
    same_donor, _ = read_audio_mono(same_path, SR)
    return item, donor, same_donor, query, other


def test_scene_contains_the_event_exactly_once(item_and_donor) -> None:
    item, _, _, _, _ = item_and_donor
    start, end = item.query_span
    for masker in item.maskers:
        scene = item.scene_for("equal_energy", masker)
        expected = (item.background + item.scaled_masker("equal_energy", masker))[
            start:end
        ][: item.event.size] + item.event
        assert np.allclose(scene[start:end], expected, atol=1e-6)
        # away from the span the scene must be exactly interferers, no event bleed
        outside = np.r_[scene[:start], scene[end:]]
        reference = np.r_[
            (item.background + item.scaled_masker("equal_energy", masker))[:start],
            (item.background + item.scaled_masker("equal_energy", masker))[end:],
        ]
        assert np.allclose(outside, reference, atol=1e-6)


def test_event_level_is_identical_across_maskers(item_and_donor) -> None:
    """The energy axis depends on this: only the masker may differ between cells."""

    item, _, _, _, _ = item_and_donor
    start, end = item.query_span
    spans = [item.scene_for("equal_energy", m)[start:end] for m in item.maskers]
    interferers = [
        item._interferer("equal_energy", m) for m in item.maskers
    ]
    # same event everywhere: subtracting the interferer must give the event back
    for span, interferer in zip(spans, interferers):
        recovered = span[: interferer.size] - interferer
        assert np.allclose(recovered, item.event, atol=1e-6)


def test_equal_energy_shares_one_masker_level(item_and_donor) -> None:
    item, _, _, _, _ = item_and_donor
    levels = {m: item.level_of("equal_energy", m) for m in item.maskers}
    assert len(set(round(v, 12) for v in levels.values())) == 1
    solved = {m: item.level_of("equal_audibility", m) for m in item.maskers}
    assert solved != levels, "the audibility arm must not reuse the shared level"


def test_threshold_excludes_the_event(item_and_donor) -> None:
    """audibility must match the documented formula, event not counted as interferer."""

    item, _, _, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    expected = audibility_margin_db(
        item.event, item._interferer("equal_energy", masker), item.sample_rate
    )
    assert item.audibility("equal_energy", masker) == pytest.approx(expected, abs=1e-9)
    # and the interferer must not contain the event
    assert item._interferer("equal_energy", masker).size == item.event.size


def test_rendered_original_matches_the_audited_scene(item_and_donor, tmp_path: Path) -> None:
    item, donor, same_donor, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    out = render_item(
        item,
        tmp_path,
        calibration="equal_energy",
        masker=masker,
        donor=donor,
        donor_span_samples=donor.size,
        same_donor=same_donor,
        same_donor_span_samples=same_donor.size,
    )
    result = audit_bit_identity(out["audio"]["original"], out["scene"], item.sample_rate)
    assert result["within_one_lsb"], result
    assert result["max_abs_delta"] <= 1.5 / 32768.0


def test_no_condition_clips(item_and_donor, tmp_path: Path) -> None:
    item, donor, same_donor, _, _ = item_and_donor
    for calibration in ("equal_energy", "equal_audibility"):
        for masker in item.maskers:
            out = render_item(
                item,
                tmp_path,
                calibration=calibration,
                masker=masker,
                donor=donor,
                donor_span_samples=donor.size,
                same_donor=same_donor,
                same_donor_span_samples=same_donor.size,
            )
            for condition, path in out["audio"].items():
                wave, _ = sf.read(path, dtype="float32")
                assert float(np.abs(wave).max()) <= 1.0, (calibration, masker, condition)


def test_sham_is_a_no_op(item_and_donor, tmp_path: Path) -> None:
    item, donor, same_donor, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    out = render_item(
        item,
        tmp_path,
        calibration="equal_energy",
        masker=masker,
        donor=donor,
        donor_span_samples=donor.size,
        same_donor=same_donor,
        same_donor_span_samples=same_donor.size,
    )
    sham, _ = sf.read(out["audio"]["sham"], dtype="float32")
    result = audit_sham(out["scene"], sham, item.query_span)
    assert result["max_abs_delta"] <= 1.5 / 32768.0, result


def test_edits_are_confined_to_the_query_span(item_and_donor, tmp_path: Path) -> None:
    """diff/same must rewrite the query span and nothing else (invariant I2/I3)."""

    item, donor, same_donor, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    out = render_item(
        item,
        tmp_path,
        calibration="equal_energy",
        masker=masker,
        donor=donor,
        donor_span_samples=donor.size,
        same_donor=same_donor,
        same_donor_span_samples=same_donor.size,
    )
    start, end = item.query_span
    original, _ = sf.read(out["audio"]["original"], dtype="float32")
    for condition in ("diff", "same"):
        edited, _ = sf.read(out["audio"][condition], dtype="float32")
        assert np.allclose(edited[:start], original[:start], atol=2 / 32768.0), condition
        assert np.allclose(edited[end:], original[end:], atol=2 / 32768.0), condition
        assert not np.allclose(edited[start:end], original[start:end], atol=1e-4), condition


def test_manifest_describes_the_audio_on_disk(item_and_donor, tmp_path: Path) -> None:
    """Every manifest row must exist on disk and carry finite, consistent numbers."""

    item, donor, same_donor, _, _ = item_and_donor
    rows: list[dict[str, object]] = []
    for calibration in ("equal_energy", "equal_audibility"):
        for masker in item.maskers:
            out = render_item(
                item,
                tmp_path,
                calibration=calibration,
                masker=masker,
                donor=donor,
                donor_span_samples=donor.size,
                same_donor=same_donor,
                same_donor_span_samples=same_donor.size,
            )
            rows.extend(out["rows"])  # type: ignore[arg-type]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    written = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(written) == len(rows) == 2 * len(item.maskers) * len(DEFAULT_CONDITIONS)
    for row in rows:
        assert Path(str(row["audio"])).exists()
        for key in ("audibility_db", "energy_snr_db", "masker_rms", "event_rms"):
            assert np.isfinite(float(row[key])), (key, row[key])
        assert 0.0 < float(row["masker_rms"]) < 100.0
        assert float(row["query_end_s"]) > float(row["query_start_s"])
        # the existing evaluator needs these two; without them a CTL manifest loads
        # but every question is empty
        assert str(row["question"]).startswith("Listen to the audio.")
        assert "Answer: A or Answer: B" in str(row["question"])
        assert set(row["option_events"]) == {"A", "B"}
        expected_truth = "B" if row["condition"] == "diff" else "A"
        assert row["ground_truth"] == expected_truth, row["condition"]


def test_event_rms_is_the_normalised_level(item_and_donor) -> None:
    item, _, _, _, _ = item_and_donor
    assert float(np.sqrt(np.mean(item.event.astype(np.float64) ** 2))) == pytest.approx(
        0.15, rel=1e-6
    )


def test_relative_audibility_target_is_reachable(item_and_donor) -> None:
    """The target is baseline - margin_drop, and the solver must actually land on it.

    An absolute target cannot work: the unmasked margin is anchored by the threshold
    in quiet and varies about 35 dB across items, so a fixed target is unreachable for
    some of them.
    """

    item, _, _, _, _ = item_and_donor
    realised = float(item.metadata["margin_drop_realised_db"])
    target = float(item.metadata["target_margin_db"])
    assert realised > 0.0
    assert target == pytest.approx(item.baseline_margin_db - realised, abs=1e-9)
    # the drop may fall back from the nominal value, but never exceed it
    assert realised <= float(item.metadata["margin_drop_nominal_db"]) + 1e-9
    achieved = item.audibility("equal_audibility", next(iter(item.maskers)))
    assert achieved == pytest.approx(target, abs=1.0)


def test_diff_and_same_arms_are_different_recordings(item_and_donor, tmp_path: Path) -> None:
    """The two edited arms must not be the same audio.

    This is the assertion whose absence let the branch ship a void contrast. From
    2026-09-16 to 2026-09-17 `render_item` was given one different-class donor and passed
    it to BOTH edited conditions, so every `__diff.wav` was byte-identical to its
    `__same.wav` while the manifest labelled them with opposite ground truths. MIG is
    identically zero for identical inputs, so the pilot's "0/240 identity tracking" and
    the stress ladder's flat MIG described the construction, not the models.
    """

    item, donor, same_donor, query, other = item_and_donor
    masker = next(iter(item.maskers))
    out = render_item(
        item,
        tmp_path,
        calibration="equal_energy",
        masker=masker,
        donor=donor,
        donor_span_samples=donor.size,
        same_donor=same_donor,
        same_donor_span_samples=same_donor.size,
    )
    diff, _ = sf.read(out["audio"]["diff"], dtype="float32")
    same, _ = sf.read(out["audio"]["same"], dtype="float32")
    assert not np.array_equal(diff, same), "diff and same are the same recording"
    assert float(np.abs(diff - same).max()) > 1e-3, "the arms differ only by rounding"

    rows = {r["condition"]: r for r in out["rows"]}  # type: ignore[union-attr]
    assert rows["diff"]["donor_used"] == other
    assert rows["same"]["donor_used"] == query
    assert rows["diff"]["donor_used"] != rows["same"]["donor_used"]


def test_same_condition_refuses_to_render_without_a_same_class_donor(item_and_donor, tmp_path: Path) -> None:
    """Falling back to the different-class donor is what caused the void contrast."""

    item, donor, _, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    with pytest.raises(AssertionError, match="same-class donor"):
        render_item(
            item,
            tmp_path,
            calibration="equal_energy",
            masker=masker,
            donor=donor,
            donor_span_samples=donor.size,
        )


def test_render_item_refuses_one_donor_array_for_both_edited_arms(item_and_donor, tmp_path: Path) -> None:
    """The builder-side differ-assertion cannot see two references to one array.

    CTL-Plan §3.4i#2: the check that ``diff`` and ``same`` differ lived only in the
    builder, so a caller passing the same donor twice still rendered two identical arms
    with opposite ground truths — the exact construction whose absence of a guard cost the
    branch two rounds of void MIG numbers. The guard belongs in the function that writes
    the files, and it has to catch both the same object and an equal copy.
    """

    item, donor, _, _, _ = item_and_donor
    masker = next(iter(item.maskers))
    for same in (donor, donor.copy()):
        with pytest.raises(AssertionError, match="same donor array"):
            render_item(
                item,
                tmp_path,
                calibration="equal_energy",
                masker=masker,
                donor=donor,
                donor_span_samples=donor.size,
                same_donor=same,
                same_donor_span_samples=same.size,
            )


def test_shared_gain_composes_across_scaled_copies(item_and_donor) -> None:
    """``shared_gain`` must record the cumulative gain, not the last factor.

    ``scaled_copy`` used to overwrite the field, so a cell fitted over several headroom
    iterations reported only its final factor. Measured on the superseded axis-duel build:
    a cell whose true gain was 0.435 reported 0.299.
    """

    item, _, _, _, _ = item_and_donor
    base_rms = float(np.sqrt(np.mean(item.event.astype(np.float64) ** 2)))
    once = item.scaled_copy(0.5)
    twice = once.scaled_copy(0.5)
    assert float(once.metadata["shared_gain"]) == pytest.approx(0.5, rel=1e-12)
    assert float(twice.metadata["shared_gain"]) == pytest.approx(0.25, rel=1e-12)
    assert [float(v) for v in twice.metadata["shared_gain_steps"]] == [1.0, 0.5, 0.5]
    assert np.sqrt(np.mean(twice.event.astype(np.float64) ** 2)) == pytest.approx(
        0.25 * base_rms, rel=1e-5
    )


def test_fit_donor_level_reference_defaults_to_the_span() -> None:
    """The mainline builds and tests are written against the span reference; keep it the default."""

    from src.synth import _fit_donor

    donor = np.full(400, 0.5, dtype=np.float32)
    span = np.full(200, 0.05, dtype=np.float32)
    event = np.full(200, 0.2, dtype=np.float32)
    default = _fit_donor(donor, 400, 200, span)
    explicit = _fit_donor(donor, 400, 200, span, level_reference=span)
    assert np.allclose(default, explicit)
    assert float(np.sqrt(np.mean(default.astype(np.float64) ** 2))) == pytest.approx(0.05, rel=1e-6)
    # an explicit reference changes the level and nothing else
    re_levelled = _fit_donor(donor, 400, 200, span, level_reference=event)
    assert re_levelled.shape == default.shape
    assert float(np.sqrt(np.mean(re_levelled.astype(np.float64) ** 2))) == pytest.approx(
        0.2, rel=1e-6
    )


def test_render_item_levels_the_donor_to_the_event_not_the_masker(item_and_donor, tmp_path: Path) -> None:
    """FIX 1, end to end: the replacement must carry the event's level, not the masker's.

    Levelling to the masked span pinned the replacement to the masker, measured up to
    +27.5 dB above the event it replaced, with its own audibility flat across a 30 dB
    ladder (CTL-Plan §3.4i#3, §3.4j). This cell is deliberately masker-dominated, so the
    old convention is visibly wrong here and the assertion is not vacuous.
    """

    item, donor, _, _, _ = item_and_donor
    start, end = item.query_span
    event_rms = float(np.sqrt(np.mean(item.event.astype(np.float64) ** 2)))
    # the most masker-dominated cell of this item, chosen by measurement, so the old
    # span-matching convention would be off by tens of dB here rather than by rounding
    calibration, masker = max(
        ((cal, m) for cal in ("equal_energy", "equal_audibility") for m in item.maskers),
        key=lambda cell: float(
            np.sqrt(np.mean(item.scene_for(*cell)[start:end].astype(np.float64) ** 2))
        ),
    )
    out = render_item(
        item,
        tmp_path,
        calibration=calibration,
        masker=masker,
        conditions=("original", "diff"),
        donor=donor,
        donor_span_samples=donor.size,
    )
    original, _ = sf.read(out["audio"]["original"], dtype="float32")
    diff, _ = sf.read(out["audio"]["diff"], dtype="float32")
    # the cell needed a shared headroom gain, so the event in the written file is scaled
    # too; compare against the scaled event rather than the build-scale one
    gain = float(out["rows"][0]["meta_shared_gain"])  # type: ignore[index]
    event_rms_written = event_rms * gain
    # measure inside the span but clear of the 10 ms crossfades: the fades legitimately
    # keep a little of the original masker at each edge, which would dominate the RMS here
    guard = int(0.02 * item.sample_rate)
    lo, hi = start + guard, end - guard
    span_rms = float(np.sqrt(np.mean(original[lo:hi].astype(np.float64) ** 2)))
    donor_rms = float(np.sqrt(np.mean(diff[lo:hi].astype(np.float64) ** 2)))
    # setup: the span is much louder than the event, so span-matching would be a big error
    assert 20.0 * np.log10(span_rms / event_rms_written) > 6.0, "needs a masker-dominated span"
    # and the replacement is at the event's level, not the span's
    assert abs(20.0 * np.log10(donor_rms / event_rms_written)) < 1.5, (donor_rms, event_rms_written)


def test_manifest_records_the_level_of_the_signal_in_the_queried_span(item_and_donor, tmp_path: Path) -> None:
    """Every row must describe what is actually in the queried span of the file it names.

    The event-referenced ``audibility_db`` is kept for compatibility, but on an edited arm
    it describes the event that was overwritten. The ``span_*`` pair is recomputed here
    from the WAV on disk, so a row that disagrees with its audio fails.
    """

    item, donor, _, _, _ = item_and_donor
    start, end = item.query_span
    # again the masker-dominated cell: only there do the two conventions really differ
    calibration, masker = max(
        ((cal, m) for cal in ("equal_energy", "equal_audibility") for m in item.maskers),
        key=lambda cell: float(
            np.sqrt(np.mean(item.scene_for(*cell)[start:end].astype(np.float64) ** 2))
        ),
    )
    out = render_item(
        item,
        tmp_path,
        calibration=calibration,
        masker=masker,
        conditions=("original", "diff"),
        donor=donor,
        donor_span_samples=donor.size,
    )
    rows = {str(r["condition"]): r for r in out["rows"]}  # type: ignore[union-attr]
    # the written array may carry a shared headroom gain; the row records it, and the
    # interferer has to be reconstructed at the same scale or the comparison is 0.6 dB off
    gain = float(rows["original"].get("meta_shared_gain", 1.0))
    scaled = item.scaled_copy(gain)
    interferer = scaled._interferer(calibration, masker)
    guard = int(round(10.0 * item.sample_rate / 1000.0))  # the crossfade length
    for condition, row in rows.items():
        assert row["span_audibility_db_reference"] == "queried_span_cue"
        assert row["audibility_db_reference"] == "event"
        on_disk, _ = sf.read(out["audio"][condition], dtype="float32")
        span = on_disk[start:end]
        # the cue in the span: the event where the span was kept, the replacement where the
        # splice overwrote it (the splice removes the interferer from the span)
        replaced = condition in {"diff", "same"}
        assert row["span_cue_signal"] == ("replacement" if replaced else "event")
        cue = span if replaced else (span - interferer).astype(np.float32)
        cue_slice = cue[guard:-guard]
        interferer_slice = interferer[guard:-guard]
        expected_audibility = audibility_margin_db(cue_slice, interferer_slice, item.sample_rate)
        expected_energy = 20.0 * np.log10(
            float(np.sqrt(np.mean(cue_slice.astype(np.float64) ** 2)))
            / float(np.sqrt(np.mean(interferer_slice.astype(np.float64) ** 2)))
        )
        assert float(row["span_audibility_db"]) == pytest.approx(expected_audibility, abs=0.1), condition
        assert float(row["span_energy_snr_db"]) == pytest.approx(expected_energy, abs=0.05), condition
    # where the span was kept, the span measurement reproduces the event-referenced one
    assert float(rows["original"]["span_audibility_db"]) == pytest.approx(
        float(rows["original"]["audibility_db"]), abs=1.0
    )
    assert float(rows["original"]["span_energy_snr_db"]) == pytest.approx(
        float(rows["original"]["energy_snr_db"]), abs=1.0
    )
    # the replacement carries the event's level (FIX 1): its recorded span RMS matches the
    # scaled event's, not the masker-dominated span's, whatever the audibility floor does
    assert float(rows["diff"]["span_rms"]) == pytest.approx(
        float(rows["diff"]["event_rms"]), rel=0.15
    )
    assert float(rows["original"]["span_rms"]) == pytest.approx(
        float(rows["original"]["event_rms"]), rel=0.15
    )
    # while the raw written span (event + loud masker) is far louder than the cue in it
    original_on_disk, _ = sf.read(out["audio"]["original"], dtype="float32")
    mixture_rms = float(np.sqrt(np.mean(original_on_disk[start:end].astype(np.float64) ** 2)))
    assert mixture_rms > 2.0 * float(rows["original"]["event_rms"])
    assert rows["original"]["span_cue_signal"] == "event"
    assert rows["diff"]["span_cue_signal"] == "replacement"
