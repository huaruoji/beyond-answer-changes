"""Regression and invariant tests for the synthetic instrument.

The central test is :func:`test_tail_crossfade_is_continuous`. The withdrawn
pre-2026-09-12 results were produced by an edit primitive whose tail weight was
reversed, so every edit ended in two full-scale discontinuities. That test is
written so it fails on that implementation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.synth import (
    CONDITIONS,
    MIN_EVENT_GAP_S,
    InstrumentError,
    PlacedEvent,
    Soundscape,
    apply_condition,
    assert_placement_disjoint,
    boundary_jump,
    background_estimate,
    crossfade_replace,
    local_background_rms,
    event_rms_from_span,
    find_free_window,
    mix_into_background,
    _fit_donor,
    plan_placements,
)


SR = 16000
DURATION = 2.0
#: A clip long enough to hold two separated events *and* a length-matched
#: event-free window for the matched control. The design's real clip is 10 s;
#: item acceptance requires that window to exist, so tests that exercise the
#: control conditions must not use a cramped fixture.
LONG = 4.0


def _tone(freq: float = 100.0, seconds: float = DURATION, amplitude: float = 0.5):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _scape(events: list[PlacedEvent], duration: float = DURATION) -> Soundscape:
    return Soundscape(
        audio_path=Path("/dev/null"),
        sample_rate=SR,
        duration_s=duration,
        background_path="background.wav",
        events=events,
    )


def _query(start: float = 0.4, end: float = 0.9) -> PlacedEvent:
    return PlacedEvent("query", "dog_bark", "q.wav", start, end, 10.0, end - start)


def _nuisance(start: float = 1.2, end: float = 1.6) -> PlacedEvent:
    return PlacedEvent("nuisance", "siren", "n.wav", start, end, 10.0, end - start)


# --- I4: boundary continuity -------------------------------------------------


def test_tail_crossfade_is_continuous() -> None:
    """The withdrawn results came from a reversed tail weight. Pin the fix.

    A correct tail runs replacement -> original, so the two edges of the tail
    region are continuous. The reversed implementation put the *original* signal
    at the inner edge, producing a full-scale jump twice.
    """

    waveform = _tone()
    start, end = 4000, 9000
    replacement = np.zeros(end - start, dtype=np.float32)
    out = crossfade_replace(waveform, start, end, replacement, fade_samples=160)

    own_step = float(np.abs(np.diff(waveform)).max())
    worst = boundary_jump(out, start, end)
    assert worst <= 3.0 * own_step, (
        f"boundary discontinuity {worst:.4f} exceeds 3x the signal's own step "
        f"{own_step:.4f}; the tail crossfade is inverted"
    )


def test_tail_crossfade_actually_reaches_original_at_the_end() -> None:
    """The tail's last sample must meet the untouched signal, not the fill."""

    waveform = _tone()
    start, end = 4000, 9000
    out = crossfade_replace(
        waveform, start, end, np.zeros(end - start, dtype=np.float32), fade_samples=160
    )
    assert out[end - 1] == pytest.approx(waveform[end - 1], abs=1e-3)
    assert out[end] == pytest.approx(waveform[end], abs=1e-6)
    assert out[start - 1] == pytest.approx(waveform[start - 1], abs=1e-6)
    assert out[start] == pytest.approx(waveform[start], abs=1e-3)


def test_tail_midpoint_is_still_the_replacement() -> None:
    """Sanity: the crossfade must not eat into the middle of the span."""

    waveform = _tone()
    start, end = 4000, 9000
    out = crossfade_replace(
        waveform, start, end, np.zeros(end - start, dtype=np.float32), fade_samples=160
    )
    assert np.allclose(out[start + 200 : end - 200], 0.0, atol=1e-6)


def test_sham_condition_is_effectively_a_noop() -> None:
    """Rewriting a span with its own samples must not perturb the audio."""

    waveform = _tone()
    scape = _scape([_query()])
    out = apply_condition(waveform, scape, "sham", rng=random.Random(0))
    assert np.allclose(out, waveform, atol=1e-6)


# --- I1: a donor is an event, never a clip -----------------------------------


def test_donor_is_trimmed_to_its_annotated_span() -> None:
    """Only the donor's span may reach the target; the rest must be discarded."""

    donor = np.concatenate(
        [np.full(1000, 9.0, dtype=np.float32), np.full(500, -1.0, dtype=np.float32)]
    )
    reference = np.full(250, 0.25, dtype=np.float32)
    fitted = _fit_donor(donor, donor_span_samples=500, target_samples=250, reference=reference)
    assert fitted.shape == (250,)
    # Everything came from the -1.0 region, so nothing carries the 9.0 sentinel.
    assert np.abs(fitted).max() < 1.0


def test_donor_level_is_matched_to_the_span_it_replaces() -> None:
    donor = np.full(500, 0.01, dtype=np.float32)
    reference = np.full(250, 0.5, dtype=np.float32)
    fitted = _fit_donor(donor, 500, 250, reference)
    donor_rms = float(np.sqrt(np.mean(np.square(fitted, dtype=np.float64))))
    assert donor_rms == pytest.approx(0.5, rel=1e-3)


def test_donor_span_outside_the_donor_is_rejected() -> None:
    with pytest.raises(InstrumentError):
        _fit_donor(np.zeros(100, dtype=np.float32), 500, 50, np.zeros(50, dtype=np.float32))


def test_replacement_conditions_require_a_donor() -> None:
    scape = _scape([_query()])
    for condition in ("target_replaced_diff", "target_replaced_same"):
        with pytest.raises(InstrumentError):
            apply_condition(_tone(), scape, condition, rng=random.Random(0))


# --- I2: control windows never overlap the evidence --------------------------


def test_free_window_never_overlaps_any_event() -> None:
    events = [_query(0.2, 0.5), _nuisance(1.0, 1.3)]
    length = int(0.3 * SR)
    rng = random.Random(7)
    for _ in range(200):
        start, end = find_free_window(events, int(DURATION * SR), SR, length, rng)
        for event in events:
            assert end <= event.start_s * SR or start >= event.end_s * SR, (
                "control window overlapped an event"
            )


def test_free_window_honours_the_guard_band() -> None:
    events = [_query(0.5, 0.8)]
    length = int(0.2 * SR)
    start, end = find_free_window(
        events, int(DURATION * SR), SR, length, random.Random(1), guard_s=0.3
    )
    guard = int(0.3 * SR)
    assert end <= 0.5 * SR - guard or start >= 0.8 * SR + guard


def test_free_window_raises_when_the_clip_is_full() -> None:
    events = [_query(0.0, 1.0), _nuisance(1.0, 2.0)]
    with pytest.raises(InstrumentError):
        find_free_window(
            events, int(DURATION * SR), SR, int(0.5 * SR), random.Random(0)
        )


def test_background_removed_does_not_touch_the_query_span() -> None:
    """The matched control must edit somewhere else entirely."""

    waveform = _tone(seconds=LONG)
    scape = _scape([_query(1.0, 1.5), _nuisance(2.5, 2.9)], duration=LONG)
    out = apply_condition(waveform, scape, "background_removed", rng=random.Random(3))
    start, end = int(1.0 * SR), int(1.5 * SR)
    assert np.allclose(out[start:end], waveform[start:end], atol=1e-6)
    assert not np.allclose(out, waveform)


def test_nuisance_removed_leaves_the_query_span_intact() -> None:
    waveform = _tone(seconds=LONG)
    scape = _scape([_query(1.0, 1.5), _nuisance(2.5, 2.9)], duration=LONG)
    out = apply_condition(waveform, scape, "nuisance_removed", rng=random.Random(3))
    qs, qe = int(1.0 * SR), int(1.5 * SR)
    ns, ne = int(2.5 * SR), int(2.9 * SR)
    assert np.allclose(out[qs:qe], waveform[qs:qe], atol=1e-6)
    assert not np.allclose(out[ns:ne], waveform[ns:ne])


def test_item_is_rejected_when_no_matched_control_window_fits() -> None:
    """A cramped clip must fail loudly rather than fall back to a bad control.

    This is the item-acceptance rule: the generator drops such items instead of
    placing the control somewhere that overlaps the evidence, which is exactly
    how 40 of 80 controls in the withdrawn L2 set came to overlap the target.
    """

    waveform = _tone(seconds=DURATION)
    scape = _scape([_query(0.4, 0.9), _nuisance(1.3, 1.6)])
    with pytest.raises(InstrumentError):
        apply_condition(waveform, scape, "background_removed", rng=random.Random(3))


def test_placement_disjoint_rejects_overlapping_events() -> None:
    scape = _scape([_query(0.5, 1.0), _nuisance(0.9, 1.2)])
    with pytest.raises(InstrumentError):
        assert_placement_disjoint(scape)


def test_placement_disjoint_rejects_a_violated_guard_band() -> None:
    gap = MIN_EVENT_GAP_S / 2
    scape = _scape([_query(0.4, 0.8), _nuisance(0.8 + gap, 1.2)])
    with pytest.raises(InstrumentError):
        assert_placement_disjoint(scape)


# --- placement planning ------------------------------------------------------


def test_planned_placements_are_disjoint_and_inside_the_clip() -> None:
    rng = random.Random(11)
    for _ in range(50):
        spans = plan_placements(
            int(LONG * SR), SR, 0.5, [0.4, 0.4], rng, duration_s=LONG
        )
        assert spans, "placement should normally succeed"
        ordered = sorted(spans)
        assert ordered[0][0] >= 0.0
        assert ordered[-1][1] <= LONG + 1e-9
        for (_, end_a), (start_b, _) in zip(ordered, ordered[1:]):
            assert start_b - end_a >= MIN_EVENT_GAP_S - 1e-6


def test_placement_returns_empty_when_events_cannot_fit() -> None:
    spans = plan_placements(
        int(DURATION * SR), SR, 1.5, [1.5, 1.5], random.Random(0), duration_s=DURATION
    )
    assert spans == []


# --- condition inventory -----------------------------------------------------


def test_silence_zeroes_the_clip() -> None:
    scape = _scape([_query()])
    out = apply_condition(_tone(), scape, "silence", rng=random.Random(0))
    assert not out.any()


def test_original_is_unchanged() -> None:
    waveform = _tone()
    scape = _scape([_query()])
    out = apply_condition(waveform, scape, "original", rng=random.Random(0))
    assert np.array_equal(out, waveform)


def test_condition_list_matches_the_design_document() -> None:
    assert CONDITIONS == (
        "original",
        "target_replaced_diff",
        "target_replaced_same",
        "nuisance_removed",
        "background_removed",
        "sham",
        "silence",
    )


def test_unknown_condition_is_rejected() -> None:
    scape = _scape([_query()])
    with pytest.raises(InstrumentError):
        apply_condition(_tone(), scape, "nonsense", rng=random.Random(0))


# --- background preservation in replacements ---------------------------------
#
# A listener (2026-09-21, confirm split) reported that a substituted sound has no
# noise floor: "替换的异类声音没有底噪了，和环境音区别有点明显". That was correct.
# The replacement overwrote the span with donor-only samples, so the clip's
# background vanished from that span. Measured: the span's noise floor fell by a
# median of ~15 dB across the grid, >10 dB on 68% of items.


def _noise_floor_db(x: np.ndarray, frame: int = 320) -> float:
    """10th-percentile frame RMS in dBFS: the floor between event onsets."""

    n = len(x) // frame
    assert n > 0
    frames = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n)]
    )
    return float(20 * np.log10(np.percentile(frames, 10) + 1e-12))


def test_mix_into_background_restores_the_floor() -> None:
    """A donor with quiet gaps must end up with the background in those gaps.

    The realistic case: an event clip is loud at its onset and quiet afterwards.
    Overwriting the span with it leaves those quiet parts *silent*, which is what
    the listener heard. Mixing the background back in fills them.
    """

    rng = np.random.default_rng(0)
    # steady background, and a donor that is a brief burst plus near-silence
    background = rng.normal(0, 0.02, 16000).astype(np.float32)
    donor = np.concatenate([
        rng.normal(0, 0.20, 1600),      # onset
        rng.normal(0, 0.0002, 6400),    # near-silent tail
    ]).astype(np.float32)
    bg_rms = float(np.sqrt(np.mean(background**2)))
    span_rms = float(np.sqrt(bg_rms**2 + 0.06**2))
    event_rms = event_rms_from_span(span_rms, bg_rms)

    donor_only = _noise_floor_db(donor)
    mixed = mix_into_background(donor, background, bg_rms, event_rms)
    assert _noise_floor_db(mixed) > donor_only + 10.0, (
        f"the background should fill the donor's quiet tail: "
        f"donor-only {donor_only:.1f} dB, mixed {_noise_floor_db(mixed):.1f} dB"
    )


def test_mix_into_background_preserves_span_power() -> None:
    """The span's total power must survive, or the SNR calibration drifts."""

    rng = np.random.default_rng(1)
    background = rng.normal(0, 0.02, 16000).astype(np.float32)
    donor = rng.normal(0, 0.05, 8000).astype(np.float32)
    bg_rms = float(np.sqrt(np.mean(background**2)))
    target_span_rms = 0.06
    event_rms = event_rms_from_span(target_span_rms, bg_rms)

    mixed = mix_into_background(donor, background, bg_rms, event_rms)
    got = float(np.sqrt(np.mean(mixed**2)))
    assert got == pytest.approx(target_span_rms, rel=0.15), (
        f"span RMS drifted: wanted ~{target_span_rms:.4f}, got {got:.4f}"
    )


def test_mix_into_background_falls_back_when_no_estimate() -> None:
    """With no background estimate the donor must pass through unchanged."""

    donor = np.full(1000, 0.3, dtype=np.float32)
    assert np.allclose(mix_into_background(donor, np.zeros(0, dtype=np.float32), 0.0, 0.1), donor)
    assert np.allclose(mix_into_background(donor, np.zeros(1000, dtype=np.float32), 0.0, 0.1), donor)
    assert np.allclose(mix_into_background(donor, np.zeros(1000, dtype=np.float32), 0.01, 0.0), donor)


def test_event_rms_is_clamped_at_zero() -> None:
    """A span quieter than the background estimate must not give imaginary power."""

    assert event_rms_from_span(0.01, 0.05) == 0.0
    assert event_rms_from_span(0.05, 0.01) == pytest.approx(
        np.sqrt(0.05**2 - 0.01**2), rel=1e-6
    )


def test_background_estimate_avoids_every_event() -> None:
    """The estimate window must not contain an event, or it is not background."""

    scape = _scape([_query(1.0, 2.0)], duration=LONG)
    waveform = _tone(seconds=LONG)
    rng = random.Random(0)
    window, rms = background_estimate(
        waveform, scape.events, (SR, 2 * SR), SR, rng
    )
    assert rms > 0.0
    assert window.size == SR


def test_local_background_rms_reads_the_quiet_frames() -> None:
    """The in-span estimate must find the background, not the event.

    Preferred over a distant event-free window because TUT noise is not
    stationary: on the confirm grid the far window exceeded the span's own RMS
    for 10/600 items, forcing a fallback to donor-only.
    """

    rng = np.random.default_rng(3)
    true_bg = 0.02
    span = np.concatenate([
        rng.normal(0, 0.15, 1600),        # event onset
        rng.normal(0, true_bg, 1600 * 4),  # background between/after onsets
    ]).astype(np.float32)
    got = local_background_rms(span)
    assert got == pytest.approx(true_bg, rel=0.5), (
        f"expected ~{true_bg}, got {got}"
    )
    # and it must be far below the span's own RMS, which the event inflates
    span_rms = float(np.sqrt(np.mean(span**2)))
    assert got < span_rms * 0.6


def test_local_background_rms_handles_short_input() -> None:
    assert local_background_rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_background_estimate_returns_nothing_when_clip_is_full() -> None:
    """No event-free window means no estimate, and the caller must cope."""

    scape = _scape([_query(0.0, LONG)], duration=LONG)
    waveform = _tone(seconds=LONG)
    window, rms = background_estimate(
        waveform, scape.events, (0, SR), SR, random.Random(0)
    )
    assert window.size == 0 and rms == 0.0


def test_replacement_keeps_the_background_in_the_span() -> None:
    """End to end: a replacement must not silence the span's noise floor.

    This is the regression for the defect a listener heard on 2026-09-21. The
    clip has a steady background; the replacement is a burst with a quiet tail.
    Overwriting the span with the donor alone leaves that tail silent, so the
    floor collapses. The fixture must therefore have a *background*, which a pure
    tone does not — an earlier version of this test used one and could not fail.
    """

    rng = np.random.default_rng(7)
    seconds = LONG
    n = int(seconds * SR)
    # steady low-level background + a query event, i.e. what a real item is
    background = rng.normal(0, 0.02, n).astype(np.float32)
    query = np.zeros(n, dtype=np.float32)
    qs, qe = SR, 2 * SR
    t = np.arange(qe - qs, dtype=np.float32) / SR
    query[qs:qe] = (0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    waveform = (background + query).astype(np.float32)

    scape = _scape([_query(1.0, 2.0)], duration=seconds)
    # donor: a burst then near-silence, the shape that exposes the defect
    donor = np.concatenate([
        rng.normal(0, 0.25, 1600),
        rng.normal(0, 0.0002, SR - 1600),
    ]).astype(np.float32)

    out = apply_condition(
        waveform, scape, "target_replaced_diff",
        donor_waveform=donor, donor_span_samples=SR, rng=random.Random(0),
    )
    # The right baseline is the clip's own background floor, not the original
    # span's: the queried event fills its span, so the original has no quiet
    # frames to compare against. The defect is that the replacement's quiet tail
    # was digital silence rather than background.
    background_floor = _noise_floor_db(waveform[: SR // 2])
    after = _noise_floor_db(out[qs:qe])
    assert after > background_floor - 6.0, (
        f"the replacement's quiet part is not background: "
        f"background floor {background_floor:.1f} dB, replacement floor {after:.1f} dB"
    )
