"""Tests for the CTL masker generator and matched-pair construction.

The branch's two load-bearing contrasts are only meaningful if these hold:

* **equal energy, different audibility** — the two conditions must have *identical*
  event energy and a large audibility gap;
* **equal audibility, different energy** — the two conditions must land on the
  *same* margin while their energies differ.

Measured on the first real run (6 s events, masker RMS 0.05, target margin +6 dB):
a 3 kHz tone under low-pass vs white maskers separates by 27.9 dB of audibility,
and the equal-audibility pair lands at +6.00 dB on both sides with a 27.9 dB energy
gap. A 150 Hz tone separates by 13.5 dB in the opposite direction (the low-pass
masker masks *it* better). Both directions are asserted below.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audibility import audibility_margin_db
from src.maskers import (
    MASKER_KINDS,
    MaskerSpec,
    event_matched_masker,
    normalise_event,
    shaped_noise,
    solve_event_gain,  # legacy, exercised only to keep its contract honest
    solve_masker_attenuation_db,
    target_rms,
)

SR = 16000
DURATION = 6.0


def _tone(freq: float, amplitude: float = 0.1, seconds: float = DURATION) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    window = np.hanning(t.size)
    return (amplitude * np.sin(2 * np.pi * freq * t) * window).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


@pytest.mark.parametrize("kind", MASKER_KINDS)
def test_shaped_noise_has_requested_rms_and_reproduces(kind: str) -> None:
    spec = MaskerSpec(kind=kind, rms=0.03, seed=7, cutoff_hz=500.0 if "pass" in kind else None)
    event = _tone(1000.0) if kind == "event_matched" else None
    a = shaped_noise(DURATION, SR, spec, event=event)
    b = shaped_noise(DURATION, SR, spec, event=event)
    assert _rms(a) == pytest.approx(0.03, rel=1e-6)
    assert np.array_equal(a, b), "maskers must be reproducible from the spec"
    assert a.shape == (int(SR * DURATION),)


def test_low_pass_masker_stays_below_its_cutoff() -> None:
    spec = MaskerSpec(kind="low_pass", rms=0.05, seed=3, cutoff_hz=400.0)
    wave = shaped_noise(DURATION, SR, spec)
    spec_pow = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(wave.size, d=1.0 / SR)
    low = spec_pow[freqs < 400.0].sum()
    high = spec_pow[freqs > 2000.0].sum()
    assert low > 100 * high, (low, high)


def test_target_rms_implements_the_snr_definition() -> None:
    masker = np.full(1000, 0.05, dtype=np.float32)
    assert target_rms(masker, 0.0) == pytest.approx(0.05, rel=1e-6)
    assert target_rms(masker, -20.0) == pytest.approx(0.005, rel=1e-6)
    # +6 dB is a factor of 10**(6/20) = 1.9953, i.e. 0.09976 — not 0.1. An earlier
    # assertion demanded exactly 0.1 and failed on a correctly behaving function.
    assert target_rms(masker, +6.0) == pytest.approx(0.05 * 10 ** (6 / 20), rel=1e-6)


def test_event_matched_masker_tracks_the_event_spectrum() -> None:
    """Its energy must sit where the event sits, unlike white noise."""

    event = _tone(3000.0)
    matched = event_matched_masker(event, SR, rms=0.05, seed=5)
    white = shaped_noise(DURATION, SR, MaskerSpec(kind="white", rms=0.05, seed=5))

    def band_share(wave: np.ndarray) -> float:
        power = np.abs(np.fft.rfft(wave)) ** 2
        freqs = np.fft.rfftfreq(wave.size, d=1.0 / SR)
        near = (freqs > 2000.0) & (freqs < 4500.0)
        return float(power[near].sum() / power.sum())

    assert band_share(matched) > band_share(white) * 2.0


def test_normalise_event_scales_to_the_target_rms() -> None:
    raw = _tone(1000.0, amplitude=0.01)
    out = normalise_event(raw, 0.15)
    assert float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) == pytest.approx(0.15, rel=1e-6)
    assert np.allclose(out / out.std(), raw / raw.std(), atol=1e-5)


def test_masker_solver_hits_its_target() -> None:
    event = normalise_event(_tone(3000.0), 0.15)
    masker = shaped_noise(DURATION + 1.0, SR, MaskerSpec(kind="white", rms=1.0, seed=3))
    for target in (-10.0, 0.0, 10.0):
        _, rms = solve_masker_attenuation_db(
            event, masker, SR, reference_rms=1.0, target_margin_db=target
        )
        achieved = audibility_margin_db(event, (masker / masker.std() * rms).astype(np.float32), SR)
        assert achieved == pytest.approx(target, abs=0.05), (target, achieved, rms)


def test_legacy_gain_solver_still_hits_its_target() -> None:
    """The legacy path is retained for the probe's readings; keep its contract honest."""

    event = _tone(3000.0)
    masker = shaped_noise(DURATION, SR, MaskerSpec(kind="white", rms=0.05, seed=9))
    for target in (-6.0, 0.0, 6.0):
        gain = solve_event_gain(event, masker, SR, target_margin_db=target)
        achieved = audibility_margin_db((event * gain).astype(np.float32), masker, SR)
        assert achieved == pytest.approx(target, abs=0.05)
