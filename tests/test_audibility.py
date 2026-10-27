"""Tests for the CTL audibility score.

These lock the three properties the CTL design depends on:

1. **Parseval calibration** — band powers are mean-square values, so the score is
   comparable across maskers and levels. Without it the event/threshold ratio is
   off by the FFT length (measured: exactly n_fft in the first prototype).
2. **Masker-spectrum discrimination** — two maskers with identical energy but
   different spectra must produce clearly different audibility for the same event.
   This is the whole point of the branch: energy and audibility have to come apart.
3. **Monotonicity in masker level** — more masker energy must lower audibility.

Absolute anchoring is *not* tested, because the implementation is only defined up
to a constant offset (see the module docstring). Everything CTL measures is a
comparison across conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audibility import (
    audibility_margin_db,
    bark_to_hz,
    hz_to_bark,
    masking_threshold_bark,
    periodogram_bark,
)

SR = 16000


def _tone(freq: float, seconds: float = 2.0, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _lowpass_noise(seconds: float = 2.0, rms: float = 0.05, seed: int = 0) -> np.ndarray:
    """Noise with almost all energy below ~500 Hz: a poor masker for a 3 kHz tone.

    Scaled so the *requested* rms is the rms actually delivered — an earlier
    version multiplied by ``rms`` and then divided by ``x.std()``, which cancelled
    the parameter and made two different levels identical.
    """

    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(int(SR * seconds)), np.ones(256) / 256, mode="same")
    x = x / float(np.sqrt(np.mean(x ** 2)))
    return (x * rms).astype(np.float32)


def _broadband_noise(seconds: float = 2.0, rms: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * seconds))
    x = x / float(np.sqrt(np.mean(x ** 2)))
    return (x * rms).astype(np.float32)


def test_bark_roundtrip_is_exact() -> None:
    """The pair must be algebraic inverses; the previous pair was off by 34%."""

    for hz in (50.0, 100.0, 500.0, 1000.0, 3000.0, 6000.0, 8000.0):
        assert bark_to_hz(hz_to_bark(hz)) == pytest.approx(hz, rel=1e-6)


def test_periodogram_recovers_mean_square_of_full_scale_sine() -> None:
    """A full-scale sine has mean square 0.5 and must sum to 0.5 across bands."""

    total = float(periodogram_bark(_tone(1000.0, amplitude=1.0), SR).sum())
    assert total == pytest.approx(0.5, rel=0.02)


def test_periodogram_recovers_white_noise_variance() -> None:
    noise = _broadband_noise(rms=0.1, seed=3)
    total = float(periodogram_bark(noise, SR).sum())
    assert total == pytest.approx(float(np.mean(noise ** 2)), rel=0.05)


def test_equal_energy_maskers_differ_in_audibility() -> None:
    """The branch's core manipulation: same masker energy, different spectrum."""

    event = _tone(3000.0)
    low = _lowpass_noise(rms=0.05, seed=1)
    wide = _broadband_noise(rms=0.05, seed=1)
    # float32 cast costs ~1% on the smoothed masker; energy match is what matters
    assert float(low.std()) == pytest.approx(float(wide.std()), rel=0.02)

    m_low = audibility_margin_db(event, low, SR)
    m_wide = audibility_margin_db(event, wide, SR)
    # A 3 kHz tone is far more audible under a masker confined below ~500 Hz.
    assert m_low > m_wide + 10.0, (m_low, m_wide)


def test_audibility_is_monotonic_in_masker_level() -> None:
    event = _tone(3000.0)
    margins = [
        audibility_margin_db(event, _broadband_noise(rms=rms, seed=2), SR)
        for rms in (0.01, 0.03, 0.1, 0.3)
    ]
    assert margins == sorted(margins, reverse=True), margins


def test_threshold_scales_with_masker_power_in_the_event_band() -> None:
    """More masker energy must raise the threshold in the *event's* band.

    Comparing at the threshold's own maximum is meaningless here: that peak sits in
    the lowest bands, where both maskers are pinned to the threshold-in-quiet floor
    and are therefore identical. Audibility is governed by the event's band.
    """

    event = _tone(3000.0)
    quiet = _broadband_noise(rms=0.01, seed=4)
    loud = _broadband_noise(rms=0.02, seed=4)
    assert float(loud.std()) == pytest.approx(2 * float(quiet.std()), rel=1e-3)

    t_quiet = masking_threshold_bark(quiet, SR)
    t_loud = masking_threshold_bark(loud, SR)
    band = int(np.argmax(periodogram_bark(event, SR)))
    assert t_loud[band] > t_quiet[band] * 1.5, (t_quiet[band], t_loud[band])


def test_no_masker_is_more_audible_than_any_masker() -> None:
    event = _tone(3000.0)
    silent = np.zeros(int(SR * 2.0), dtype=np.float32)
    noisy = _broadband_noise(rms=0.05, seed=5)
    assert audibility_margin_db(event, silent, SR) > audibility_margin_db(event, noisy, SR)
