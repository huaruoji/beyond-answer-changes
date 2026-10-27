"""Masker generation and level calibration for the CTL branch.

The CTL design needs two matched contrasts that a single broadband SNR knob cannot
build:

* **equal energy, different audibility** — same masker RMS, different masker
  spectrum, so the event sits at a very different distance from the masking
  threshold;
* **equal audibility, different energy** — the event is driven to the same
  distance from the threshold, which forces different event energies because the
  maskers mask differently.

This module supplies both. The gain solver is exact rather than iterative: band
power and threshold both scale as ``gain**2``, so the margin moves as
``20*log10(gain)`` and can be solved in closed form. An iterative search would work
too but would make the calibration depend on a convergence criterion.

Invariant I6 (see docs/protocol-invariants.md) lives here: nothing in this module peak-normalises the
mix. Every level is set explicitly, before mixing, from the requested axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.audibility import (
    audibility_margin_db,
    masking_threshold_bark,
    periodogram_bark,
    threshold_in_quiet_db,
)

__all__ = [
    "MASKER_KINDS",
    "normalise_event",
    "MaskerSpec",
    "shaped_noise",
    "event_matched_masker",
    "target_rms",
    "solve_event_gain",  # legacy, see its docstring
    "solve_masker_attenuation_db",
    "solve_masker_for_energy_snr_db",
]

MASKER_KINDS = (
    "white",
    "pink",
    "low_pass",
    "high_pass",
    "event_matched",
    "threshold_shaped",
    "threshold_inverse",
)

# Relative floor for the event-matched masker's envelope: bins the event does not
# occupy stay at this fraction of its peak power, so the masker remains noise-like
# instead of collapsing onto the event's spectral lines.
_MATCH_FLOOR = 1e-3

#: Dynamic range of the threshold-shaped maskers, in dB of amplitude.
#:
#: Two readings of "hearing-threshold-shaped" are implemented, because they are not
#: equivalent and the choice decides whether the audibility axis can be separated from
#: the energy axis at all (measured 2026-09-17 on the 36-class corpus, airport
#: background, at matched energy SNR):
#:
#: ``threshold_shaped``
#:     PSD proportional to the threshold in quiet, i.e. amplitude ``10 ** (+LTQ/20)``.
#:     The masker is loud exactly where hearing is *insensitive* (sub-300 Hz carries
#:     99.9% of its energy), so equal energy buys the least possible masking. Measured
#:     separation from uniform white noise: **~30 dB of audibility at matched energy** on
#:     a single example item (a bicycle bell: white leaves the event at -2.5 dB, this
#:     masker at +28.7 dB for the same energy SNR), and **median ~12 dB** across the
#:     36-class axis-duel corpus (2026-09-18, ``calibration_report.json``). This is the
#:     literal reading and the only one that makes the two axes genuinely different knobs.
#: ``threshold_inverse``
#:     PSD proportional to ``1 / LTQ``, i.e. amplitude ``10 ** (-LTQ/20)``: the
#:     audiological "threshold-shaped noise" that sits at a constant distance above the
#:     threshold in every band. Measured separation from white: **median 2.96 dB**
#:     (range -7.7 to +4.2 dB across items), i.e. near-collinear with white noise at
#:     16 kHz sampling, where 50% of white noise's energy is already inside 1-5 kHz.
#:     Substituting it for ``threshold_shaped`` collapses the duel's separation to ~3 dB.
#:
#: The cap exists because the Terhardt curve spans >100 dB between 20 Hz and 8 kHz; at
#: 16 kHz sampling the curve's own range inside 0-8 kHz is ~28 dB, so a 40 dB cap is
#: non-binding for ``threshold_inverse`` and only trims the sub-30 Hz tail for
#: ``threshold_shaped``.
_THRESHOLD_SHAPE_RANGE_DB = 40.0


@dataclass(frozen=True)
class MaskerSpec:
    """Everything needed to regenerate a masker deterministically."""

    kind: str
    rms: float
    seed: int
    cutoff_hz: float | None = None
    order: int = 4

    def __post_init__(self) -> None:
        if self.kind not in MASKER_KINDS:
            raise ValueError(f"unknown masker kind {self.kind!r}")
        if self.kind in {"low_pass", "high_pass"} and self.cutoff_hz is None:
            raise ValueError(f"{self.kind} requires cutoff_hz")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def normalise_event(event: np.ndarray, target_rms: float = 0.15) -> np.ndarray:
    """Scale an event to a fixed RMS.

    Required before any audibility computation, because the audibility scale is
    anchored by the threshold in quiet: a harvested FSD50K event can sit arbitrarily
    far below it, and measured on the first attempt the whole cell was fixed by that
    offset (-10.31 dB at *any* masker level) rather than by the masker. Normalising
    the event puts every item on a comparable footing and keeps the energy axis
    meaningful, since the event level is then identical across maskers by
    construction.
    """

    x = np.asarray(event, dtype=np.float64).reshape(-1)
    current = float(np.sqrt(np.mean(x ** 2)))
    if current <= 0.0:
        raise ValueError("cannot normalise a silent event")
    return (x * (float(target_rms) / current)).astype(np.float32)


def target_rms(reference: np.ndarray, snr_db: float) -> float:
    """Event RMS that sits ``snr_db`` above ``reference`` (the masker) in energy."""

    return float(_rms(reference)) * float(10.0 ** (snr_db / 20.0))


def shaped_noise(
    seconds: float,
    sample_rate: int,
    spec: MaskerSpec,
    *,
    event: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic noise with the requested spectrum, scaled to ``spec.rms``.

    Shaping is done in the frequency domain with a random phase, then inverse
    transformed and scaled. Scaling *after* shaping is what keeps the masker RMS
    exact, which the equal-energy contrast depends on.
    """

    rng = np.random.default_rng(spec.seed)
    n = int(round(seconds * sample_rate))
    if n <= 0:
        raise ValueError("seconds must be positive")
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    shape = _shaping_curve(spec, freqs, event, sample_rate)
    shaped = spectrum * shape
    wave = np.fft.irfft(shaped, n=n).astype(np.float64)
    current = _rms(wave)
    if current == 0.0:
        raise ValueError("shaping produced silence")
    return (wave * (spec.rms / current)).astype(np.float32)


def _shaping_curve(
    spec: MaskerSpec,
    freqs: np.ndarray,
    event: np.ndarray | None,
    sample_rate: int,
) -> np.ndarray:
    if spec.kind == "white":
        return np.ones_like(freqs)
    if spec.kind == "pink":
        # 1/f amplitude: power falls at 3 dB/octave
        return 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    if spec.kind == "low_pass":
        return 1.0 / (1.0 + (freqs / float(spec.cutoff_hz)) ** spec.order)
    if spec.kind == "high_pass":
        return 1.0 / (1.0 + (float(spec.cutoff_hz) / np.maximum(freqs, 1e-6)) ** spec.order)
    if spec.kind == "event_matched":
        if event is None:
            raise ValueError("event_matched masker requires the event waveform")
        return _event_spectrum_envelope(event, freqs, sample_rate)
    if spec.kind in {"threshold_shaped", "threshold_inverse"}:
        return _threshold_shaping_curve(freqs, inverse=spec.kind == "threshold_inverse")
    raise ValueError(f"unhandled kind {spec.kind!r}")


def _threshold_shaping_curve(freqs: np.ndarray, *, inverse: bool) -> np.ndarray:
    """Amplitude envelope of a hearing-threshold-shaped masker.

    ``inverse=False`` gives the literal shape (PSD follows the threshold in quiet, so
    the masker is loud where hearing is insensitive); ``inverse=True`` gives the
    equal-masking shape (PSD follows the inverse threshold). See
    :data:`_THRESHOLD_SHAPE_RANGE_DB` for what each one measures against white noise.
    """

    amp_db = -threshold_in_quiet_db(freqs) if inverse else threshold_in_quiet_db(freqs)
    amp_db = np.clip(amp_db, float(amp_db.max()) - _THRESHOLD_SHAPE_RANGE_DB, float(amp_db.max()))
    return 10.0 ** (amp_db / 20.0)


def _event_spectrum_envelope(
    event: np.ndarray, freqs: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Amplitude envelope of the event's own spectrum, on ``freqs``.

    Two earlier implementations failed and both failures are worth recording,
    because "match the masker to the event" looks trivial and is not:

    1. interpolating **magnitudes** across a grid mismatch collapsed the envelope
       (a windowed tone's main lobe is ~1e-4 while its skirt is ~1e-24, so linear
       interpolation between grids destroyed the peak);
    2. smoothing the **log power** is worse than useless here: a windowed tone's
       skirt decays steeply, so a wide log-domain average is dominated by the
       low-frequency tail and the -24 dB-per-decade slope survives into the
       envelope; the masker then sat *away* from the event and had less energy in
       the event's band than white noise.

    The fix is to stop smoothing and stop peaking: normalise the event's power
    spectrum to unit peak, then add a relative floor so the masker cannot be
    arbitrarily quiet anywhere. That keeps the event's bands dominant without
    introducing a comb and without a frequency-dependent bias.
    """

    x = np.asarray(event, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 8:
        raise ValueError("event too short to estimate a spectrum")
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window, n)) / float(n)
    source_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    power = np.interp(freqs, source_freqs, spec ** 2, left=spec[0] ** 2, right=spec[-1] ** 2)
    peak = float(power.max())
    if peak <= 0.0:
        return np.ones_like(power)
    envelope = power / peak
    return envelope + _MATCH_FLOOR


def event_matched_masker(
    event: np.ndarray,
    sample_rate: int,
    *,
    rms: float,
    seed: int,
) -> np.ndarray:
    """Noise whose spectrum follows the event's: a strong masker for that event."""

    seconds = len(np.asarray(event).reshape(-1)) / float(sample_rate)
    spec = MaskerSpec(kind="event_matched", rms=rms, seed=seed)
    return shaped_noise(seconds, sample_rate, spec, event=event)


def solve_event_gain(
    event: np.ndarray,
    masker: np.ndarray,
    sample_rate: int,
    *,
    target_margin_db: float,
    tolerance_db: float = 0.05,
) -> float:
    """LEGACY event-gain parameterisation. Use :func:`solve_masker_attenuation_db`.

    Kept because the decoupling probe measures with it and its readings are cited in
    the project record. Do not use it to build items: measured, it demanded a 67x
    event gain against an ``event_matched`` masker, which pushed the scene to a peak of
    25.5 and made PCM_16 writing clip the event away entirely.

    Gain that puts the event exactly ``target_margin_db`` above the threshold.

    Both the event band power and the masker's threshold are fixed once the masker
    is fixed, so the margin is ``margin_0 + 20*log10(gain)`` and the solve is exact.
    The returned gain is verified against a fresh measurement and raises if it does
    not land within ``tolerance_db`` — a wrong gain would silently corrupt the
    equal-audibility contrast.
    """

    baseline = audibility_margin_db(event, masker, sample_rate)
    gain = float(10.0 ** ((float(target_margin_db) - baseline) / 20.0))
    achieved = audibility_margin_db((event * gain).astype(np.float32), masker, sample_rate)
    if abs(achieved - float(target_margin_db)) > tolerance_db:
        raise AssertionError(
            f"gain solver missed the target: wanted {target_margin_db:+.2f} dB, "
            f"got {achieved:+.2f} dB (baseline {baseline:+.2f}, gain {gain:.4f})"
        )
    return gain


def solve_masker_attenuation_db(
    event: np.ndarray,
    masker_template: np.ndarray,
    sample_rate: int,
    *,
    reference_rms: float,
    target_margin_db: float,
    background: np.ndarray | None = None,
    min_attenuation_db: float = -120.0,
    max_attenuation_db: float = 45.0,
    max_iterations: int = 60,
    tolerance_db: float = 0.05,
) -> tuple[float, float]:
    """Masker attenuation (dB) that puts the event, at unity gain, on the target margin.

    Returns ``(attenuation_db, masker_rms)``. The audibility axis belongs on the
    **masker**, not on the event gain, for two measured reasons:

    * solving for the event gain returned **67x** with an ``event_matched`` masker, so
      the scene peaked at 25.5 and PCM_16 writing clipped it to 0.99 — the rendered
      audio no longer contained the event while the reported audibility still
      described it (invariant I7 caught the consequence, the parameterisation was the
      cause);
    * with the event at a fixed level, the required masker level spans a wide range
      across maskers and items, and a bounded *linear* RMS search is not monotone in
      the way the physics is: measured, a full-scale masker still left the margin at
      +38.7 dB, i.e. the target needed a masker quieter than the lower bound. Searching
      in dB is monotone across the whole useful range.

    The template is used only for its spectrum; its own level is irrelevant.

    ``background`` is the background over the query span (already sliced to the event's
    length). When given it is added to the scaled masker before the margin is measured,
    so the solve lands on the *realised* interferer rather than on the masker alone.
    Without it the solve can be a few tenths of a dB optimistic on a quiet item and
    several dB off on a loud one — which is exactly the discrepancy that makes a
    "calibrated" axis not calibrated.
    """

    template = np.asarray(masker_template, dtype=np.float64)
    template_rms = float(np.sqrt(np.mean(template ** 2)))
    if template_rms <= 0.0:
        raise ValueError("masker template is silent")
    unit = (template / template_rms).astype(np.float32)
    base = None if background is None else np.asarray(background, dtype=np.float32)

    def margin_at(attenuation_db: float) -> float:
        rms = float(reference_rms) * 10.0 ** (attenuation_db / 20.0)
        interferer = (unit * rms).astype(np.float32)
        if base is not None:
            interferer = (base + interferer).astype(np.float32)
        return audibility_margin_db(event, interferer, sample_rate)

    # The margin *rises* as the masker is attenuated (less masker -> more audible).
    # The direction is measured from the two bounds rather than assumed: an earlier
    # version assumed the opposite and bisected the wrong way, converging to a point
    # 6.5 dB off the target.
    lo, hi = float(min_attenuation_db), float(max_attenuation_db)
    margin_lo, margin_hi = margin_at(lo), margin_at(hi)
    if margin_lo < target_margin_db:
        raise AssertionError(
            f"even a {lo:.0f} dB-attenuated masker leaves the margin at {margin_lo:+.2f} dB, "
            f"below the {target_margin_db:+.2f} dB target"
        )
    if margin_hi > target_margin_db:
        raise AssertionError(
            f"even a masker at {hi:+.0f} dB (reference RMS {reference_rms:g}) leaves the "
            f"margin at {margin_hi:+.2f} dB, above the {target_margin_db:+.2f} dB target"
        )
    best, best_err = None, float("inf")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        margin = margin_at(mid)
        err = abs(margin - target_margin_db)
        if err < best_err:
            best, best_err = mid, err
        if err <= tolerance_db:
            break
        # margin falls with attenuation: too audible -> increase attenuation
        if margin > target_margin_db:
            lo = mid
        else:
            hi = mid
    assert best is not None
    achieved = margin_at(best)
    if abs(achieved - target_margin_db) > tolerance_db:
        raise AssertionError(
            f"masker attenuation solver missed the target: wanted {target_margin_db:+.2f} dB, "
            f"got {achieved:+.2f} dB at {best:+.2f} dB attenuation"
        )
    return float(best), float(reference_rms) * 10.0 ** (best / 20.0)


def solve_masker_for_energy_snr_db(
    event: np.ndarray,
    masker_template: np.ndarray,
    sample_rate: int,
    *,
    reference_rms: float,
    target_snr_db: float,
    background: np.ndarray | None = None,
    min_attenuation_db: float = -120.0,
    max_attenuation_db: float = 60.0,
    max_iterations: int = 80,
    tolerance_db: float = 0.05,
) -> tuple[float, float]:
    """Masker attenuation (dB) that puts the event at ``target_snr_db`` in energy.

    Energy SNR is the definition the energy axis uses::

        SNR = 20 log10( rms(event) / rms(background + masker) )

    measured over the query span. ``background`` should be that span, sliced to the
    event's length; including it is what makes the realised SNR land on the target
    rather than on the masker-alone approximation (the corpus backgrounds are quiet,
    ~0.002 RMS, but they set the ceiling: an event at 0.15 RMS cannot be asked for more
    than ~+38 dB).

    Bisection in dB for the same reason as the audibility solver — the RMS of the sum is
    not monotone in the masker's *linear* amplitude across the whole useful range, but it
    is monotone in dB once the masker is the dominant term. Returns
    ``(attenuation_db, masker_rms)``.
    """

    template = np.asarray(masker_template, dtype=np.float64)
    template_rms = float(np.sqrt(np.mean(template ** 2)))
    if template_rms <= 0.0:
        raise ValueError("masker template is silent")
    unit = (template / template_rms).astype(np.float32)
    base = None if background is None else np.asarray(background, dtype=np.float32)
    event_rms = float(np.sqrt(np.mean(np.asarray(event, dtype=np.float64) ** 2)))
    if event_rms <= 0.0:
        raise ValueError("cannot calibrate an energy SNR for a silent event")

    def snr_at(attenuation_db: float) -> float:
        rms = float(reference_rms) * 10.0 ** (attenuation_db / 20.0)
        interferer = (unit * rms).astype(np.float32)
        if base is not None:
            interferer = (base + interferer).astype(np.float32)
        ref_rms = float(np.sqrt(np.mean(interferer.astype(np.float64) ** 2)))
        return float(20.0 * np.log10(max(event_rms, 1e-12) / max(ref_rms, 1e-12)))

    # SNR falls as the masker is attenuated less (i.e. as attenuation_db rises): a
    # louder masker means a smaller ratio. Direction measured from the bounds, not
    # assumed — the audibility solver shipped with the opposite assumption once.
    lo, hi = float(min_attenuation_db), float(max_attenuation_db)
    snr_lo, snr_hi = snr_at(lo), snr_at(hi)
    if snr_lo < target_snr_db:
        raise AssertionError(
            f"even a {lo:.0f} dB-attenuated masker leaves the energy SNR at {snr_lo:+.2f} dB, "
            f"below the {target_snr_db:+.2f} dB target"
        )
    if snr_hi > target_snr_db:
        raise AssertionError(
            f"even a masker at {hi:+.0f} dB (reference RMS {reference_rms:g}) leaves the "
            f"energy SNR at {snr_hi:+.2f} dB, above the {target_snr_db:+.2f} dB target "
            f"(the background alone sets this ceiling)"
        )
    best, best_err = None, float("inf")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        snr = snr_at(mid)
        err = abs(snr - target_snr_db)
        if err < best_err:
            best, best_err = mid, err
        if err <= tolerance_db:
            break
        # too much SNR -> the masker is too quiet -> less attenuation
        if snr > target_snr_db:
            lo = mid
        else:
            hi = mid
    assert best is not None
    achieved = snr_at(best)
    if abs(achieved - target_snr_db) > tolerance_db:
        raise AssertionError(
            f"energy-SNR solver missed the target: wanted {target_snr_db:+.2f} dB, "
            f"got {achieved:+.2f} dB at {best:+.2f} dB attenuation"
        )
    return float(best), float(reference_rms) * 10.0 ** (best / 20.0)
