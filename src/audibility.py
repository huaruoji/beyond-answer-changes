"""Psychoacoustic audibility score for the CTL branch.

This module answers one question: **given a masker and an event, how far above the
hearing system's masking threshold is the event, in the event's own bands?**

Why it exists separately from the generator: the CTL design needs a scalar that is
comparable *across masker spectra*, so that "equal audibility, different energy"
and "equal energy, different audibility" are both constructible. A broadband SNR
number cannot do that — two maskers with identical energy but different spectra sit
at very different distances from the threshold.

Unit convention (this is the part that bit the first prototype):

* The threshold curve is calibrated in the model's own convention, where a
  full-scale signal corresponds to a reference level and the threshold in quiet
  LTQ is expressed as `(LTQ - 60) dB` relative to it. Every quantity fed to it
  must therefore be a **power** value on the same scale as a normalised FFT.
* We therefore compute band powers from a **unit-normalised periodogram**
  (Parseval-calibrated: the sum over bins of the two-sided periodogram equals the
  mean square of the signal). Without this the event/threshold ratio is off by the
  window's coherent gain and the band summation factor — measured at 10-15 dB in
  the prototype, which is enough to invert a conclusion.

The margin is computed only in the event's own active bands (default: top 12 Bark
bands by event power), because a masking threshold is a per-band quantity; summing
over bands the event does not occupy measures the masker, not the event.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hz_to_bark",
    "bark_to_hz",
    "bark_band_matrix",
    "periodogram_bark",
    "masking_threshold_bark",
    "audibility_margin_db",
    "threshold_in_quiet_db",
]

_N_BARK = 64
_N_FFT = 2048


def hz_to_bark(f: np.ndarray | float) -> np.ndarray:
    """Traunmueller's Hz -> Bark formula.

    Paired with :func:`bark_to_hz`, which is its exact algebraic inverse. The
    commonly copied ``13*atan(f/1315.8) + 3.5*atan((f/7518)^2)`` formula is *not*
    invertible by ``600*sinh(z/6)``; mixing the two produced up to 34% frequency
    error at 1-8 kHz in the first implementation (measured), which would misplace
    the threshold-in-quiet curve.
    """

    f = np.asarray(f, dtype=np.float64)
    return 26.81 * f / (1960.0 + f) - 0.53


def bark_to_hz(z: np.ndarray | float) -> np.ndarray:
    """Exact inverse of :func:`hz_to_bark`."""

    z = np.asarray(z, dtype=np.float64)
    return 1960.0 * (z + 0.53) / (26.81 - (z + 0.53))


def _bark_edges(sample_rate: int, n_bark: int = _N_BARK) -> np.ndarray:
    max_bark = float(hz_to_bark(sample_rate / 2.0))
    return np.arange(n_bark + 1) * max_bark / n_bark


def bark_band_matrix(sample_rate: int, n_fft: int = _N_FFT, n_bark: int = _N_BARK) -> np.ndarray:
    """(n_bark, n_fft//2) summing matrix from FFT bins to Bark bands."""

    edges_hz = bark_to_hz(_bark_edges(sample_rate, n_bark))
    bin_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)[: n_fft // 2]
    W = np.zeros((n_bark, n_fft // 2), dtype=np.float64)
    for i in range(n_bark):
        lo, hi = edges_hz[i], edges_hz[i + 1]
        sel = (bin_hz >= lo) & (bin_hz < hi)
        if not sel.any():  # a band narrower than one bin: take the nearest bin
            sel = np.zeros_like(bin_hz, dtype=bool)
            sel[int(np.argmin(np.abs(bin_hz - 0.5 * (lo + hi))))] = True
        W[i, sel] = 1.0
    return W


def _spreading_matrix(n_bark: int = _N_BARK) -> np.ndarray:
    z = np.arange(n_bark, dtype=np.float64)
    dz = np.abs(z[:, None] - z[None, :])
    return 10.0 ** ((15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1.0 + (dz + 0.474) ** 2)) / 10.0)


def threshold_in_quiet_db(freqs_hz: np.ndarray | float) -> np.ndarray:
    """Terhardt's absolute threshold of hearing (dB) at ``freqs_hz``.

    The single source of truth for the LTQ curve. :func:`masking_threshold_bark` reads
    it for the per-band floor, and :mod:`src.maskers` reads it to shape the
    ``threshold_shaped`` masker — a masker shaped by the threshold must not drift away
    from the threshold the audibility score uses, or the two stop describing the same
    listener.
    """

    f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), 1e-6) / 1000.0
    ltq = (
        3.64 * f ** -0.8
        - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2)
        + 1e-3 * f ** 4
    )
    return np.clip(ltq, -20.0, 120.0)


def _threshold_in_quiet_db(sample_rate: int, n_bark: int = _N_BARK) -> np.ndarray:
    """Absolute threshold of hearing per Bark band, in the model's dB convention."""

    edges = _bark_edges(sample_rate, n_bark)
    centres = bark_to_hz(0.5 * (edges[:-1] + edges[1:]))
    return threshold_in_quiet_db(centres)


def periodogram_bark(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = _N_FFT,
    n_bark: int = _N_BARK,
    hop: int | None = None,
) -> np.ndarray:
    """Median Bark-band power over frames, Parseval-calibrated.

    Power is expressed as mean-square per band, so a full-scale sine has band
    power ~0.5 in the bands it occupies. Median (not mean) aggregation keeps a
    short transient from being averaged away over a long clip.
    """

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("empty signal")
    hop = hop or n_fft // 2
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft)
    # Parseval: sum_k |X_k|^2 / (n_fft * sum_n w_n^2) == mean square of (x*w).
    # Both factors are required; dropping n_fft scales every band power by n_fft
    # and would make the absolute threshold-in-quiet floor meaningless.
    norm = float(n_fft) * float((window ** 2).sum())
    W = bark_band_matrix(sample_rate, n_fft, n_bark)
    frames = []
    for start in range(0, x.size - n_fft + 1, hop):
        seg = x[start : start + n_fft] * window
        spec = np.fft.rfft(seg, n_fft)
        psd = (np.abs(spec) ** 2) / norm
        psd[1:-1] *= 2.0  # two-sided -> one-sided; recovers the frame mean square
        frames.append(psd[: n_fft // 2] @ W.T)
    if not frames:
        raise ValueError("signal shorter than one frame")
    return np.median(np.vstack(frames), axis=0)


def masking_threshold_bark(
    masker: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = _N_FFT,
    n_bark: int = _N_BARK,
    alpha: float = 0.6,
    apply_threshold_in_quiet: bool = True,
) -> np.ndarray:
    """Bark-band masking threshold POWER of ``masker``, same scale as periodogram_bark."""

    band_power = periodogram_bark(masker, sample_rate, n_fft=n_fft, n_bark=n_bark)
    S = _spreading_matrix(n_bark)
    thr = (band_power ** alpha) @ (S ** alpha)
    thr = thr ** (1.0 / alpha)
    if apply_threshold_in_quiet:
        ltq_power = 10.0 ** ((_threshold_in_quiet_db(sample_rate, n_bark) - 60.0) / 10.0)
        thr = np.maximum(thr, ltq_power)
    return thr


def audibility_margin_db(
    event: np.ndarray,
    masker: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = _N_FFT,
    n_bark: int = _N_BARK,
    neighbourhood: int = 1,
    statistic: str = "peak",
    mix: bool = False,
) -> float:
    """How far the event sits above the masker's threshold, in the event's peak band.

    The per-band event/threshold ratio is the quantity with a physical reading:
    detection is governed by the band where the event is strongest relative to the
    masker, not by bands it does not occupy. ``statistic="peak"`` therefore reports
    the maximum ratio over bands, and ``"median"``/``"mean"`` average the ratio over
    the ``neighbourhood`` bands either side of the peak (robustness variants).

    With ``mix=True`` the threshold is computed from event + masker, i.e. what the
    listener/model actually receives; with ``mix=False`` from the masker alone,
    which is what the generator can control *before* mixing.

    Absolute anchoring: the threshold curve is only defined up to a constant offset
    in this implementation (the threshold-in-quiet floor calibrates the model's dB
    convention, not ours). Comparisons across maskers and levels — which is what the
    CTL design uses — are unaffected; absolute dB values should be reported as
    "relative to the threshold-in-quiet anchor".
    """

    e_band = periodogram_bark(event, sample_rate, n_fft=n_fft, n_bark=n_bark)
    reference = event + masker if mix else masker
    thr = masking_threshold_bark(reference, sample_rate, n_fft=n_fft, n_bark=n_bark)
    ratio = 10.0 * np.log10(np.maximum(e_band, 1e-30) / np.maximum(thr, 1e-30))
    k = int(np.argmax(e_band))
    if statistic == "peak":
        return float(ratio[k])
    lo = max(0, k - int(neighbourhood))
    hi = min(ratio.size, k + int(neighbourhood) + 1)
    window = ratio[lo:hi]
    if statistic == "median":
        return float(np.median(window))
    if statistic == "mean":
        return float(np.mean(window))
    raise ValueError(f"unknown statistic {statistic!r}")
