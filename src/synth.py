"""Synthetic soundscape rendering and causal editing.

This module owns two things that the rest of the project must not bypass:

1. **The scaper environment contract.** scaper 1.6.5 calls ``np.Inf``, removed in
   NumPy 2.0, and needs the conda-provided sox shared library on the loader path.
   The compatibility shim runs at import time here, so nothing else may
   ``import scaper`` directly.

2. **The ADR-011 invariants.** Every edit is expressed as
   :func:`crossfade_replace`, which only accepts an explicit sample span. There is
   deliberately no code path that substitutes a whole donor file: an edit is
   always "this interval, replaced by these samples". Control windows are chosen
   by :func:`find_free_window`, which refuses to return a window overlapping any
   placed event, so a matched control cannot accidentally land on the evidence.

See ``docs/synthesis_design.md`` for the experiment this supports.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

# --- scaper compatibility shim (must precede the import) ---------------------
# scaper 1.6.5 predates NumPy 2.0 and uses the removed alias np.Inf in four
# places in scaper/core.py. Restoring the alias is a compatibility shim, not a
# behavioural change. Verified to be the only NumPy-2 incompatibility across
# scaper, jams and soxbindings.
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

import scaper  # noqa: E402  (import order is load-bearing)

DEFAULT_DURATION_S = 10.0
DEFAULT_FADE_MS = 10.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TARGET_PEAK = 0.90
PLACEMENT_GRID_S = 0.01
MIN_EVENT_GAP_S = 0.30

#: Condition names, in the order they appear in the design document.
CONDITIONS = (
    "original",
    "target_replaced_diff",
    "target_replaced_same",
    "nuisance_removed",
    "background_removed",
    "sham",
    "silence",
)

#: Conditions that only the natural-audio route uses. Natural recordings do not
#: come with a class ontology, so "replace the evidence with another instance of
#: the same class" is not constructible there: measured on the TEMPO tacos split,
#: 2000 clips carry 3848 distinct free-text event descriptions and only 92 occur
#: in more than one clip. The matched control is therefore re-anchored on the
#: *same donor waveform written by the same operation into a different span*:
#: overwriting the nuisance span with a third class flips nothing, while
#: overwriting the query span with the option's class flips the answer.
EXTRA_CONDITIONS = ("nuisance_replaced",)

#: Which conditions rewrite a span, for the manifest audit.
SPAN_EDIT_CONDITIONS = (
    "target_replaced_diff",
    "target_replaced_same",
    "nuisance_removed",
    "background_removed",
    "sham",
)


class InstrumentError(RuntimeError):
    """Raised when a construction invariant would be violated."""


@dataclasses.dataclass(frozen=True)
class PlacedEvent:
    """One event placed in a soundscape, with the span it actually occupies."""

    role: str  # "query" or "nuisance"
    label: str
    source_path: str
    start_s: float
    end_s: float
    snr_db: float
    source_span_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def samples(self) -> tuple[float, float]:
        return self.start_s, self.end_s


@dataclasses.dataclass
class Soundscape:
    """A rendered soundscape plus the exact events inside it."""

    audio_path: Path
    sample_rate: int
    duration_s: float
    background_path: str
    events: list[PlacedEvent]

    @property
    def query(self) -> PlacedEvent:
        for event in self.events:
            if event.role == "query":
                return event
        raise InstrumentError("soundscape has no query event")

    @property
    def nuisance(self) -> PlacedEvent | None:
        for event in self.events:
            if event.role == "nuisance":
                return event
        return None


# --- sample-span editing -----------------------------------------------------


def crossfade_replace(
    waveform: np.ndarray,
    start: int,
    end: int,
    replacement: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Overwrite ``waveform[start:end]`` with ``replacement``, seamlessly.

    Both edges use the same rising raised-cosine weight ``w`` (0 at the outer
    edge of a fade, 1 at the inner edge). The head therefore runs
    ``original -> replacement`` and the tail runs ``replacement -> original``,
    which keeps all four boundaries continuous.

    The previous implementation, now in :mod:`withdrawn.src.interventions`, applied the weight
    reversed at the tail, so the tail ran ``original -> replacement`` and both
    the entry and the exit of the tail were full-scale discontinuities. That bug
    is why every causal number produced before 2026-09-12 was withdrawn; the
    regression is pinned by ``tests/test_synth.py::test_tail_crossfade_is_continuous``.
    """

    waveform = np.asarray(waveform, dtype=np.float32)
    replacement = np.asarray(replacement, dtype=np.float32)
    if replacement.shape != waveform[start:end].shape:
        raise InstrumentError(
            f"replacement shape {replacement.shape} does not match span "
            f"{waveform[start:end].shape}"
        )
    if not (0 <= start < end <= len(waveform)):
        raise InstrumentError(f"span [{start}, {end}) outside [0, {len(waveform)})")

    fade = int(min(fade_samples, (end - start) // 2))
    out = waveform.copy()
    block = out[start:end].copy()
    filled = replacement.copy()
    if fade > 0:
        phase = np.linspace(0.0, np.pi, fade, endpoint=True, dtype=np.float32)
        weight = 0.5 - 0.5 * np.cos(phase)  # rising, 0 -> 1
        filled[:fade] = (1.0 - weight) * block[:fade] + weight * filled[:fade]
        filled[-fade:] = (1.0 - weight) * filled[-fade:] + weight * block[-fade:]
    out[start:end] = filled
    return out


def boundary_jump(waveform: np.ndarray, start: int, end: int) -> float:
    """Largest first difference across the four edges of a rewritten span.

    Used by the audit and by the I4 regression test. A correct crossfade leaves
    this no larger than the surrounding signal's own first difference.
    """

    edges = [(start - 1, start), (start, start + 1), (end - 1, end), (end, end + 1)]
    worst = 0.0
    for left, right in edges:
        if 0 <= left < len(waveform) and 0 <= right < len(waveform):
            worst = max(worst, abs(float(waveform[right]) - float(waveform[left])))
    return worst


def event_mask(
    events: Iterable[PlacedEvent], n_samples: int, sample_rate: int, guard_s: float
) -> np.ndarray:
    """Boolean mask of samples that lie inside an event or its guard band."""

    mask = np.zeros(n_samples, dtype=bool)
    guard = int(round(guard_s * sample_rate))
    for event in events:
        lo = max(0, int(round(event.start_s * sample_rate)) - guard)
        hi = min(n_samples, int(round(event.end_s * sample_rate)) + guard)
        mask[lo:hi] = True
    return mask


def find_free_window(
    events: Sequence[PlacedEvent],
    n_samples: int,
    sample_rate: int,
    length_samples: int,
    rng: random.Random,
    guard_s: float = MIN_EVENT_GAP_S,
    exclude: tuple[int, int] | None = None,
    attempts: int = 200,
) -> tuple[int, int]:
    """Return a sample span of ``length_samples`` containing no event.

    This is what makes ``background_removed`` a matched control rather than a
    guess: the window provably holds no placed event, so the edit differs from
    the target edit only in *what* was there, not in *whether* something was.

    Raises:
        InstrumentError: if the clip has no event-free window that long. Callers
            treat that as a rejected item, never as a fallback.
    """

    if length_samples <= 0 or length_samples > n_samples:
        raise InstrumentError("requested window length does not fit the clip")
    mask = event_mask(events, n_samples, sample_rate, guard_s)
    if exclude is not None:
        mask[max(0, exclude[0]) : min(n_samples, exclude[1])] = True
    starts = [
        index
        for index in range(0, n_samples - length_samples + 1, max(1, sample_rate // 100))
        if not mask[index : index + length_samples].any()
    ]
    if not starts:
        raise InstrumentError(
            "no event-free window of the requested length; item must be rejected"
        )
    chosen = rng.choice(starts)
    return chosen, chosen + length_samples


def _background_fill(
    waveform: np.ndarray,
    span: tuple[int, int],
    events: Sequence[PlacedEvent],
    sample_rate: int,
    rng: random.Random,
    fade_samples: int,
) -> np.ndarray:
    """Replace a span with background copied from an event-free window."""

    start, end = span
    donor_start, donor_end = find_free_window(
        events,
        len(waveform),
        sample_rate,
        end - start,
        rng,
        exclude=span,
    )
    return crossfade_replace(
        waveform, start, end, waveform[donor_start:donor_end].copy(), fade_samples
    )


# --- conditions --------------------------------------------------------------


def resolve_condition_span(
    scape: Soundscape,
    condition: str,
    n_samples: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    """The sample span a condition rewrites, or None if it rewrites none/all.

    Exposing this separately is what makes invariant I2 auditable from the
    manifest: the generator records the returned span, and the audit checks that
    a control edit does not overlap the query span. Recovering the span by
    differencing waveforms, as the pre-2026-09-12 audit had to, is exactly how
    the overlapping controls went unnoticed.
    """

    if condition not in CONDITIONS + EXTRA_CONDITIONS:
        raise InstrumentError(f"unknown condition {condition!r}")
    sample_rate = scape.sample_rate
    query = scape.query
    query_span = (
        int(round(query.start_s * sample_rate)),
        int(round(query.end_s * sample_rate)),
    )
    if condition in {"original", "silence"}:
        return None
    if condition in {
        "sham",
        "target_replaced_diff",
        "target_replaced_same",
        "target_removed",
    }:
        return query_span
    if condition in {"nuisance_removed", "nuisance_replaced"}:
        nuisance = scape.nuisance
        if nuisance is None:
            raise InstrumentError("soundscape has no nuisance event")
        return (
            int(round(nuisance.start_s * sample_rate)),
            int(round(nuisance.end_s * sample_rate)),
        )
    if condition == "background_removed":
        return find_free_window(
            scape.events, n_samples, sample_rate, query_span[1] - query_span[0], rng
        )
    raise InstrumentError(f"unhandled condition {condition!r}")


def apply_condition(
    waveform: np.ndarray,
    scape: Soundscape,
    condition: str,
    *,
    donor_waveform: np.ndarray | None = None,
    donor_span_samples: int | None = None,
    donor_level_reference: np.ndarray | None = None,
    span: tuple[int, int] | None = None,
    rng: random.Random,
    fade_ms: float = DEFAULT_FADE_MS,
    preserve_background: bool = True,
) -> np.ndarray:
    """Render one condition of the seven-condition grid.

    ``preserve_background`` controls whether a replacement is laid *into* the
    clip's background or *in place of* it. The mainline grid sets it (the default), so
    the substituted sound keeps the noise floor a listener expects; without it the
    span went digitally quiet, which a listener heard on 2026-09-21 and
    measurement confirmed (median noise-floor drop ~15 dB). The CTL branch passes
    ``False``: its calibration was derived against donor-only replacements and is
    archived and closed, so changing it would invalidate a frozen artifact for no
    gain. See :func:`mix_into_background`.

    ``target_replaced_diff`` and ``target_replaced_same`` need a donor: the
    caller passes the donor's own samples together with the sample count of the
    donor's annotated span. Requiring the span explicitly is invariant I1 — a
    whole donor file can never be passed as if it were an event.

    ``donor_level_reference`` is the signal the donor's level is matched to. When it is
    omitted the donor is levelled to the span it replaces, which is the behaviour every
    mainline build and test was written against and is therefore kept as the default. CTL
    passes the **event being replaced** instead: matching to the span means matching to
    the masker, because at the hard end of a ladder the span is mostly masker. Measured
    on the axis-duel build (CTL-Plan §3.4i#3, §3.4j), the span reference put replacements
    up to **+27.5 dB above the event they replaced**, with the replacement's own
    audibility flat to ±3 dB across a 30 dB difficulty ladder — so the ladder only bit on
    the ``original`` arm and the manifest's audibility described a signal that was not in
    the ``diff`` audio. The reference is the *array*, not a span, so the caller cannot
    accidentally pass the masked mix it is trying to avoid.

    ``span`` overrides the rewritten interval. Callers that record the span in a
    manifest should resolve it with :func:`resolve_condition_span` first and pass
    it here, so the manifest and the rendered audio cannot disagree.
    """

    if condition not in CONDITIONS + EXTRA_CONDITIONS:
        raise InstrumentError(f"unknown condition {condition!r}")
    sample_rate = scape.sample_rate
    fade_samples = int(round(fade_ms * sample_rate / 1000.0))
    query = scape.query
    start = int(round(query.start_s * sample_rate))
    end = int(round(query.end_s * sample_rate))
    out = np.asarray(waveform, dtype=np.float32)

    if condition == "original":
        return out.copy()

    if condition == "silence":
        return np.zeros_like(out)

    if condition == "sham":
        return crossfade_replace(out, start, end, out[start:end].copy(), fade_samples)

    if condition == "target_replaced_diff" or condition == "target_replaced_same":
        if donor_waveform is None or donor_span_samples is None:
            raise InstrumentError(f"{condition} requires a donor and its span")
        fitted = _fit_donor(
            donor_waveform,
            donor_span_samples,
            end - start,
            out[start:end],
            level_reference=donor_level_reference,
        )
        # Put the replacement back INTO the background instead of in place of it.
        # Overwriting removed the clip's background from the span, which a listener
        # heard as "the substituted sound has no noise floor" (2026-09-21). See
        # mix_into_background for the level arithmetic.
        if preserve_background:
            # The background *signal* comes from an event-free window (so it is
            # genuinely background), but its *level* is measured inside the span
            # itself: TUT noise is not stationary across ten seconds, and a
            # distant window can be louder than the span it describes (10/600
            # items on the confirm grid).
            background, _ = background_estimate(
                out, scape.events, (start, end), sample_rate, rng
            )
            background_rms = local_background_rms(out[start:end])
            span_rms = float(
                np.sqrt(np.mean(np.square(out[start:end], dtype=np.float64)))
            )
            if donor_level_reference is not None:
                # CTL passes the event itself, so that array *is* the event level.
                event_rms = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.asarray(donor_level_reference, dtype=np.float32)
                            )
                        )
                    )
                )
            else:
                event_rms = event_rms_from_span(span_rms, background_rms)
            fitted = mix_into_background(fitted, background, background_rms, event_rms)
        return crossfade_replace(out, start, end, fitted, fade_samples)

    if condition == "nuisance_replaced":
        # Same operation, same donor, same duration as target_replaced_diff; only
        # the overwritten span differs, so the answer should not move.
        if donor_waveform is None or donor_span_samples is None:
            raise InstrumentError("nuisance_replaced requires a donor and its span")
        nuisance_span = resolve_condition_span(scape, condition, len(out), rng)
        if nuisance_span is None:
            raise InstrumentError("nuisance_replaced has no span")
        low, high = nuisance_span
        fitted = _fit_donor(
            donor_waveform, donor_span_samples, high - low, out[low:high]
        )
        return crossfade_replace(out, low, high, fitted, fade_samples)

    resolved = span or resolve_condition_span(scape, condition, len(out), rng)
    if resolved is None:
        raise InstrumentError(f"{condition} has no editable span")
    return _background_fill(out, resolved, scape.events, sample_rate, rng, fade_samples)


def background_estimate(
    waveform: np.ndarray,
    events: Sequence[PlacedEvent],
    span: tuple[int, int],
    sample_rate: int,
    rng: random.Random,
) -> tuple[np.ndarray, float]:
    """An event-free window of the clip's background, and its RMS.

    Returns ``(empty, 0.0)`` when no event-free window of the needed length
    exists, so callers fall back to donor-only behaviour instead of inventing a
    level.
    """

    start, end = span
    try:
        lo, hi = find_free_window(
            events, len(waveform), sample_rate, end - start, rng, exclude=span
        )
    except InstrumentError:
        return np.zeros(0, dtype=np.float32), 0.0
    window = np.asarray(waveform[lo:hi], dtype=np.float32)
    if window.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    return window, float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def local_background_rms(span: np.ndarray, frame: int = 320, percentile: float = 10.0) -> float:
    """The background level *inside* a span, from its own quietest frames.

    The span is event plus background, so its low-percentile frame RMS reads the
    background between the event's onsets. This is preferred to measuring an
    event-free window elsewhere in the clip: TUT urban noise is not stationary
    across ten seconds, so a distant window can be louder than the span it is
    meant to describe. Measured on the confirm grid, that happened for 10/600
    items, where the far window exceeded the span's own RMS and the mix had to
    fall back to donor-only.

    It also has no RNG dependence, so the estimate cannot drift when the window
    search changes.
    """

    x = np.asarray(span, dtype=np.float32).reshape(-1)
    n = len(x) // frame
    if n < 3:
        return 0.0
    frames = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n)]
    )
    return float(np.percentile(frames, percentile))


def mix_into_background(
    donor: np.ndarray,
    background: np.ndarray,
    background_rms: float,
    event_rms: float,
) -> np.ndarray:
    """Lay ``donor`` onto ``background`` at the level the replaced event had.

    WHY THIS EXISTS. The replacement used to *overwrite* the span with the donor's
    own samples, which removed the clip's background from that span along with the
    queried event. A listener on 2026-09-21 heard exactly that ("替换的异类声音
    没有底噪了，和环境音区别有点明显") and measurement agreed: the span's noise
    floor fell by a median of ~15 dB, and 68% of items by more than 10 dB. The
    replacement was donor-only, not donor-mixed-into-background.

    ``event_rms`` is the level the donor should carry. Inside a span the original
    power is approximately ``event^2 + background^2`` (event and background are
    independent recordings), so the event's own power is the difference between
    the span's power and the background's. The donor is scaled to that level and
    added to the background, which restores the noise floor while leaving the
    event-to-background ratio the SNR knob set intact.

    ``background`` is tiled or truncated to the donor's length. When either RMS is
    zero the donor is returned unchanged, preserving the previous behaviour rather
    than guessing a level.
    """

    donor = np.asarray(donor, dtype=np.float32).reshape(-1)
    if donor.size == 0 or background_rms <= 0.0 or event_rms <= 0.0:
        return donor

    background = np.asarray(background, dtype=np.float32).reshape(-1)
    if background.size == 0:
        return donor
    if background.size != donor.size:
        background = np.resize(background, donor.size)

    donor_rms = float(np.sqrt(np.mean(np.square(donor, dtype=np.float64))))
    if donor_rms <= 1e-8:
        return donor
    scaled = donor * (event_rms / donor_rms)
    return (scaled + background).astype(np.float32)


def event_rms_from_span(span_rms: float, background_rms: float) -> float:
    """The event's own RMS, given the span's and the background's.

    Powers add because the event and the background are independent recordings:
    ``span^2 = event^2 + background^2``. Clamped at zero, because a span quieter
    than the background estimate means the estimate is noisy, not that the event
    has negative power.
    """

    return float(np.sqrt(max(span_rms**2 - background_rms**2, 0.0)))


def _fit_donor(
    donor_waveform: np.ndarray,
    donor_span_samples: int,
    target_samples: int,
    reference: np.ndarray,
    *,
    level_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Cut a donor to its annotated span, fit it to the target span, match level.

    I1: the donor is trimmed to its own span first; the rest of the donor file is
    never used. I3: the result is resampled to the target length and its level is
    matched to the span it replaces by default, so an edit is comparable across
    conditions.

    ``reference`` is the span being replaced and is what the level is matched to by
    default. ``level_reference`` overrides *only* the level target — the length still
    comes from ``target_samples`` — and exists because matching to the span conflates
    the event with its masker: in a CTL cell the span is ``event + interferer``, so a
    loud masker makes the replacement loud too. CTL therefore passes the event array
    itself (see :func:`apply_condition`), which makes the replacement carry the event's
    own level and its difficulty track the ladder.
    """

    donor = np.asarray(donor_waveform, dtype=np.float32).reshape(-1)
    if not 0 < donor_span_samples <= len(donor):
        raise InstrumentError(
            f"donor span {donor_span_samples} outside donor of {len(donor)} samples"
        )
    donor = donor[:donor_span_samples]
    if target_samples <= 0:
        raise InstrumentError("target span must be positive")
    if len(donor) != target_samples:
        positions = np.linspace(0.0, len(donor) - 1, target_samples)
        donor = np.interp(positions, np.arange(len(donor)), donor).astype(np.float32)
    donor_rms = float(np.sqrt(np.mean(np.square(donor, dtype=np.float64))))
    level_source = reference if level_reference is None else level_reference
    level_source = np.asarray(level_source, dtype=np.float32).reshape(-1)
    if level_source.size == 0:
        raise InstrumentError("donor level reference is empty")
    reference_rms = float(np.sqrt(np.mean(np.square(level_source, dtype=np.float64))))
    if donor_rms > 1e-8 and reference_rms > 1e-8:
        donor = (donor * (reference_rms / donor_rms)).astype(np.float32)
    return donor


# --- rendering ---------------------------------------------------------------


def render_soundscape(
    output_audio: Path,
    output_jams: Path,
    foreground_dir: Path,
    background_dir: Path,
    plan: Sequence[dict[str, Any]],
    background: dict[str, Any],
    *,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_peak: float | None = DEFAULT_TARGET_PEAK,
    ref_db: float = -30.0,
    seed: int = 0,
) -> Soundscape:
    """Render one base soundscape with scaper and return it with exact events.

    ``plan`` is an ordered list of placements, each
    ``{"role", "label", "path", "start_s", "duration_s", "snr_db", "span_s"}``.
    ``background`` is ``{"label", "path"}``. Every scaper parameter is ``const``
    so that the JAMS file records exactly what we asked for.

    ``sample_rate`` is enforced on the output. scaper hardcodes ``sr = 44100`` on
    the Scaper object and resamples every source into it, so without this the
    rendered clips come back at 44.1 kHz regardless of the material.
    """

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    scaper_instance = scaper.Scaper(
        duration_s, str(foreground_dir), str(background_dir)
    )
    scaper_instance.sr = int(sample_rate)
    scaper_instance.ref_db = ref_db
    scaper_instance.add_background(
        label=("const", background["label"]),
        source_file=("const", str(background["path"])),
        source_time=("const", 0),
    )
    for placement in plan:
        scaper_instance.add_event(
            label=("const", placement["label"]),
            source_file=("const", str(placement["path"])),
            source_time=("const", 0),
            event_time=("const", float(placement["start_s"])),
            event_duration=("const", float(placement["duration_s"])),
            snr=("const", float(placement["snr_db"])),
            pitch_shift=("const", 0),
            time_stretch=("const", 1.0),
        )
    scaper_instance.generate(
        str(output_audio),
        str(output_jams),
        allow_repeated_label=True,
        allow_repeated_source=False,
        no_audio=False,
        # scaper defaults this to False. Without it, loud event classes drive the
        # mix past full scale and the render has to be rejected; measured at
        # roughly 30% of attempts before this was switched on. Normalisation is a
        # single gain on the whole mix, so the background-to-event ratio that the
        # SNR knob sets survives it, and every item lands at the same peak.
        peak_normalization=True,
    )

    waveform, sample_rate = sf.read(output_audio, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # scaper's peak_normalization scales the mix to exactly full scale, which is
    # both hotter than the models are usually fed and indistinguishable from
    # real clipping in an audit. Pull every item to the same target peak with a
    # single gain: the within-item background-to-event ratio is untouched, which
    # is the only level relationship the design depends on.
    peak = float(np.abs(waveform).max()) if waveform.size else 0.0
    if peak > 0 and target_peak is not None:
        scaled = (waveform * (float(target_peak) / peak)).astype(np.float32)
        sf.write(output_audio, scaled, sample_rate, subtype="PCM_16")
        waveform = sf.read(output_audio, dtype="float32", always_2d=False)[0]
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

    events = [
        PlacedEvent(
            role=placement["role"],
            label=placement["label"],
            source_path=str(placement["path"]),
            start_s=float(placement["start_s"]),
            end_s=float(placement["start_s"]) + float(placement["duration_s"]),
            snr_db=float(placement["snr_db"]),
            source_span_s=float(placement["span_s"]),
        )
        for placement in plan
    ]
    scape = Soundscape(
        audio_path=output_audio,
        sample_rate=int(sample_rate),
        duration_s=float(len(waveform)) / float(sample_rate),
        background_path=str(background["path"]),
        events=events,
    )
    assert_placement_disjoint(scape)
    return scape


def assert_placement_disjoint(scape: Soundscape) -> None:
    """Invariant I2, checked at construction time rather than audited later."""

    spans = sorted(
        (event.start_s, event.end_s, event.label) for event in scape.events
    )
    for (_, end_a, label_a), (start_b, _, label_b) in zip(spans, spans[1:]):
        if start_b - end_a < MIN_EVENT_GAP_S - 1e-6:
            raise InstrumentError(
                f"events {label_a!r} and {label_b!r} are {start_b - end_a:.3f}s "
                f"apart, below the {MIN_EVENT_GAP_S}s guard"
            )


# --- material indexing -------------------------------------------------------


def load_vocab(path: Path) -> list[str]:
    """Read a class vocabulary.

    Accepts the FSD-MIX-SED ``vocab.json`` (either a list or an index -> name
    mapping) and, as a convenience for screening runs and tests, a plain
    newline-separated text file.
    """

    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        payload = json.loads(text)
        if isinstance(payload, dict):
            return [str(payload[key]) for key in sorted(payload, key=lambda k: int(k))]
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def index_wavs(root: Path, min_duration_s: float = 0.0) -> dict[str, list[Path]]:
    """Group ``root/<label>/*.wav`` into a label -> paths mapping."""

    root = Path(root)
    index: dict[str, list[Path]] = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(label_dir.glob("*.wav"))
        if min_duration_s > 0:
            files = [f for f in files if wav_duration(f) >= min_duration_s]
        if files:
            index[label_dir.name] = files
    return index


def index_backgrounds(root: Path, min_duration_s: float = DEFAULT_DURATION_S) -> list[Path]:
    """Flat list of background files long enough to fill a clip."""

    return sorted(
        path
        for path in Path(root).rglob("*")
        if path.suffix.lower() in {".wav", ".flac"}
        and wav_duration(path) >= min_duration_s
    )


def wav_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception:
        return 0.0
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def read_audio_mono(
    path: Path, target_sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Read a file as mono float32, optionally resampling to a target rate.

    Needed because the event library is 44.1 kHz while the rendered soundscapes
    are 16 kHz. Passing a donor at its native rate makes its sample count
    disagree with the span arithmetic, which is exactly the bug the smoke test
    caught: a 0.9 s donor was reported as 39690 samples (0.9 s at 44.1 kHz) while
    the array held 14400 (0.9 s at 16 kHz).
    """

    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if target_sample_rate and int(target_sample_rate) != int(rate):
        from math import gcd

        from scipy.signal import resample_poly

        target = int(target_sample_rate)
        divisor = gcd(target, int(rate))
        waveform = resample_poly(
            waveform, target // divisor, int(rate) // divisor
        ).astype(np.float32)
        rate = target
    return waveform, int(rate)


def plan_placements(
    n_samples: int,
    sample_rate: int,
    query_duration_s: float,
    nuisance_durations: Sequence[float],
    rng: random.Random,
    duration_s: float = DEFAULT_DURATION_S,
) -> list[tuple[float, float]]:
    """Choose disjoint onsets for the query and nuisance events.

    Onsets sit on a grid, keep :data:`MIN_EVENT_GAP_S` between events, and stay
    inside the clip. Rejection is bounded: an item that cannot be placed is
    rejected rather than nudged into an overlap.
    """

    durations = [query_duration_s, *nuisance_durations]
    grid = max(1, int(round(PLACEMENT_GRID_S * sample_rate)))
    total = int(round(duration_s * sample_rate))
    for _ in range(500):
        spans: list[tuple[int, int]] = []
        for duration in durations:
            length = int(round(duration * sample_rate))
            if length >= total:
                return []
            slots = range(0, total - length + 1, grid)
            start = rng.choice(list(slots))
            spans.append((start, start + length))
        ordered = sorted(spans)
        ok = all(
            (b[0] - a[1]) >= int(round(MIN_EVENT_GAP_S * sample_rate))
            for a, b in zip(ordered, ordered[1:])
        )
        if ok:
            return [(a / sample_rate, b / sample_rate) for a, b in spans]
    return []
