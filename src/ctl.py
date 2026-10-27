"""CTL item construction: render matched-pair scenes and their manifest.

Each CTL item is one event in one background, rendered under four conditions:

===================  ==========================================================
``original``         scene as mixed, no waveform edit
``diff``             the queried span replaced by a different-class donor
``same``             the same span replaced by a same-class donor (magnitude control)
``sham``             the same span rewritten from its own samples (no-op floor)
===================  ==========================================================

```text
MIG = E_i[ 1(f(x_i^diff) != f(x_i^orig)) - 1(f(x_i^same) != f(x_i^orig)) ]
```

**Level parameterisation.** The event is normalised to a fixed RMS and then stays at
unity gain in *every* cell; the audibility axis is carried by the **masker level**,
solved per cell. Two measured failures forced this design:

* solving for the *event gain* returned 67x with an ``event_matched`` masker, so the
  scene peaked at 25.5 and PCM_16 writing clipped it to 0.99 — the rendered audio no
  longer contained the event while the reported audibility still described it
  (invariant I7 caught the consequence; the parameterisation was the cause);
* without normalising the event, the audibility of a harvested FSD50K event is fixed
  by the threshold in quiet rather than by the masker — measured at **-10.31 dB for
  every masker level**, so the "manipulation" would not have manipulated anything.

Keeping the event at a fixed level also makes the energy axis meaningful: the event is
identical across maskers by construction, so any difference in the mix is masker-only.

The axes are properties of the mix, so ``audibility`` and ``energy_snr`` are computed
**from the rendered scene**. The manifest therefore cannot describe a signal other
than the one on disk, which is what invariant I7 is about.

**Two level conventions, both recorded.** ``audibility_db`` / ``energy_snr_db`` describe
the *event* and are what the level solvers target. ``span_audibility_db`` /
``span_energy_snr_db`` describe the **cue that is actually in the queried span** of the
file that was written — the event for ``original``/``sham``, the fitted replacement for
``diff``/``same`` — and are computed from the rendered array rather than from the item
(see :func:`span_metrics` for why the mixture is the wrong signal to measure). An analysis
that wants to know how hard a recording was must fit against the ``span_*`` pair: on an
edited arm the event is not in the audio, so the event-referenced pair describes a signal
the model never heard. The edited arms level their donor against the **event it replaces**,
not against the masked span (see :func:`src.synth.apply_condition`), so the replacement
carries the event's difficulty rather than the masker's.

Output layout::

    outputs/ctl/<split>/audio/<item_id>__<calibration>__<masker>__<condition>.wav
    outputs/ctl/<split>/manifest.jsonl
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audibility import audibility_margin_db
from src.maskers import (
    MaskerSpec,
    normalise_event,
    shaped_noise,
    solve_masker_attenuation_db,
    solve_masker_for_energy_snr_db,
)
from src.synth import (
    DEFAULT_DURATION_S,
    DEFAULT_FADE_MS,
    DEFAULT_SAMPLE_RATE,
    PlacedEvent,
    Soundscape,
    apply_condition,
    boundary_jump,
    read_audio_mono,
)

__all__ = [
    "CTLItem",
    "CTL_GRID",
    "LevelTarget",
    "build_item",
    "resolve_level_targets",
    "render_item",
    "span_metrics",
    "write_manifest",
    "question_for",
    "audit_bit_identity",
    "audit_sham",
]

CTL_GRID = {
    "calibration": ("equal_energy", "equal_audibility"),
    "maskers": ("low_pass_400", "event_matched"),
}
MARGIN_DROP_DB = 15.0  # nominal audibility drop; falls back per item if unreachable
MARGIN_DROP_FALLBACKS = (15.0, 12.0, 9.0, 6.0)
EVENT_TARGET_RMS = 0.15
MASKER_REFERENCE_RMS = 1.0  # attenuation is measured against a unit-RMS template
PEAK_CEILING = 0.98  # shared-gain headroom; leaves room for the donor edit
DEFAULT_CONDITIONS = ("original", "diff", "same", "sham")
LEVEL_QUANTITIES = ("energy_snr_db", "audibility_margin_db")


@dataclass(frozen=True)
class LevelTarget:
    """One (calibration, masker) cell whose level is solved for a stated quantity.

    The default CTL grid solves two shared cells per item (equal energy, equal
    audibility) because its contrast is *between masker spectra* at one level. The axis
    duel needs the opposite: one cell per level, each solved for a different quantity —
    an energy SNR on the energy axis, an audibility margin on the audibility axis. Both
    are the same operation (pick a masker level so a measured property of the mix hits a
    number), so it lives here rather than in the builder and every render still goes
    through :func:`render_item` and its assertions.

    ``calibration`` becomes the cell's key in ``CTLItem.masker_rms`` and in the audio
    filename, so a duel item carries e.g. ``energy_L2`` and ``audibility_L2`` cells.
    ``relative_to_baseline`` expresses an audibility target as a *drop* below the item's
    own unmasked margin, which is the only form that is reachable across items: the
    unmasked margin itself varies by ~35 dB across the corpus.
    """

    calibration: str
    masker: str
    quantity: str
    target: float
    relative_to_baseline: bool = False

    def __post_init__(self) -> None:
        if self.quantity not in LEVEL_QUANTITIES:
            raise ValueError(f"unknown level quantity {self.quantity!r}")
        if self.relative_to_baseline and self.quantity != "audibility_margin_db":
            raise ValueError("only an audibility margin can be relative to the baseline")


@dataclass
class CTLItem:
    """One CTL item with everything the manifest and the audit need."""

    item_id: str
    query_class: str
    donor_class: str
    event: np.ndarray
    background: np.ndarray
    maskers: dict[str, np.ndarray]  # unit-RMS templates
    masker_rms: dict[str, dict[str, float]]  # [calibration][masker] -> solved level
    baseline_margin_db: float  # margin with no masker at all (this item's anchor)
    sample_rate: int
    query_start_sample: int
    seed: int
    same_donor_class: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def query_span(self) -> tuple[int, int]:
        return self.query_start_sample, self.query_start_sample + self.event.size

    def level_of(self, calibration: str, masker: str) -> float:
        return self.masker_rms[calibration][masker]

    def scaled_masker(self, calibration: str, masker: str) -> np.ndarray:
        return (self.maskers[masker] * self.level_of(calibration, masker)).astype(np.float32)

    def scene_for(self, calibration: str, masker: str) -> np.ndarray:
        """background + masker at this cell's solved level + event at unity gain."""

        out = (self.background + self.scaled_masker(calibration, masker)).astype(np.float32)
        start, end = self.query_span
        out[start:end] = out[start:end] + self.event
        return out

    def _interferer(self, calibration: str, masker: str) -> np.ndarray:
        """Background + masker over the query span — the event is deliberately excluded.

        The threshold must be computed from the *interferers*, not from the scene. An
        earlier version passed the whole scene span here, so the event was counted into
        its own masking threshold; two conventions then coexisted in the codebase
        (masker-only in the maskers module, scene-span here) and disagreed by ~40 dB on
        the same item, which made the level solver chase an unreachable target.
        """

        start, end = self.query_span
        interferer = (self.background + self.scaled_masker(calibration, masker)).astype(np.float32)
        return interferer[start:end][: self.event.size]

    def audibility(self, calibration: str, masker: str) -> float:
        return float(
            audibility_margin_db(
                self.event, self._interferer(calibration, masker), self.sample_rate
            )
        )

    def unmasked_margin_db(self) -> float:
        """Audibility margin of the event against the background alone, at this scale.

        ``baseline_margin_db`` is the value measured when the item was assembled, i.e. at
        unit gain. A build that applies a shared gain for headroom moves it, because the
        threshold-in-quiet floor does not scale with the mix: at low gain the event's
        band power falls while the floor stays put, so the unmasked margin falls with it.
        An audibility target expressed as "drop below the item's own unmasked baseline"
        has to be measured against the baseline of the *audio that is rendered*, which is
        what this returns.
        """

        start, end = self.query_span
        return float(
            audibility_margin_db(
                self.event,
                self.background[start:end][: self.event.size],
                self.sample_rate,
            )
        )

    def energy_snr(self, calibration: str, masker: str) -> float:
        event_rms = float(np.sqrt(np.mean(self.event.astype(np.float64) ** 2)))
        ref = self._interferer(calibration, masker)
        ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
        return float(20.0 * np.log10(max(event_rms, 1e-12) / max(ref_rms, 1e-12)))

    def soundscape(self) -> Soundscape:
        start = self.query_start_sample / self.sample_rate
        return Soundscape(
            audio_path=Path(""),
            sample_rate=self.sample_rate,
            duration_s=self.background.size / self.sample_rate,
            background_path=str(self.metadata.get("background_source", "")),
            events=[
                PlacedEvent(
                    role="query",
                    label=self.query_class,
                    source_path=str(self.metadata.get("event_source", "")),
                    start_s=start,
                    end_s=start + self.event.size / self.sample_rate,
                    snr_db=0.0,
                    source_span_s=self.event.size / self.sample_rate,
                )
            ],
        )

    def scaled_copy(self, gain: float) -> "CTLItem":
        """A copy with event, background and masker templates all scaled by one gain.

        This is the only permitted level correction (docs/protocol-invariants.md, §2.3 rule 2): one gain on
        the whole mix. Energy SNR is exactly invariant under it, and the audibility
        margin is invariant wherever the masker rather than the threshold in quiet sets
        the threshold. Scaling the masker alone (or the event alone) would re-couple the
        two axes, which is the failure the whole parameterisation exists to avoid.

        It exists for headroom: a masker loud enough to bury the event can push the mix
        over the PCM_16 ceiling, and the splice can add to that. The copy keeps the
        caller's item intact, so a build that renders one item into many cells cannot
        accumulate gains, and it keeps the recorded calibration describing the audio on
        disk rather than the audio before the gain.
        """

        gain = float(gain)
        if not gain > 0.0:
            raise ValueError(f"shared gain must be positive, got {gain}")
        # ``shared_gain`` records the *cumulative* gain on the mix. It used to be
        # overwritten with the latest factor, so a cell fitted over several headroom
        # iterations reported only the last one and understated the level change by the
        # product of the earlier ones (measured on the axis-duel build: a cell whose true
        # gain was 0.435 reported 0.299). The individual factors are kept alongside it so
        # the record can be checked rather than trusted.
        previous = float(self.metadata.get("shared_gain", 1.0))
        steps = [float(v) for v in self.metadata.get("shared_gain_steps", [1.0])]
        return dataclasses.replace(
            self,
            event=(self.event * gain).astype(np.float32),
            background=(self.background * gain).astype(np.float32),
            maskers={k: (v * gain).astype(np.float32) for k, v in self.maskers.items()},
            metadata={
                **self.metadata,
                "shared_gain": previous * gain,
                "shared_gain_steps": [*steps, gain],
            },
        )


def build_item(
    item_id: str,
    query_class: str,
    donor_class: str,
    event_source: Path,
    background_source: Path,
    donor_source: Path,
    *,
    same_donor_class: str = "",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    event_start_s: float = 2.0,
    margin_drop_db: float = MARGIN_DROP_DB,
    event_target_rms: float = EVENT_TARGET_RMS,
    maskers: dict[str, MaskerSpec] | None = None,
    seed: int = 0,
    level_targets: tuple[LevelTarget, ...] | list[LevelTarget] | None = None,
) -> CTLItem:
    """Assemble one CTL item: normalised event, background, masker pair, solved levels.

    ``level_targets`` replaces the default two-cell calibration with explicit cells, one
    per level, each solved for its own quantity (see :class:`LevelTarget`). The default
    path is unchanged when it is omitted, so existing builds keep their exact levels.
    """

    rng = random.Random(seed)
    specs = maskers or _default_specs()
    event = normalise_event(read_audio_mono(event_source, sample_rate)[0], event_target_rms)
    background = _background_track(background_source, sample_rate, duration_s, rng)
    donor, _ = read_audio_mono(donor_source, sample_rate)

    n = int(duration_s * sample_rate)
    start = int(event_start_s * sample_rate)
    if start + event.size > n:
        raise ValueError(
            f"event of {event.size} samples does not fit at {event_start_s}s in {duration_s}s"
        )
    if event.size >= int(0.6 * n):
        raise ValueError(f"event occupies {event.size}/{n} samples; too long for a clean background")

    masker_waves: dict[str, np.ndarray] = {}
    solved_levels: dict[str, float] = {}
    for name, spec in specs.items():
        template = shaped_noise(duration_s, sample_rate, _with_seed(spec, seed), event=event)
        template_rms = float(np.sqrt(np.mean(template.astype(np.float64) ** 2)))
        masker_waves[name] = (template / template_rms).astype(np.float32)  # unit-RMS template

    # Anchor the audibility axis to each item's own unmasked margin. An absolute
    # target cannot work here: the margin with no masker at all is set by the
    # threshold in quiet and varies across items by ~35 dB (measured +3.5 to +38.7 dB),
    # so a fixed target asks some items for a masker level that does not exist.
    baseline = float(
        audibility_margin_db(
            event, background[start : start + event.size][: event.size], sample_rate
        )
    )
    # A fixed drop is not always reachable: how much a masker can lower the margin
    # depends on the event's own spectrum and level, and measured across classes the
    # achievable drop spans roughly 8-30 dB. The drop is therefore nominal and falls
    # back, and what was actually achieved is recorded so the analysis can report the
    # realised manipulation rather than the intended one.
    templates = {
        name: shaped_noise(duration_s, sample_rate, _with_seed(spec, seed), event=event)
        for name, spec in specs.items()
    }
    if level_targets is not None:
        item = CTLItem(
            item_id=item_id,
            query_class=query_class,
            donor_class=donor_class,
            same_donor_class=same_donor_class,
            event=event,
            background=background,
            maskers=masker_waves,
            masker_rms={},
            baseline_margin_db=baseline,
            sample_rate=sample_rate,
            query_start_sample=start,
            seed=seed,
            metadata=_base_metadata(
                event_source, background_source, donor_source, donor, duration_s,
                event_target_rms, margin_drop_db, baseline,
            ),
        )
        resolve_level_targets(item, level_targets)
        item.metadata["level_targets"] = [dataclasses.asdict(t) for t in level_targets]
        return item

    attempted: list[tuple[float, float, dict[str, float], float]] = []
    for drop in (float(margin_drop_db),) + tuple(
        d for d in MARGIN_DROP_FALLBACKS if d < float(margin_drop_db)
    ):
        target = baseline - drop
        levels: dict[str, float] = {}
        try:
            for name, template in templates.items():
                _, level = solve_masker_attenuation_db(
                    event,
                    template,
                    sample_rate,
                    reference_rms=MASKER_REFERENCE_RMS,
                    target_margin_db=target,
                )
                levels[name] = level
        except AssertionError:
            continue
        achieved_levels = []
        for name, level in levels.items():
            interferer = (background + templates[name] * level).astype(np.float32)
            achieved_levels.append(
                audibility_margin_db(
                    event, interferer[start : start + event.size][: event.size], sample_rate
                )
            )
        achieved = float(np.mean(achieved_levels))
        attempted.append((drop, target, levels, achieved))
        break
    if not attempted:
        raise AssertionError(
            f"no fallback drop was reachable for {item_id}: baseline {baseline:+.2f} dB"
        )
    realised_drop, target, solved_levels, achieved = attempted[0]

    # equal_energy: one shared level so the event sits at equal energy under both
    # maskers and only the spectrum differs. It cannot also hit the audibility target
    # exactly (that is the point of the contrast), so the shared level is the
    # geometric mean of the two solved levels.
    shared = float(np.exp(np.mean(np.log(list(solved_levels.values())))))
    masker_rms = {
        "equal_energy": {name: shared for name in solved_levels},
        "equal_audibility": dict(solved_levels),
    }

    return CTLItem(
        item_id=item_id,
        query_class=query_class,
        donor_class=donor_class,
        same_donor_class=same_donor_class,
        event=event,
        background=background,
        maskers=masker_waves,
        masker_rms=masker_rms,
        baseline_margin_db=baseline,
        sample_rate=sample_rate,
        query_start_sample=start,
        seed=seed,
        metadata={
            **_base_metadata(
                event_source, background_source, donor_source, donor, duration_s,
                event_target_rms, margin_drop_db, baseline,
            ),
            "margin_drop_realised_db": float(realised_drop),
            "target_margin_db": target,
        },
    )


def _base_metadata(
    event_source: Path,
    background_source: Path,
    donor_source: Path,
    donor: np.ndarray,
    duration_s: float,
    event_target_rms: float,
    margin_drop_db: float,
    baseline: float,
) -> dict[str, object]:
    """Provenance every CTL item carries, whichever calibration path built it."""

    return {
        "event_source": str(event_source),
        "background_source": str(background_source),
        "donor_source": str(donor_source),
        "donor_samples": int(donor.size),
        "duration_s": float(duration_s),
        "event_target_rms": float(event_target_rms),
        "margin_drop_nominal_db": float(margin_drop_db),
        "baseline_margin_db": float(baseline),
    }


def resolve_level_targets(
    item: CTLItem,
    targets: tuple[LevelTarget, ...] | list[LevelTarget],
) -> dict[str, dict[str, float]]:
    """Solve every target's masker level and install it in ``item.masker_rms``.

    Both solvers measure the *realised* interferer (background + scaled masker over the
    query span) rather than the masker alone, so the value recorded in the manifest is
    the value that was solved for. The masker slice is normalised by its own RMS inside
    the solver and the returned level is divided by that RMS, because a level is defined
    as the multiplier on the item's *full-length* unit-RMS template — the slice's RMS is
    within ~2% of 1, which is 0.2 dB of axis error if ignored.

    Callable again after :meth:`CTLItem.scaled_copy`: the energy targets are invariant
    under a shared gain, but an audibility target is only invariant while the masker and
    not the threshold in quiet sets the threshold, so a re-solve is the honest thing to
    do rather than assuming it did not matter.
    """

    span = item.query_span
    levels: dict[str, dict[str, float]] = {}
    for target in targets:
        if target.masker not in item.maskers:
            raise KeyError(f"no masker template {target.masker!r} on {item.item_id}")
        template = item.maskers[target.masker]
        sliced = template[span[0] : span[1]][: item.event.size].astype(np.float64)
        slice_rms = float(np.sqrt(np.mean(sliced ** 2)))
        if slice_rms <= 0.0:
            raise ValueError(f"empty masker slice for {target.masker!r}")
        background = item.background[span[0] : span[1]][: item.event.size]
        if target.quantity == "energy_snr_db":
            _, level = solve_masker_for_energy_snr_db(
                item.event,
                sliced,
                item.sample_rate,
                reference_rms=MASKER_REFERENCE_RMS,
                target_snr_db=float(target.target),
                background=background,
            )
        else:
            wanted = (
                item.baseline_margin_db - float(target.target)
                if target.relative_to_baseline
                else float(target.target)
            )
            _, level = solve_masker_attenuation_db(
                item.event,
                sliced,
                item.sample_rate,
                reference_rms=MASKER_REFERENCE_RMS,
                target_margin_db=wanted,
                background=background,
            )
        levels.setdefault(target.calibration, {})[target.masker] = level / slice_rms
    item.masker_rms = levels
    return levels


def _default_specs() -> dict[str, MaskerSpec]:
    return {
        "low_pass_400": MaskerSpec(
            kind="low_pass", rms=MASKER_REFERENCE_RMS, seed=0, cutoff_hz=400.0
        ),
        "event_matched": MaskerSpec(kind="event_matched", rms=MASKER_REFERENCE_RMS, seed=0),
    }


def _with_seed(spec: MaskerSpec, seed: int) -> MaskerSpec:
    return MaskerSpec(
        kind=spec.kind, rms=spec.rms, seed=seed, cutoff_hz=spec.cutoff_hz, order=spec.order
    )


def _background_track(
    path: Path, sample_rate: int, duration_s: float, rng: random.Random
) -> np.ndarray:
    wave, _ = read_audio_mono(path, sample_rate)
    need = int(duration_s * sample_rate)
    if wave.size >= need:
        offset = rng.randrange(0, wave.size - need + 1)
        return wave[offset : offset + need].copy()
    reps = int(np.ceil(need / wave.size))
    return np.tile(wave, reps)[:need].copy()


def render_item(
    item: CTLItem,
    output_dir: Path,
    *,
    calibration: str,
    masker: str,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    donor: np.ndarray | None = None,
    donor_span_samples: int | None = None,
    same_donor: np.ndarray | None = None,
    same_donor_span_samples: int | None = None,
) -> dict[str, object]:
    """Write the recordings for one cell; return manifest rows and the audited scene.

    ``donor`` is the **different-class** donor and ``same_donor`` the **same-class**
    one; both must be the donor's own event span, never a whole clip (invariant I1).

    Two separate donors are mandatory, not a convenience. Until 2026-09-17 this
    function took a single ``donor`` and passed it to *both* edited conditions, so
    ``diff`` and ``same`` were the same bytes while the manifest gave them opposite
    ground truths — every MIG computed from them was therefore identically zero by
    construction, and two rounds of conclusions were drawn from it. Asking for an
    explicit same-class donor, and refusing to render ``same`` without one, is the
    structural fix; the build also asserts the two files differ. The same guard is
    repeated here for the caller who passes two *references* to one array, which the
    builder-side check could not see (CTL-Plan §3.4i#2): identical donors make the two
    arms identical, so the function refuses rather than rendering a void contrast.

    The donor is levelled to the **event it replaces**, not to the masked span: the span
    is ``event + interferer``, so levelling to it would make the replacement as loud as
    the masker. See :func:`src.synth.apply_condition`.

    Every row records the level of the cue that is actually in the queried span of the
    file that was written (``span_audibility_db`` / ``span_energy_snr_db``), next to the
    event-referenced ``audibility_db`` / ``energy_snr_db`` that predate this and are kept
    for compatibility. For ``diff``/``same`` the two differ: the event is gone from the
    audio, so the event-referenced pair describes a signal that is not in that recording.
    """

    if "same" in conditions and same_donor is None:
        raise AssertionError(
            "rendering the `same` condition requires an explicit same-class donor; "
            "falling back to the different-class donor makes diff and same identical "
            "and drives MIG to zero by construction"
        )
    if donor is not None and same_donor is not None:
        donor_array = np.asarray(donor).reshape(-1)
        same_array = np.asarray(same_donor).reshape(-1)
        if donor_array is same_array or (
            donor_array.shape == same_array.shape and np.array_equal(donor_array, same_array)
        ):
            raise AssertionError(
                "diff and same were given the same donor array; the two edited arms "
                "would be byte-identical while the manifest assigns them opposite ground "
                "truths, which drives MIG to zero by construction"
            )

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    scene = item.scene_for(calibration, masker)
    peak = float(np.abs(scene).max())
    row_item = item
    if peak > PEAK_CEILING:
        # One gain on the whole mix. This is the only permitted level correction
        # (docs/protocol-invariants.md, §2.3 rule 2): it scales background, masker and event together, so
        # the event-to-interferer relationship that the two axes control is untouched.
        # Per-condition or per-component scaling here would re-couple energy and
        # audibility, which is the failure this whole parameterisation exists to avoid.
        #
        # The manifest is then written from a scaled *copy*: the calibration it records
        # must describe the array that was written to disk, not the array from before
        # the gain, and the caller's item must not be mutated (a build that renders one
        # item into many cells would otherwise accumulate gains across cells).
        gain = PEAK_CEILING / peak
        scene = (scene * gain).astype(np.float32)
        row_item = item.scaled_copy(gain)
        item.metadata.setdefault("applied_gains", {})
        item.metadata["applied_gains"][f"{calibration}__{masker}"] = float(gain)  # type: ignore[index]
        peak = float(np.abs(scene).max())
    if peak > 1.0:
        raise AssertionError(
            f"scene still peaks at {peak:.3f} after the shared gain "
            f"({item.item_id}/{calibration}/{masker})"
        )
    rng = random.Random(item.seed)
    scape = item.soundscape()

    rows: list[dict[str, object]] = []
    written: dict[str, Path] = {}
    for condition in conditions:
        # each edited arm gets its own donor; this is what makes the contrast a contrast
        if condition == "same":
            wave, span_samples = same_donor, same_donor_span_samples
        else:
            wave, span_samples = donor, donor_span_samples
        edited = apply_condition(
            scene,
            scape,
            _condition_name(condition),
            donor_waveform=wave,
            donor_span_samples=span_samples,
            # The replacement is levelled to the event it replaces, at the scale of the
            # array being written (``row_item``, not ``item``: a headroom gain scales the
            # scene and the event together, and the reference must be on the same scale as
            # the scene the splice is applied to).
            donor_level_reference=row_item.event if condition in {"diff", "same"} else None,
            span=item.query_span if condition in {"diff", "same", "sham"} else None,
            rng=rng,
            # CTL is closed and its calibration was fitted against donor-only
            # replacements, so keep that behaviour rather than invalidating the
            # frozen artifacts. See src.synth.apply_condition.
            preserve_background=False,
        )
        # The pre-splice peak check above cannot see the splice, and the fitted donor is
        # level-matched to the span it replaces — so an impulsive donor can and did push
        # an edited arm into hard clipping (154/240 `diff` files in sample60_v2, peak
        # 32768/32768). The check that matters is after the splice, on the array that is
        # about to be written.
        condition_peak = float(np.abs(edited).max())
        if condition_peak > 1.0:
            raise AssertionError(
                f"{item.item_id}/{calibration}/{masker}/{condition} peaks at "
                f"{condition_peak:.4f} > 1.0 after the splice; reserve headroom with "
                f"CTLItem.scaled_copy or lower the masker level"
            )
        path = audio_dir / f"{item.item_id}__{calibration}__{masker}__{condition}.wav"
        sf.write(path, edited.astype(np.float32), item.sample_rate, subtype="PCM_16")
        written[condition] = path
        rows.append(
            _manifest_row(
                row_item,
                calibration,
                masker,
                condition,
                path,
                span_metrics=span_metrics(
                    row_item,
                    calibration,
                    masker,
                    edited,
                    replaced=condition in {"diff", "same"},
                ),
            )
        )
    return {"rows": rows, "audio": written, "scene": scene, "peak": peak}


def _condition_name(condition: str) -> str:
    return {
        "original": "original",
        "diff": "target_replaced_diff",
        "same": "target_replaced_same",
        "sham": "sham",
    }[condition]


def question_for(query_class: str, absent_class: str) -> str:
    """The closed-form question, byte-identical in shape to the mainline template.

    Reusing the template matters for more than convenience: the evaluator parses
    answers by looking for `Answer: A` and for a leading option letter, so a
    differently worded question would silently change the parse rate and make the
    CTL numbers incomparable with the mainline's.
    """

    return (
        "Listen to the audio. Which of these two sounds is present in it? "
        f"Option A: '{query_class}'. Option B: '{absent_class}'. "
        "Respond with exactly Answer: A or Answer: B."
    )


def span_metrics(
    item: CTLItem,
    calibration: str,
    masker: str,
    waveform: np.ndarray,
    *,
    replaced: bool,
) -> dict[str, object]:
    """Level of the cue that is actually in the queried span, versus its interferer.

    This is the honest answer to "how hard is this recording?", and it is what a reader
    must fit accuracy against. ``CTLItem.audibility`` measures the *event*, which is the
    right quantity for the ``original`` arm and for the level solver, but the wrong one for
    an edited arm: there the event has been overwritten, so the event-referenced number
    describes a signal that is no longer in the file. Before this existed, the axis-duel
    manifest reported one event-referenced audibility for both arms, and on 400 of 800 rows
    it described audio that was not on disk.

    **The cue, not the mixture.** ``replaced`` says whether the condition overwrote the
    span with a donor (``diff``/``same``) or kept the event (``original``/``sham``):

    * kept — the span is ``event + interferer``, so the cue is ``span - interferer`` (the
      event). Measuring the whole span instead measures the *mixture*, whose band power at
      the hard end is the masker's own, so it saturates near 0 dB: measured on the
      superseded build, the original arm's whole-span audibility moved 5.8 dB across a
      30 dB ladder while the event's own margin moved 30.7 dB.
    * replaced — the splice removes the interferer from the span, so the span *is* the
      replacement and the cue is the span itself.

    The crossfade edges are excluded: they legitimately retain a little of the original
    content, and including them made the hardest cells look 3.5 dB easier than the
    replacement actually is (measured: whole-span energy SNR -21.5 dB against the
    replacement's own -25.0 dB).

    Everything is measured on the array that is written (the caller passes ``edited``), so
    invariant I7 holds by construction rather than by review. The interferer is the
    background plus the scaled masker over the same span, event excluded — the same
    convention as :meth:`CTLItem.audibility`, so the numbers are comparable.
    """

    start, end = item.query_span
    span = np.asarray(waveform[start:end], dtype=np.float32)
    interferer = item._interferer(calibration, masker)[: span.size]
    if interferer.size != span.size:
        raise AssertionError(
            f"interferer of {interferer.size} samples does not match the queried span of "
            f"{span.size} ({item.item_id}/{calibration}/{masker})"
        )
    # the crossfade length apply_condition uses by default; excluding those edges keeps the
    # measurement about the cue rather than about the splice
    guard = int(round(DEFAULT_FADE_MS * item.sample_rate / 1000.0))
    if span.size > 4 * guard:
        interior = slice(guard, span.size - guard)
    else:
        interior = slice(0, span.size)
    span_slice = span[interior]
    interferer_slice = interferer[interior]
    cue = span_slice if replaced else (span_slice - interferer_slice).astype(np.float32)
    cue_rms = float(np.sqrt(np.mean(cue.astype(np.float64) ** 2)))
    interferer_rms = float(np.sqrt(np.mean(interferer_slice.astype(np.float64) ** 2)))
    return {
        "span_rms": cue_rms,
        "span_interferer_rms": interferer_rms,
        "span_cue_signal": "replacement" if replaced else "event",
        "span_audibility_db": float(
            audibility_margin_db(cue, interferer_slice, item.sample_rate)
        ),
        "span_energy_snr_db": float(
            20.0 * np.log10(max(cue_rms, 1e-12) / max(interferer_rms, 1e-12))
        ),
    }


def _manifest_row(
    item: CTLItem,
    calibration: str,
    masker: str,
    condition: str,
    path: Path,
    *,
    span_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    start, end = item.query_span
    row: dict[str, object] = {
        "item_id": item.item_id,
        "pair_id": f"{item.item_id}__{calibration}__{masker}__{condition}",
        "source_pair_id": f"{item.item_id}__{calibration}__{masker}",
        "condition": condition,
        "audio": str(path),
        # fields the existing evaluator and analysis expect (see its PROVENANCE_FIELDS)
        "question": question_for(item.query_class, item.donor_class),
        "option_events": {"A": item.query_class, "B": item.donor_class},
        "surface_order": 0,
        "intervention_role": condition,
        # A = the queried sound, B = the donor sound. In `original` the queried sound
        # is present; in `diff` it has been replaced by the donor's class, so the
        # answer flips to B; in `same` it was replaced by another instance of its own
        # class, so it stays A; `sham` rewrites the span with its own samples.
        "ground_truth": "B" if condition == "diff" else "A",
        # CTL axes, independent by construction
        "calibration": calibration,
        "masker": masker,
        # These two describe the *event* and are
        # kept unchanged for backward compatibility; on an edited arm the event is no
        # longer in the audio, so they describe the cue the cell was calibrated for, not
        # the cue that is present. ``audibility_db_reference`` says so in the row itself.
        "audibility_db": item.audibility(calibration, masker),
        "energy_snr_db": item.energy_snr(calibration, masker),
        "audibility_db_reference": "event",
        "masker_rms": item.level_of(calibration, masker),
        "event_rms": float(np.sqrt(np.mean(item.event.astype(np.float64) ** 2))),
        # parity with the mainline manifest
        "query_class": item.query_class,
        "absent_class": item.donor_class,
        "same_class": item.same_donor_class,
        "donor_used": item.same_donor_class if condition == "same" else (
            item.donor_class if condition == "diff" else None),
        "query_start_s": start / item.sample_rate,
        "query_end_s": end / item.sample_rate,
        "sample_rate": item.sample_rate,
        "seed": item.seed,
        "split": "ctl",
        **{f"meta_{k}": v for k, v in item.metadata.items()},
    }
    if span_metrics is not None:
        # what is actually in the queried span of the file that was written
        row.update(
            {
                **span_metrics,
                "span_audibility_db_reference": "queried_span_cue",
                "span_energy_snr_db_reference": "queried_span_cue",
            }
        )
    return row


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_bit_identity(
    original_path: Path, scene: np.ndarray, sample_rate: int
) -> dict[str, float | bool]:
    """Invariant I7: rendered audio equals the audited array up to PCM_16 quantisation."""

    on_disk, rate = sf.read(original_path, dtype="float32", always_2d=False)
    if int(rate) != int(sample_rate):
        raise AssertionError(f"sample rate changed on write: {rate} != {sample_rate}")
    delta = np.abs(on_disk - scene)
    return {
        "max_abs_delta": float(delta.max()),
        "within_one_lsb": bool(delta.max() <= 1.5 / 32768.0),
    }


def audit_sham(scene: np.ndarray, sham: np.ndarray, span: tuple[int, int]) -> dict[str, float]:
    """The sham must be a true no-op at the edit boundary (the withdrawn build was not)."""

    return {
        "max_abs_delta": float(np.abs(sham - scene).max()),
        "boundary_jump_original": float(boundary_jump(scene, *span)),
        "boundary_jump_sham": float(boundary_jump(sham, *span)),
    }
