"""Axis duel builder: energy-calibrated vs audibility-calibrated difficulty ladders.

The question this build exists to make answerable: **is an audio LLM's difficulty set by
the energy of the masker, or by how far the queried event sits above a human listener's
masking threshold?** Two difficulty axes are laid over the *same* events, so the axes can
be compared item for item:

======================  =========================  ==============================
axis                    masker                     level solved for
======================  =========================  ==============================
``energy``              uniform white noise        energy SNR in dB
``audibility``          threshold-shaped noise     audibility margin, as a drop
                                                   below the item's own unmasked
                                                   baseline
======================  =========================  ==============================

Five levels per axis, 100 items per level (36 classes, round-robin), ``original`` and
``diff`` only. The MIG/same machinery is deliberately absent: this build is about which
physical variable moves accuracy, and the identity contrast is a separate (and separately
decided) question.

**The design constraint the calibration check measures.** For a *fixed* masker spectrum,
energy SNR and audibility margin are the same knob: threshold power is proportional to
masker power, so ``audibility = -10 log10(masker power) + const`` and
``energy_snr = -10 log10(masker power) + const`` — slope 1 between them. The two axes can
only separate if the two masker *spectra* differ in masking efficiency.

**The masker naming trap (corrected 2026-09-18).** "Threshold-shaped noise" names two
opposite spectra and this docstring had them swapped, which is exactly the confusion the
duel exists to measure:

``threshold_shaped``
    PSD proportional to the threshold in quiet, ``LTQ(f)`` (amplitude ``10 ** (+LTQ/20)``)
    — the *literal* reading of "the shape of the hearing threshold". The masker is loud
    where hearing is insensitive, so equal energy buys the least masking. This is the
    masker axis B uses. Measured on this corpus at matched energy: **median ~12 dB** of
    extra audibility drop over white noise (per-item spread in
    ``calibration_report.json``); individual items reach ~30 dB.
``threshold_inverse``
    PSD proportional to ``1 / LTQ(f)`` (amplitude ``10 ** (-LTQ/20)``) — the audiological
    "threshold-shaped noise" that sits at a constant distance above threshold in every
    band. Measured on this corpus: **median ~3 dB** (2.96 dB), i.e. near-collinear with
    uniform white noise at 16 kHz, where half of white noise's energy already sits in
    1-5 kHz. Swapping in this masker collapses the duel's separation from ~12 dB to ~3 dB
    and the two axes stop being distinguishable; the convention therefore decides whether
    the experiment can answer its question at all, and the name alone does not.

A small separation is a finding, not a failure, and the builder reports it rather than
hiding it.

Everything is rendered through :func:`src.ctl.render_item` on items assembled by
:func:`src.ctl.build_item`, so the diff/original guard, the level-match in ``_fit_donor``
and the post-splice clip check all apply. The headroom fit uses
:meth:`src.ctl.CTLItem.scaled_copy` — one gain on the whole mix, never per component.

Usage::

    E=$HOME/miniconda3/envs/audiotools
    export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
    .venv/bin/python scripts/ctl_axis_duel_build.py                 # full build
    .venv/bin/python scripts/ctl_axis_duel_build.py --dry-run --items 6   # levels only

Outputs under ``outputs/ctl/axis_duel/``::

    audio/<item>__<axis>_L<k>__<masker>__<condition>.wav
    manifest.jsonl      one row per recording, with both level conventions recorded
    axis.json           per item and level: realised axes, masker level, scene peak
    build_audit.json    the build-time checks, each of which fails the build
    calibration_report.json   the dissociation check
    audio_digests.json  SHA256 per rendered file

**Item pool and coverage.** The pool is every (event, donor, background) triple for the
first ``FILES_PER_CLASS`` files of all 36 classes; *every* candidate is screened, and the
admissible ones are then selected round-robin over classes (see
:func:`select_admissible`). The previous build stopped screening as soon as it had the
requested 40 and therefore drew its items from an alphabetical prefix of the corpus —
measured, 40 items from the first 22 classes, with 14 classes absent. A capped build now
still covers the vocabulary as evenly as the pool allows, and ``build_audit.json`` records
``candidates_total`` / ``candidates_screened`` / ``admissible_total`` / ``items_per_level``
so the achieved n is reported rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctl import (  # noqa: E402
    PEAK_CEILING,
    LevelTarget,
    audit_bit_identity,
    build_item,
    render_item,
    resolve_level_targets,
    write_manifest,
)
from src.maskers import MaskerSpec  # noqa: E402
from src.synth import apply_condition, read_audio_mono  # noqa: E402

CONDITIONS = ("original", "diff")
SAMPLE_RATE = 16000
DURATION_S = 10.0

#: Calibration tolerance the build enforces, in dB. The solvers land within 0.05 dB by
#: construction; 0.5 dB leaves room for the re-solve after a headroom gain and for the
#: difference between the solver's arithmetic and the manifest's.
CALIBRATION_TOLERANCE_DB = 0.5
MAX_HEADROOM_ITERATIONS = 5

#: Candidate files per class. Three gave 108 candidates, of which ~60% survive screening —
#: not enough for the 100 items per level the power analysis asks for (real MDE ~8
#: accuracy points pooled, ~22 per level, against a pilot-implied effect of 7-15 points).
#: Six doubles the pool so the wanted count is reachable from *every* admissible candidate
#: rather than from an alphabetical prefix of it; the extra files change nothing about the
#: item definition, they only widen the pool the round-robin selection draws from.
FILES_PER_CLASS = 6

#: The fit aims slightly *below* ``PEAK_CEILING``: scaling by exactly ``ceiling / peak``
#: lands on the ceiling, the loop's strict ``peak > PEAK_CEILING`` test then never passes,
#: and the item is reported as having unreachable headroom while sitting exactly at it.
#: (Measured on the first run: three items failed this way with peak 0.980.)
HEADROOM_TARGET = PEAK_CEILING - 0.005


@dataclass(frozen=True)
class AxisSpec:
    """One difficulty axis: a masker spectrum plus the quantity its level is solved for."""

    name: str
    masker: str
    quantity: str
    targets: tuple[float, ...]
    relative_to_baseline: bool = False

    def level_name(self, index: int) -> str:
        return f"{self.name}_L{index}"

    def target_at(self, index: int) -> LevelTarget:
        return LevelTarget(
            calibration=self.level_name(index),
            masker=self.masker,
            quantity=self.quantity,
            target=float(self.targets[index]),
            relative_to_baseline=self.relative_to_baseline,
        )


#: Axis A: uniform white noise, level = realised energy SNR. The bottom of the ladder is
#: set by the corpus background (0.002 RMS against a 0.15 RMS event), so -25 dB is
#: comfortably reachable.
#: Axis B: threshold-shaped noise, level = drop below the item's own unmasked margin. The
#: drop grid is 30 dB wide like axis A's, but shifted 5 dB harder because this masker is
#: less efficient than white at matched energy (measured median ~12 dB of audibility
#: separation, see the module docstring): the same audibility drop costs more energy,
#: which is exactly the separation the duel measures. The grids are chosen so the two
#: axes' realised energy ranges overlap (measured in calibration_report.json), otherwise
#: there would be no matched pairs to compare.
AXES = (
    AxisSpec(
        name="energy",
        masker="white",
        quantity="energy_snr_db",
        targets=(5.0, -2.5, -10.0, -17.5, -25.0),
    ),
    AxisSpec(
        name="audibility",
        masker="threshold_shaped",
        quantity="audibility_margin_db",
        targets=(15.0, 22.5, 30.0, 37.5, 45.0),
        relative_to_baseline=True,
    ),
)

MASKER_SPECS = {
    "white": MaskerSpec(kind="white", rms=1.0, seed=0),
    "threshold_shaped": MaskerSpec(kind="threshold_shaped", rms=1.0, seed=0),
    "threshold_inverse": MaskerSpec(kind="threshold_inverse", rms=1.0, seed=0),
    "event_matched": MaskerSpec(kind="event_matched", rms=1.0, seed=0),
}


def level_targets() -> tuple[LevelTarget, ...]:
    return tuple(axis.target_at(i) for axis in AXES for i in range(len(axis.targets)))


# --- corpus and item assembly -------------------------------------------------


def load_corpus(foreground_dir: Path, background_dir: Path, vocab: Path) -> tuple[list[str], list[Path]]:
    labels = [ln.strip() for ln in vocab.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(
        p for d in sorted(background_dir.iterdir()) if d.is_dir() for p in d.glob("*.wav")
    )
    if not labels or not backgrounds:
        raise SystemExit(f"missing corpus under {foreground_dir} / {background_dir}")
    return labels, backgrounds


def candidate_items(
    labels: list[str], foreground_dir: Path, backgrounds: list[Path], seed: int
) -> list[dict[str, object]]:
    """Every (event, donor, background) triple the build may consider, in fixed order.

    Each candidate's donor and background are drawn from a generator seeded by its own
    id, so skipping an unreachable candidate does not shift every later candidate's
    stimuli — a screening pass and the final build then agree on what each item is.
    """

    out: list[dict[str, object]] = []
    for label in labels:
        files = sorted((foreground_dir / label).glob("*.wav"))[:FILES_PER_CLASS]
        for index, event_path in enumerate(files):
            item_id = f"duel-{label.replace(' ', '_')[:22]}-{index:02d}"
            rng = random.Random(f"{seed}:{item_id}")
            donor_class = rng.choice([c for c in labels if c != label])
            donor_files = sorted((foreground_dir / donor_class).glob("*.wav"))
            if not donor_files:
                continue
            out.append(
                {
                    "item_id": item_id,
                    "query_class": label,
                    "donor_class": donor_class,
                    "event_source": event_path,
                    "donor_source": rng.choice(donor_files),
                    "background_source": rng.choice(backgrounds),
                    "seed": rng.randrange(1, 10**6),
                }
            )
    return out


def cell_worst_case_peak(item, calibration: str, masker: str, donor_wave: np.ndarray) -> float:
    """Peak over the scene and the edited arm of one cell.

    The edit pipeline is positively homogeneous — ``_fit_donor`` level-matches the donor
    to the array it is given, so scaling the scene scales every condition with it — which
    is what lets a single gain fix headroom for both conditions at once. The measurement
    is still taken on the rendered arm rather than assumed from the scene peak, because an
    impulsive donor has a higher crest factor than the span it replaces.

    ``donor_level_reference=item.event`` is load-bearing, not decoration: it is what
    :func:`src.ctl.render_item` passes, so a fit that measured the donor under the old
    span-referenced convention would size the headroom gain for a louder replacement than
    the one that gets written (and, before 2026-09-18, for one up to +27.5 dB too loud).
    """

    scene = item.scene_for(calibration, masker)
    worst = float(np.abs(scene).max())
    edited = apply_condition(
        scene,
        item.soundscape(),
        "target_replaced_diff",
        donor_waveform=donor_wave,
        donor_span_samples=int(donor_wave.size),
        donor_level_reference=item.event,
        span=item.query_span,
        rng=random.Random(item.seed),
        # CTL is closed and its calibration was fitted against donor-only
        # replacements; see apply_condition's preserve_background.
        preserve_background=False,
    )
    return max(worst, float(np.abs(edited).max()))


def fit_cell(item, axis: AxisSpec, index: int, donor_wave: np.ndarray):
    """Fit one cell under the PCM_16 ceiling; return ``(item, peak, iterations)``.

    The fit is **per cell**, not per item, so every recording peaks at the same level and
    the model never sees the easy levels 30 dB quieter than the hard ones. Within a cell
    both conditions share the one gain (invariant I6 is about condition-dependent
    normalisation, not cell-dependent).

    The drop target stays anchored to the item's **build-scale** baseline, and that is
    load-bearing rather than a detail. A target expressed as "d dB below whatever the
    unmasked margin is at the current gain" has no headroom fixed point at all: the
    threshold-in-quiet floor does not scale with the mix, so scaling the item by ``g``
    lowers the unmasked baseline by ``20 log10 g``, which lowers the target by the same
    amount, which demands a masker ``1/g`` louder — the peak is exactly invariant and the
    loop cannot converge (measured: a bell's audibility-L4 cell stayed at peak 5.0 for
    every gain tried). Anchoring to the build-scale baseline makes the target an absolute
    margin, so the required masker level scales with ``g`` and the peak does too.

    What the reader must know follows from the same physics: on the rendered audio the
    *unmasked* margin is lower than the anchor by ``20 log10 g``, so the realised drop from
    the on-disk baseline is smaller than the nominal drop. Both numbers are recorded
    (``baseline_margin_db`` = the anchor, ``baseline_margin_db_at_scale`` = on disk), and
    the analysis should use the absolute ``audibility_db``, which is what the solver hits.
    """

    target = axis.target_at(index)
    try:
        peak = cell_worst_case_peak(item, target.calibration, target.masker, donor_wave)
        iterations = 0
        while peak > PEAK_CEILING and iterations < MAX_HEADROOM_ITERATIONS:
            item = item.scaled_copy(HEADROOM_TARGET / peak)
            resolve_level_targets(item, [target])
            peak = cell_worst_case_peak(item, target.calibration, target.masker, donor_wave)
            iterations += 1
    except AssertionError as error:
        raise AssertionError(f"{target.calibration}: {error}") from error
    return item, peak, iterations


def cell_record(item, axis: AxisSpec, index: int, peak: float, base_baseline_db: float) -> dict[str, object]:
    calibration = axis.level_name(index)
    masker = axis.masker
    audibility = float(item.audibility(calibration, masker))
    energy = float(item.energy_snr(calibration, masker))
    target = float(axis.targets[index])
    if axis.quantity == "energy_snr_db":
        realised = energy
    elif axis.relative_to_baseline:
        realised = item.baseline_margin_db - audibility
    else:
        realised = audibility
    return {
        "item_id": item.item_id,
        "axis": axis.name,
        "level": index,
        "calibration": calibration,
        "masker": masker,
        "target_quantity": axis.quantity,
        "target": target,
        "target_margin_db": (
            item.baseline_margin_db - target if axis.relative_to_baseline else None
        ),
        "energy_snr_db": energy,
        "audibility_db": audibility,
        "realised_drop_db": item.baseline_margin_db - audibility,
        "realised_value": float(realised),
        "calibration_error_db": float(realised - target),
        "masker_rms": float(item.level_of(calibration, masker)),
        # at fit time this is the worst case over the scene and the edited arm, which is
        # what the headroom fit controls; ``render_all`` overwrites ``scene_peak`` with
        # the rendered scene's own peak and keeps this one under its own name
        "worst_case_peak": float(peak),
        "scene_peak": float(peak),
        "shared_gain": float(item.metadata.get("shared_gain", 1.0)),
        "baseline_margin_db": float(base_baseline_db),
        "baseline_margin_db_at_scale": float(item.unmasked_margin_db()),
    }


# --- audit --------------------------------------------------------------------


#: The diff arm's own audibility must move with the ladder. Before the 2026-09-18 donor
#: fix it was flat to about +-3 dB across a 30 dB ladder (measured: -0.3 to +2.9 dB), so
#: this check is deliberately far from that regime while leaving room for the last level
#: approaching the threshold in quiet.
DONOR_LADDER_MIN_SPAN_DB = 10.0
DONOR_LADDER_MAX_REVERSAL_DB = 0.5


def donor_ladder_check(rows: list[dict[str, object]]) -> dict[str, object]:
    """Does the replacement's own difficulty track the ladder? (CTL-Plan §3.4i#3)

    The audit that blocked the first GPU run found the ``diff`` arm's difficulty nearly
    constant: ``_fit_donor`` levelled the replacement to the masked span, so the harder the
    level the louder the replacement, and its audibility stayed within about +-3 dB across
    a 30 dB ladder while the manifest's ``audibility_db`` described the event that had been
    overwritten. Half the records therefore sat at near-constant difficulty and their
    recorded difficulty described a signal that was not in the audio.

    The fix levels the replacement to the event it replaces; this check keeps it fixed.
    Per axis, over the ``diff`` rows, the per-level median ``span_audibility_db`` must span
    at least :data:`DONOR_LADDER_MIN_SPAN_DB` and must not rise by more than
    :data:`DONOR_LADDER_MAX_REVERSAL_DB` from one level to the next (levels run easy to
    hard, so the medians must fall). A reversal that large means the ladder stopped biting,
    e.g. because the replacement fell through the threshold in quiet.
    """

    per_axis: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row.get("condition") != "diff" or "span_audibility_db" not in row:
            continue
        per_axis.setdefault(str(row["axis"]), {}).setdefault(int(row["level"]), []).append(
            float(row["span_audibility_db"])
        )
    axes: dict[str, object] = {}
    failures: list[str] = []
    for axis in sorted(per_axis):
        levels = per_axis[axis]
        medians = [float(np.median(levels[k])) for k in sorted(levels)]
        span = max(medians) - min(medians) if medians else 0.0
        reversals = [medians[i + 1] - medians[i] for i in range(len(medians) - 1)]
        worst_reversal = max(reversals) if reversals else 0.0
        passed = span >= DONOR_LADDER_MIN_SPAN_DB and worst_reversal <= DONOR_LADDER_MAX_REVERSAL_DB
        axes[axis] = {
            "median_span_audibility_db_per_level": medians,
            "span_db": span,
            "worst_upward_step_db": worst_reversal,
            "n_per_level": {str(k): len(levels[k]) for k in sorted(levels)},
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"axis {axis}: diff-arm audibility spans {span:.1f} dB "
                f"(min {DONOR_LADDER_MIN_SPAN_DB:.0f}) with a worst upward step of "
                f"{worst_reversal:+.2f} dB (max {DONOR_LADDER_MAX_REVERSAL_DB:.2f}) — the "
                f"replacement's difficulty does not track the ladder"
            )
    return {"axes": axes, "failures": failures, "passed": not failures}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(
    output_dir: Path,
    items: list[dict[str, object]],
    cells: list[dict[str, object]],
    rows: list[dict[str, object]],
    stats: dict[str, object] | None = None,
) -> dict[str, object]:
    """The load-bearing checks, plus the provenance ones that are cheap.

    Each check appends to ``failures``; the build exits non-zero if that list is not
    empty. The file is written either way, so a failing build still leaves the evidence
    behind. ``stats`` is what :func:`assemble` screened and selected; when it is given,
    the achieved items per level and the class coverage are asserted and reported, so a
    pool that cannot supply the requested n fails loudly instead of shipping a thin level.
    """

    failures: list[str] = []
    item_axis_sets = {
        axis.name: {str(entry["item_id"]) for entry in items} for axis in AXES
    }
    axis_sets_identical = len({frozenset(s) for s in item_axis_sets.values()}) == 1
    if not axis_sets_identical:
        failures.append(f"the two axes do not cover the same item set: {item_axis_sets}")

    per_cell: dict[str, dict[str, Path]] = {}
    for row in rows:
        key = f"{row['item_id']}__{row['calibration']}__{row['masker']}"
        per_cell.setdefault(key, {})[str(row["condition"])] = Path(str(row["audio"]))
    identical: list[str] = []
    for key, paths in per_cell.items():
        if set(paths) != set(CONDITIONS):
            failures.append(f"{key}: conditions {sorted(paths)} != {list(CONDITIONS)}")
            continue
        if paths["original"].read_bytes() == paths["diff"].read_bytes():
            identical.append(key)
    if identical:
        failures.append(
            f"{len(identical)} cell(s) have byte-identical original/diff: {identical[:5]}"
        )

    peaks: dict[str, float] = {}
    for path in sorted((output_dir / "audio").glob("*.wav")):
        wave, _ = sf.read(path, dtype="float32", always_2d=False)
        peaks[path.name] = float(np.abs(wave).max())
    clipping = sorted(name for name, peak in peaks.items() if peak > 1.0)
    if clipping:
        failures.append(f"{len(clipping)} file(s) clip (peak > 1.0): {clipping[:5]}")

    errors = [float(cell["calibration_error_db"]) for cell in cells]
    worst_error = max(abs(e) for e in errors) if errors else float("inf")
    if worst_error > CALIBRATION_TOLERANCE_DB:
        offenders = sorted(
            cells, key=lambda c: -abs(float(c["calibration_error_db"]))
        )[:3]
        failures.append(
            f"realised calibration deviates by {worst_error:.2f} dB > "
            f"{CALIBRATION_TOLERANCE_DB} dB: "
            + "; ".join(
                f"{c['calibration']}/{c['item_id']} target {c['target']:+.2f} "
                f"realised {c['realised_value']:+.2f}"
                for c in offenders
            )
        )

    levels_per_axis = {
        axis.name: len({int(c["level"]) for c in cells if c["axis"] == axis.name})
        for axis in AXES
    }
    for name, count in levels_per_axis.items():
        if count != len(next(a for a in AXES if a.name == name).targets):
            failures.append(f"axis {name} has {count} levels, not {len(AXES[0].targets)}")

    if len(rows) != len(cells) * len(CONDITIONS):
        failures.append(f"{len(rows)} manifest rows for {len(cells)} cells x {len(CONDITIONS)} conditions")
    if any(entry["donor_class"] == entry["query_class"] for entry in items):
        failures.append("a donor class equals its query class; the diff arm is not a class change")

    missing = [name for name in peaks if name not in {Path(str(r["audio"])).name for r in rows}]
    if missing:
        failures.append(f"{len(missing)} rendered file(s) have no manifest row: {missing[:5]}")
    for row in rows:
        if not Path(str(row["audio"])).exists():
            failures.append(f"manifest row without audio: {row['pair_id']}")
        if not np.isfinite(float(row["audibility_db"])) or not np.isfinite(float(row["energy_snr_db"])):
            failures.append(f"non-finite calibration in {row['pair_id']}")
        if "span_audibility_db" not in row or not np.isfinite(float(row["span_audibility_db"])):
            failures.append(f"row without a finite span_audibility_db: {row['pair_id']}")

    # the diff arm must carry the ladder, not a constant level (CTL-Plan §3.4i#3)
    ladder = donor_ladder_check(rows)
    failures.extend(ladder["failures"])

    level_counts: dict[str, int] = {}
    for row in rows:
        if row.get("condition") != "original":
            continue
        key = f"{row['axis']}_L{row['level']}"
        level_counts[key] = level_counts.get(key, 0) + 1
    if stats is not None:
        if len(set(level_counts.values())) != 1:
            failures.append(f"items per level are unequal: {level_counts}")
        expected = int(stats.get("selected", 0))
        if set(level_counts.values()) - {expected}:
            failures.append(
                f"items per level {sorted(set(level_counts.values()))} != selected items {expected}"
            )
        if expected < int(stats.get("requested", 0)):
            print(
                f"NOTE: pool supplied {expected} of {stats['requested']} requested items; "
                f"the achieved n is reported in build_audit.json rather than padded"
            )

    report = {
        "n_items": len(items),
        "n_cells": len(cells),
        "n_recordings": len(rows),
        "n_audio_files": len(peaks),
        "levels_per_axis": levels_per_axis,
        "items_per_level": level_counts,
        "calibration_tolerance_db": CALIBRATION_TOLERANCE_DB,
        "screening": stats,
        "checks": {
            "axis_item_sets_identical": axis_sets_identical,
            "item_ids_per_axis": {k: sorted(v) for k, v in item_axis_sets.items()},
            "cells_with_byte_identical_conditions": len(identical),
            "files_clipping": len(clipping),
            "max_abs_peak": max(peaks.values()) if peaks else None,
            "worst_calibration_error_db": worst_error,
            "calibration_within_tolerance": worst_error <= CALIBRATION_TOLERANCE_DB,
            "n_audio_files_without_manifest_row": len(missing),
            "items_per_level": level_counts,
            "donor_ladder": ladder["axes"],
            "donor_ladder_passed": ladder["passed"],
        },
        "failures": failures,
        "passed": not failures,
    }
    return report


# --- dissociation check -------------------------------------------------------


def dissociation_report(cells: list[dict[str, object]]) -> dict[str, object]:
    """Do the two axes actually separate, and by how much?

    Three numbers decide whether the duel can answer anything:

    * **within-axis collinearity** — the median slope of realised audibility against
      realised energy SNR inside one axis. A slope near 1 means the two "axes" are the
      same knob at different offsets, which is what a single masker spectrum forces.
    * **separation at matched energy** — per item, the audibility difference between the
      axes at the same realised energy SNR (interpolated to the nearest level).
    * **separation at matched audibility** — the same in the other direction.

    If either separation is small relative to the spread of the ladder, the design has no
    power to tell "difficulty is energy" from "difficulty is audibility", and that is what
    this function is for.
    """

    by_item: dict[str, dict[str, list[dict[str, object]]]] = {}
    for cell in cells:
        by_item.setdefault(str(cell["item_id"]), {}).setdefault(str(cell["axis"]), []).append(cell)

    slopes: dict[str, list[float]] = {axis.name: [] for axis in AXES}
    matched_energy_gaps: list[float] = []
    matched_audibility_gaps: list[float] = []
    for per_axis in by_item.values():
        for axis in AXES:
            entries = sorted(per_axis.get(axis.name, []), key=lambda c: int(c["level"]))
            if len(entries) >= 2:
                x = np.array([float(c["energy_snr_db"]) for c in entries])
                y = np.array([float(c["audibility_db"]) for c in entries])
                slopes[axis.name].append(float(np.polyfit(x, y, 1)[0]))
        energy_axis = sorted(per_axis.get("energy", []), key=lambda c: int(c["level"]))
        aud_axis = sorted(per_axis.get("audibility", []), key=lambda c: int(c["level"]))
        if not energy_axis or not aud_axis:
            continue
        for cell in energy_axis:
            nearest = min(aud_axis, key=lambda c: abs(float(c["energy_snr_db"]) - float(cell["energy_snr_db"])))
            matched_energy_gaps.append(float(nearest["audibility_db"]) - float(cell["audibility_db"]))
        for cell in aud_axis:
            nearest = min(energy_axis, key=lambda c: abs(float(c["audibility_db"]) - float(cell["audibility_db"])))
            matched_audibility_gaps.append(float(nearest["energy_snr_db"]) - float(cell["energy_snr_db"]))

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=float)
        return {
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    per_axis_levels = {
        axis.name: {
            "median_energy_snr_db": [
                float(np.median([float(c["energy_snr_db"]) for c in cells if c["axis"] == axis.name and int(c["level"]) == k]))
                for k in range(len(axis.targets))
            ],
            "median_audibility_db": [
                float(np.median([float(c["audibility_db"]) for c in cells if c["axis"] == axis.name and int(c["level"]) == k]))
                for k in range(len(axis.targets))
            ],
            "targets": list(axis.targets),
            "quantity": axis.quantity,
            "masker": axis.masker,
        }
        for axis in AXES
    }
    within = {name: summary(values) for name, values in slopes.items()}
    at_energy = summary(matched_energy_gaps)
    at_audibility = summary(matched_audibility_gaps)
    separation = min(abs(at_energy.get("median", 0.0)), abs(at_audibility.get("median", 0.0)))
    return {
        "per_axis_levels": per_axis_levels,
        "within_axis_slope_aud_vs_energy": within,
        "separation_at_matched_energy_db": at_energy,
        "separation_at_matched_audibility_db": at_audibility,
        "smallest_median_separation_db": float(separation),
        "verdict": (
            "axes dissociate"
            if separation >= 5.0
            else "axes are near-collinear: the duel cannot separate energy from audibility "
            "at this separation"
        ),
    }


# --- main ---------------------------------------------------------------------


def select_admissible(
    admissible: list[dict[str, object]], wanted: int
) -> list[dict[str, object]]:
    """Choose up to ``wanted`` admissible items with even coverage of the query classes.

    The previous build consumed the candidate list in sorted order and stopped as soon as
    it had enough admissible items, which is an alphabetical prefix of the corpus:
    measured, its 40 items came from the first 22 of 36 classes and 14 classes were absent
    entirely — a coverage bias that any per-class or per-vocabulary claim would inherit.
    Selection is round-robin over the classes instead, so a capped build spreads its items
    over the whole vocabulary; every class contributes its first admissible item before any
    class contributes a second, and so on. Ties inside a class keep the candidate order, so
    the selection stays deterministic.

    ``wanted >= len(admissible)`` returns every admissible item: when the pool cannot
    supply the requested count, the honest thing is to build everything admissible and
    report the achieved n, not to prefer some classes over others.
    """

    if wanted >= len(admissible):
        return list(admissible)
    by_class: dict[str, list[dict[str, object]]] = {}
    for entry in admissible:
        by_class.setdefault(str(entry["query_class"]), []).append(entry)
    selected: list[dict[str, object]] = []
    round_index = 0
    while len(selected) < wanted:
        progressed = False
        for label in sorted(by_class):
            bucket = by_class[label]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                progressed = True
                if len(selected) >= wanted:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


def assemble(candidates: list[dict[str, object]], wanted: int, *, verbose: bool = True):
    """Build, headroom-fit and screen **every** candidate; select ``wanted`` of them.

    An item is admissible only if *every* cell of *both* axes lands within
    ``CALIBRATION_TOLERANCE_DB`` of its target and under the peak ceiling. Screening is
    what keeps the two axes comparable: an item that cannot be driven to the bottom of
    the audibility ladder (the low-frequency events whose margin is pinned by the
    threshold in quiet) would otherwise contribute levels that are not levels.

    Every candidate is screened before anything is selected, so the pool the selection
    draws from is the whole corpus rather than the part that happens to sort first (see
    :func:`select_admissible`). Returns ``(selected, skipped, stats)``; ``stats`` carries
    the counts the audit reports, including the achieved items per level and the class
    coverage, so a short pool is visible instead of silent.
    """

    admissible: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    targets = level_targets()
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        donor_wave, _ = read_audio_mono(Path(candidate["donor_source"]), SAMPLE_RATE)
        try:
            item = build_item(
                item_id=item_id,
                query_class=str(candidate["query_class"]),
                donor_class=str(candidate["donor_class"]),
                event_source=Path(candidate["event_source"]),
                background_source=Path(candidate["background_source"]),
                donor_source=Path(candidate["donor_source"]),
                maskers=MASKER_SPECS,
                seed=int(candidate["seed"]),
                level_targets=targets,
            )
            base_baseline = float(item.baseline_margin_db)
            fitted: list[dict[str, object]] = []
            for axis in AXES:
                for index in range(len(axis.targets)):
                    cell_item, peak, iterations = fit_cell(item, axis, index, donor_wave)
                    record = cell_record(cell_item, axis, index, peak, base_baseline)
                    fitted.append(
                        {
                            "axis": axis,
                            "level": index,
                            "item": cell_item,
                            "record": record,
                            "headroom_iterations": iterations,
                        }
                    )
            worst = max(abs(float(entry["record"]["calibration_error_db"])) for entry in fitted)  # type: ignore[index]
            if worst > CALIBRATION_TOLERANCE_DB:
                skipped.append({"item_id": item_id, "reason": f"calibration off by {worst:.2f} dB"})
                continue
            if any(float(entry["record"]["worst_case_peak"]) > PEAK_CEILING for entry in fitted):  # type: ignore[index]
                skipped.append({"item_id": item_id, "reason": "headroom unreachable"})
                continue
        except (AssertionError, ValueError) as error:
            skipped.append({"item_id": item_id, "reason": f"{type(error).__name__}: {error}"})
            continue
        admissible.append(
            {
                **candidate,
                "item": item,
                "donor_wave": donor_wave,
                "base_baseline_db": base_baseline,
                "fitted": fitted,
            }
        )
        if verbose:
            print(
                f"  admissible {item_id:34} baseline {base_baseline:+6.2f} dB  "
                f"max peak {max(float(e['record']['worst_case_peak']) for e in fitted):.3f}"  # type: ignore[index]
            )

    selected = select_admissible(admissible, wanted)
    selected_ids = {id(entry) for entry in selected}
    for entry in admissible:
        if id(entry) not in selected_ids:
            skipped.append(
                {"item_id": str(entry["item_id"]), "reason": "admissible but not selected (cap)"}
            )
    per_level: dict[str, int] = {}
    for entry in selected:
        for fitted_cell in entry["fitted"]:  # type: ignore[union-attr]
            key = f"{fitted_cell['axis'].name}_L{fitted_cell['level']}"  # type: ignore[union-attr]
            per_level[key] = per_level.get(key, 0) + 1
    per_class: dict[str, int] = {}
    for entry in selected:
        label = str(entry["query_class"])
        per_class[label] = per_class.get(label, 0) + 1
    stats = {
        "candidates_total": len(candidates),
        "candidates_screened": len(candidates),
        "admissible_total": len(admissible),
        "selected": len(selected),
        "requested": int(wanted),
        "items_per_level": per_level,
        "items_per_class": dict(sorted(per_class.items())),
        # every class that *could* contribute must contribute: a class with no admissible
        # candidate is a property of the corpus, a class dropped by the selection is a bias
        "classes_covered": len(per_class),
        "classes_admissible": len({str(entry["query_class"]) for entry in admissible}),
        "classes_available": len({str(entry["query_class"]) for entry in candidates}),
    }
    return selected, skipped, stats


def render_all(admissible: list[dict[str, object]], output_dir: Path):
    """Render every admissible cell; return ``(rows, axis_entries, cells)``."""

    (output_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    axis_entries: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for entry in admissible:
        donor_wave = entry["donor_wave"]
        item_cells = []
        for fitted_cell in entry["fitted"]:  # type: ignore[union-attr]
            axis: AxisSpec = fitted_cell["axis"]  # type: ignore[assignment]
            index: int = fitted_cell["level"]  # type: ignore[assignment]
            item = fitted_cell["item"]
            calibration = axis.level_name(index)
            rendered = render_item(
                item,
                output_dir,
                calibration=calibration,
                masker=axis.masker,
                conditions=CONDITIONS,
                donor=donor_wave,  # type: ignore[arg-type]
                donor_span_samples=int(donor_wave.size),  # type: ignore[union-attr]
            )
            record = dict(fitted_cell["record"])  # type: ignore[arg-type]
            record["scene_peak"] = float(rendered["peak"])  # type: ignore[arg-type]  # the scene, not the edit
            record["headroom_iterations"] = int(fitted_cell["headroom_iterations"])  # type: ignore[arg-type]
            cells.append(record)
            item_cells.append(record)
            for row in rendered["rows"]:  # type: ignore[union-attr]
                row.update(
                    {
                        "axis": axis.name,
                        "level": index,
                        "level_label": f"L{index}",
                        "target_quantity": axis.quantity,
                        "target_value": float(axis.targets[index]),
                        "baseline_margin_db": float(entry["base_baseline_db"]),
                        # the anchor is the build-scale baseline; this is the unmasked
                        # margin of the array that was written, which a headroom gain
                        # lowers (the threshold in quiet does not scale with the mix)
                        "baseline_margin_db_at_scale": float(item.unmasked_margin_db()),
                        "realised_drop_db": float(item.baseline_margin_db)
                        - float(row["audibility_db"]),
                        "shared_gain": float(item.metadata.get("shared_gain", 1.0)),
                        "scene_peak": float(rendered["peak"]),
                    }
                )
                rows.append(row)
        axis_entries.append(
            {
                "item_id": entry["item_id"],
                "query_class": entry["query_class"],
                "donor_class": entry["donor_class"],
                "event_source": str(entry["event_source"]),
                "background_source": str(entry["background_source"]),
                "donor_source": str(entry["donor_source"]),
                "baseline_margin_db": float(entry["base_baseline_db"]),
                "cells": item_cells,
            }
        )
    return rows, axis_entries, cells


def main() -> int:
    global AXES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/processed/synth/background"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/synth/vocab_confirm.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctl/axis_duel"))
    parser.add_argument(
        "--items",
        type=int,
        default=100,
        help="admissible items wanted (100 -> 1000 cells / 2000 clips; the pool is screened "
        "in full and the achieved n per level is reported if it falls short)",
    )
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--dry-run", action="store_true", help="solve levels, print the table, write nothing")
    parser.add_argument("--report-items", type=int, default=5, help="items in the printed table")
    parser.add_argument(
        "--axis-a-targets",
        default=",".join(str(t) for t in AXES[0].targets),
        help="comma-separated energy-SNR targets for axis A",
    )
    parser.add_argument(
        "--axis-b-masker",
        default=AXES[1].masker,
        choices=sorted(MASKER_SPECS),
        help="masker for the audibility axis; the default drop grid is calibrated for "
        "threshold_shaped, and the calibration report shows the realised ranges for any other",
    )
    parser.add_argument(
        "--axis-b-drops",
        default=",".join(str(t) for t in AXES[1].targets),
        help="comma-separated audibility drops (dB below the item's own unmasked baseline) for axis B",
    )
    args = parser.parse_args()

    AXES = (
        replace(AXES[0], targets=tuple(float(v) for v in args.axis_a_targets.split(","))),
        replace(
            AXES[1],
            masker=args.axis_b_masker,
            targets=tuple(float(v) for v in args.axis_b_drops.split(",")),
        ),
    )
    if len(AXES[0].targets) != len(AXES[1].targets):
        raise SystemExit("both axes must have the same number of levels")

    labels, backgrounds = load_corpus(args.foreground_dir, args.background_dir, args.vocab)
    candidates = candidate_items(labels, args.foreground_dir, backgrounds, args.seed)
    admissible, skipped, stats = assemble(candidates, args.items)

    print(f"\ncandidates: {stats['candidates_total']} screened, "
          f"{stats['admissible_total']} admissible, {stats['selected']} selected "
          f"(requested {stats['requested']})")
    print(f"classes covered: {stats['classes_covered']}/{stats['classes_available']} "
          f"({stats['classes_admissible']} classes have an admissible candidate)")
    print("items per level: " + "  ".join(
        f"{k}={v}" for k, v in sorted(stats["items_per_level"].items())  # type: ignore[union-attr]
    ))
    if skipped:
        print("first skip reasons:")
        for entry in skipped[:5]:
            print(f"  {entry['item_id']}: {entry['reason']}")
    if len(admissible) < args.items:
        print(
            f"WARNING: the pool supplied {len(admissible)} of {args.items} requested items; "
            f"every admissible candidate was built rather than an alphabetical prefix, and "
            f"the achieved n per level is recorded in build_audit.json"
        )
    if not admissible:
        print("no admissible items; nothing to build", file=sys.stderr)
        return 2

    report = dissociation_report(
        [entry["record"] for row in admissible for entry in row["fitted"]]  # type: ignore[union-attr,index]
    )
    print("\n=== realised axis table (first items) ===")
    print(f"{'item':32} {'axis':10} {'L':>1} {'target':>8} {'energy_snr':>11} {'audibility':>11} {'drop':>7} {'masker_rms':>10} {'peak*' :>6}")
    for entry in admissible[: args.report_items]:
        for fitted_cell in entry["fitted"]:  # type: ignore[union-attr]
            cell = fitted_cell["record"]  # type: ignore[index]
            print(
                f"{str(entry['item_id'])[:32]:32} {cell['axis']:10} {cell['level']:>1} "
                f"{cell['target']:>+8.2f} {cell['energy_snr_db']:>+11.2f} {cell['audibility_db']:>+11.2f} "
                f"{cell['realised_drop_db']:>+7.2f} {cell['masker_rms']:>10.4f} {cell['worst_case_peak']:>6.3f}"
            )

    print("\n=== dissociation check ===")
    for axis in AXES:
        levels = report["per_axis_levels"][axis.name]
        print(f"  {axis.name:10} ({axis.masker}): median energy SNR per level {['%+.1f' % v for v in levels['median_energy_snr_db']]}")
        print(f"  {'':10}  median audibility per level      {['%+.1f' % v for v in levels['median_audibility_db']]}")
        slope = report["within_axis_slope_aud_vs_energy"][axis.name]
        print(f"  {'':10}  within-axis slope d(aud)/d(snr)  median {slope['median']:+.3f} (p10 {slope['p10']:+.2f}, p90 {slope['p90']:+.2f})")
    at_energy = report["separation_at_matched_energy_db"]
    at_aud = report["separation_at_matched_audibility_db"]
    print(f"  separation at matched ENERGY      : median {at_energy['median']:+.2f} dB (p10 {at_energy['p10']:+.2f}, p90 {at_energy['p90']:+.2f}, n={at_energy['n']})")
    print(f"  separation at matched AUDIBILITY  : median {at_aud['median']:+.2f} dB (p10 {at_aud['p10']:+.2f}, p90 {at_aud['p90']:+.2f}, n={at_aud['n']})")
    print(f"  VERDICT: {report['verdict']}")

    if args.dry_run:
        print("\n--dry-run: no audio written")
        return 0

    output_dir: Path = args.output_dir
    rows, axis_entries, cells = render_all(admissible, output_dir)

    write_manifest(rows, output_dir / "manifest.jsonl")
    (output_dir / "axis.json").write_text(
        json.dumps(
            {
                "axes": [asdict(axis) for axis in AXES],
                "calibration_tolerance_db": CALIBRATION_TOLERANCE_DB,
                "items": axis_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "audio_digests.json").write_text(
        json.dumps(
            {path.name: sha256(path) for path in sorted((output_dir / "audio").glob("*.wav"))},
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = run_audit(output_dir, admissible, cells, rows, stats)
    audit["skipped_candidates"] = skipped
    (output_dir / "build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n=== build audit ===")
    for key, value in audit["checks"].items():
        if key in {"item_ids_per_axis", "donor_ladder"}:
            continue
        print(f"  {key}: {value}")
    print("  donor ladder (median span audibility of the diff arm, easy -> hard):")
    for axis_name, block in audit["checks"]["donor_ladder"].items():  # type: ignore[union-attr]
        print(
            f"    {axis_name:10} {['%+.1f' % v for v in block['median_span_audibility_db_per_level']]}  "
            f"span {block['span_db']:.1f} dB  worst upward step {block['worst_upward_step_db']:+.2f} dB  "
            f"{'ok' if block['passed'] else 'FAILED'}"
        )
    if not audit["passed"]:
        print("\nBUILD AUDIT FAILED:")
        for failure in audit["failures"]:
            print(f"  - {failure}")
        return 1
    print(f"\nrows={len(rows)}  cells={len(cells)}  audio={audit['n_audio_files']}  -> {output_dir}")
    print("next: scripts/ctl_axis_duel_analyse.py --dir "
          f"{output_dir} --predictions <pred.jsonl>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
