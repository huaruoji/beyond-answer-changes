"""Stress the audibility axis: can a much stronger masker actually degrade recognition?

The pilot came back saturated. On 240 original-condition items the model recognised the
queried sound at 0.967 (3B) and 0.929 (7B), and accuracy did not fall with audibility:
the 3B held 0.938 even below -25 dB relative to its own unmasked baseline, and 0.875
below -35 dB. The strong masker therefore never created a difficulty gradient, so the
equal-energy contrast had nothing to move along — which is a statement about the axis,
not about the idea.

This builds a ladder far beyond the pilot's operating point (-15 dB drop) and asks the
one question that decides the branch:

* accuracy falls somewhere on the ladder  -> the axis works, it was simply mis-centred;
* accuracy never falls even at -60 dB     -> the model is not relying on the masked
  band for this task, which is a different and equally reportable finding.

Only the three conditions the MIG needs are rendered (original/diff/same); sham is
skipped because the pipeline floor is already established at 1 LSB and 0.000.

**Two repairs (2026-09-18, CTL-Plan §3.4i#1/#3).** The script crashed after the donor fix
because it rendered the ``same`` condition from the different-class donor, which
``render_item`` now refuses; it draws an explicit same-class donor per item. And its
ladder predates the donor-level fix: the replacement used to be levelled to the masked
span, so the numbers in §3.4f describe a diff arm whose own difficulty was nearly
constant. Re-running this script now produces a diff arm that tracks the ladder (see
``scripts/ctl_donor_audibility_probe.py``); the protected ``outputs/ctl/stress`` directory
is retained as evidence of the old behaviour and is not overwritten by this repair.

Usage::

    python scripts/ctl_stress_axis.py --dir outputs/ctl/stress --items 24
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctl import CTLItem, question_for, render_item  # noqa: E402
from src.maskers import (  # noqa: E402
    MaskerSpec,
    normalise_event,
    shaped_noise,
    solve_masker_attenuation_db,
)
from src.synth import apply_condition, read_audio_mono  # noqa: E402

RATE = 16000
DURATION = 10.0
EVENT_START = 2.0
MASKER_KIND = "event_matched"
LADDER_DB = (15.0, 25.0, 40.0, 60.0)  # drop below the item's own unmasked baseline
CONDITIONS = ("original", "diff", "same")
PEAK_CEILING = 0.98
HEADROOM_TARGET = PEAK_CEILING - 0.005  # aim just under it, or the strict test never passes
MAX_HEADROOM_ITERATIONS = 5


def worst_case_peak(item: CTLItem, donor: np.ndarray, same_donor: np.ndarray) -> float:
    """Peak over the scene and both edited arms, with the same donor level as render_item.

    The pre-splice scene peak cannot see the splice, and an impulsive donor has a higher
    crest factor than the span it replaces — measured here, a stress cell whose scene
    peaked under the ceiling reached 1.2053 in the ``diff`` arm (CTL-Plan §3.4i#4).
    ``donor_level_reference=item.event`` is what ``render_item`` passes, so the fit sizes
    the gain for the replacement that actually gets written.
    """

    scene = item.scene_for("equal_energy", MASKER_KIND)
    worst = float(np.abs(scene).max())
    for condition, wave in (("target_replaced_diff", donor), ("target_replaced_same", same_donor)):
        edited = apply_condition(
            scene,
            item.soundscape(),
            condition,
            donor_waveform=wave,
            donor_span_samples=int(wave.size),
            donor_level_reference=item.event,
            span=item.query_span,
            rng=random.Random(item.seed),
            preserve_background=False,
        )
        worst = max(worst, float(np.abs(edited).max()))
    return worst


def fit_headroom(item: CTLItem, donor: np.ndarray, same_donor: np.ndarray) -> tuple[CTLItem, float]:
    """Scale the whole mix by one gain until every arm is under the ceiling.

    One gain on background, masker and event together is the only permitted level
    correction (docs/protocol-invariants.md, §2.3 rule 2); per-component scaling would re-couple the energy
    and audibility axes. Returns the fitted item and its measured worst-case peak.
    """

    peak = worst_case_peak(item, donor, same_donor)
    iterations = 0
    while peak > PEAK_CEILING and iterations < MAX_HEADROOM_ITERATIONS:
        item = item.scaled_copy(HEADROOM_TARGET / peak)
        peak = worst_case_peak(item, donor, same_donor)
        iterations += 1
    return item, peak


def build_ladder_item(
    item_id: str,
    query_class: str,
    donor_class: str,
    event_path: Path,
    background_path: Path,
    donor_path: Path,
    same_donor_path: Path,
    *,
    drop_db: float,
    rng: random.Random,
) -> CTLItem:
    """One item whose single masker is solved to ``drop_db`` below its own baseline.

    ``same_donor_path`` is the **same-class** donor. It is required rather than optional:
    ``render_item`` refuses to render the ``same`` condition without one, because the
    branch's void contrast came from passing the different-class donor to both edited
    arms. That refusal is what broke this script after the donor fix (CTL-Plan §3.4i#1):
    it was written before the guard existed and rendered three conditions from one donor.
    """

    seed = rng.randrange(1, 10**6)
    event = normalise_event(read_audio_mono(event_path, RATE)[0], 0.15)
    background, _ = read_audio_mono(background_path, RATE)
    need = int(DURATION * RATE)
    if background.size >= need:
        offset = rng.randrange(0, background.size - need + 1)
        background = background[offset : offset + need].copy()
    else:
        background = np.tile(background, int(np.ceil(need / background.size)))[:need].copy()

    start = int(EVENT_START * RATE)
    template = shaped_noise(
        DURATION, RATE, MaskerSpec(kind=MASKER_KIND, rms=1.0, seed=seed), event=event
    )
    template_rms = float(np.sqrt(np.mean(template.astype(np.float64) ** 2)))
    unit = (template / template_rms).astype(np.float32)

    from src.audibility import audibility_margin_db

    baseline = float(
        audibility_margin_db(event, background[start : start + event.size][: event.size], RATE)
    )
    _, level = solve_masker_attenuation_db(
        event,
        template,
        RATE,
        reference_rms=1.0,
        target_margin_db=baseline - drop_db,
        max_attenuation_db=70.0,
    )
    donor, _ = read_audio_mono(donor_path, RATE)
    return CTLItem(
        item_id=item_id,
        query_class=query_class,
        donor_class=donor_class,
        same_donor_class=query_class,
        event=event,
        background=background,
        maskers={MASKER_KIND: unit},
        masker_rms={"equal_energy": {MASKER_KIND: level}, "equal_audibility": {MASKER_KIND: level}},
        sample_rate=RATE,
        query_start_sample=start,
        baseline_margin_db=baseline,
        seed=seed,
        metadata={
            "drop_db": drop_db,
            "baseline_margin_db": baseline,
            "event_source": str(event_path),
            "background_source": str(background_path),
            "donor_source": str(donor_path),
            "same_donor_source": str(same_donor_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/ctl/stress"))
    parser.add_argument("--items", type=int, default=24, help="items per ladder rung")
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/processed/synth/background"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/synth/vocab_confirm.txt"))
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    labels = [ln.strip() for ln in args.vocab.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(
        p for d in sorted(args.background_dir.iterdir()) if d.is_dir() for p in d.glob("*.wav")
    )
    rng = random.Random(args.seed)
    rows: list[dict] = []
    axis: list[dict] = []
    for drop in LADDER_DB:
        built = skipped = 0
        for index in range(args.items):
            label = labels[index % len(labels)]
            files = sorted((args.foreground_dir / label).glob("*.wav"))
            if not files:
                skipped += 1
                continue
            event_path = files[rng.randrange(len(files))]
            same_candidates = [f for f in files if f != event_path]
            if not same_candidates:
                skipped += 1
                continue
            same_path = same_candidates[rng.randrange(len(same_candidates))]
            donor_label = labels[(index + 7) % len(labels)]
            donor_files = sorted((args.foreground_dir / donor_label).glob("*.wav"))
            if not donor_files:
                skipped += 1
                continue
            donor_path = donor_files[rng.randrange(len(donor_files))]
            item_id = f"stress-d{int(drop):02d}-{index:03d}"
            try:
                item = build_ladder_item(
                    item_id,
                    label,
                    donor_label,
                    event_path,
                    rng.choice(backgrounds),
                    donor_path,
                    same_path,
                    drop_db=drop,
                    rng=rng,
                )
            except AssertionError:
                skipped += 1
                continue
            donor_wave, _ = read_audio_mono(donor_path, RATE)
            same_donor_wave, _ = read_audio_mono(same_path, RATE)
            item, peak = fit_headroom(item, donor_wave, same_donor_wave)
            out = render_item(
                item,
                args.dir,
                calibration="equal_energy",
                masker=MASKER_KIND,
                conditions=CONDITIONS,
                donor=donor_wave,
                donor_span_samples=donor_wave.size,
                same_donor=same_donor_wave,
                same_donor_span_samples=same_donor_wave.size,
            )
            for row in out["rows"]:  # type: ignore[union-attr]
                row["question"] = question_for(label, donor_label)
                rows.append(row)
            axis.append(
                {
                    "item_id": item_id,
                    "drop_db": drop,
                    "baseline_margin_db": item.baseline_margin_db,
                    # the fitted item, so the recorded value describes the audio on disk
                    "audibility_db": item.audibility("equal_energy", MASKER_KIND),
                    "masker_rms": item.masker_rms["equal_energy"][MASKER_KIND],
                    "shared_gain": float(item.metadata.get("shared_gain", 1.0)),
                    "peak": peak,
                }
            )
            built += 1
        print(f"drop {drop:5.1f} dB: built {built} items ({skipped} skipped)")

    (args.dir / "axis.json").write_text(json.dumps(axis, indent=2), encoding="utf-8")
    from src.ctl import write_manifest

    write_manifest(rows, args.dir / "manifest.jsonl")
    import soundfile as sf

    from src.ctl import audit_bit_identity

    worst = 0.0
    for row in rows:
        if row["condition"] != "original":
            continue
        on_disk, _ = sf.read(row["audio"], dtype="float32")
        worst = max(worst, float(np.abs(on_disk).max()))
    print(f"\nrows={len(rows)}  max |original| = {worst:.4f}  -> {args.dir / 'manifest.jsonl'}")

    # summarise the realised axis
    for drop in LADDER_DB:
        sel = [a for a in axis if a["drop_db"] == drop]
        if not sel:
            continue
        real = [a["baseline_margin_db"] - a["audibility_db"] for a in sel]
        print(f"  requested drop {drop:5.1f} dB -> realised median {np.median(real):5.1f} dB "
              f"(audibility {np.median([a['audibility_db'] for a in sel]):+.1f} dB, "
              f"masker rms {np.median([a['masker_rms'] for a in sel]):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
