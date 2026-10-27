"""Measure the replacement's own audibility per ladder level: before vs after the donor fix.

The defect this exists to measure (CTL-Plan §3.4i#3, §3.4j). ``_fit_donor`` was called
with ``reference = out[start:end]`` — the **masked mix** span — so the replacement was
levelled to the masker rather than to the event it replaced. On the axis duel that put the
replacement up to +27.5 dB above the event, and its own audibility was nearly flat across
the 30 dB ladder: the ladder only bit on the ``original`` arm, and the manifest's
``audibility_db`` described a signal that was not in half the audio.

This probe measures that quantity directly, on the same items and level targets the
builder uses, under both conventions:

* ``span``  — the superseded convention (``_fit_donor``'s default reference, still the
  default for the mainline builds and tests);
* ``event`` — the fix: the donor is levelled to the event it replaces, which is what
  ``src.ctl.render_item`` now passes.

It is a *measurement*, not a renderer: items come from :func:`src.ctl.build_item`, the
level targets from :mod:`scripts.ctl_axis_duel_build`, and nothing is written to disk. The
rendered-artifact version of the same number is in ``build_audit.json`` under
``checks.donor_ladder`` (per-level median ``span_audibility_db`` of the ``diff`` arm).

Usage::

    E=$HOME/miniconda3/envs/audiotools
    export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
    .venv/bin/python scripts/ctl_donor_audibility_probe.py --items 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ctl_axis_duel_build as duel  # noqa: E402
from src.audibility import audibility_margin_db  # noqa: E402
from src.ctl import build_item  # noqa: E402
from src.synth import _fit_donor, read_audio_mono  # noqa: E402

RATE = duel.SAMPLE_RATE


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _db(value: float, reference: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12) / max(reference, 1e-12)))


def measure_item(entry: dict, targets) -> list[dict[str, float]]:
    """Both fitting conventions for every cell of one item; no audio is written."""

    item = build_item(
        item_id=str(entry["item_id"]),
        query_class=str(entry["query_class"]),
        donor_class=str(entry["donor_class"]),
        event_source=Path(str(entry["event_source"])),
        background_source=Path(str(entry["background_source"])),
        donor_source=Path(str(entry["donor_source"])),
        maskers=duel.MASKER_SPECS,
        seed=int(entry["seed"]),
        level_targets=targets,
    )
    donor, _ = read_audio_mono(Path(str(entry["donor_source"])), RATE)
    start, end = item.query_span
    event_rms = _rms(item.event)
    out: list[dict[str, float]] = []
    for axis in duel.AXES:
        for index in range(len(axis.targets)):
            calibration = axis.level_name(index)
            masker = axis.masker
            scene = item.scene_for(calibration, masker)
            interferer = item._interferer(calibration, masker)
            span = scene[start:end]
            old = _fit_donor(donor, donor.size, item.event.size, span)
            new = _fit_donor(
                donor, donor.size, item.event.size, span, level_reference=item.event
            )
            out.append(
                {
                    "axis": axis.name,
                    "level": index,
                    "event_audibility_db": float(
                        audibility_margin_db(item.event, interferer, RATE)
                    ),
                    "old_donor_audibility_db": float(
                        audibility_margin_db(old, interferer, RATE)
                    ),
                    "new_donor_audibility_db": float(
                        audibility_margin_db(new, interferer, RATE)
                    ),
                    "old_donor_level_vs_event_db": _db(_rms(old), event_rms),
                    "new_donor_level_vs_event_db": _db(_rms(new), event_rms),
                }
            )
    return out


def summarise(records: list[dict[str, float]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for axis in sorted({str(r["axis"]) for r in records}):
        levels = sorted({int(r["level"]) for r in records if r["axis"] == axis})
        block = {
            "event_audibility_db": [],
            "old_donor_audibility_db": [],
            "new_donor_audibility_db": [],
            "old_donor_level_vs_event_db": [],
            "new_donor_level_vs_event_db": [],
            "n": [],
        }
        for level in levels:
            sel = [r for r in records if r["axis"] == axis and int(r["level"]) == level]
            block["n"].append(len(sel))
            for key in (
                "event_audibility_db",
                "old_donor_audibility_db",
                "new_donor_audibility_db",
                "old_donor_level_vs_event_db",
                "new_donor_level_vs_event_db",
            ):
                block[key].append(float(np.median([float(r[key]) for r in sel])))
        for key in ("old_donor_audibility_db", "new_donor_audibility_db"):
            values = block[key]
            block[f"{key}_span"] = float(max(values) - min(values))  # type: ignore[arg-type]
        out[axis] = block
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=int, default=12)
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/processed/synth/background"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/synth/vocab_confirm.txt"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, help="optional JSON path; nothing is written otherwise")
    args = parser.parse_args()

    labels, backgrounds = duel.load_corpus(args.foreground_dir, args.background_dir, args.vocab)
    candidates = duel.candidate_items(labels, args.foreground_dir, backgrounds, args.seed)
    targets = duel.level_targets()

    records: list[dict[str, float]] = []
    used = 0
    for entry in candidates:
        if used >= args.items:
            break
        try:
            records.extend(measure_item(entry, targets))
        except (AssertionError, ValueError) as error:
            print(f"  skipped {entry['item_id']}: {type(error).__name__}: {error}")
            continue
        used += 1
    if not records:
        print("no measurable items", file=sys.stderr)
        return 2

    report = {"n_items": used, "per_axis": summarise(records)}
    print(f"\ndonor audibility per level, {used} items, median over items "
          f"(easy -> hard); the diff arm's cue is the donor, not the event\n")
    for axis, block in report["per_axis"].items():  # type: ignore[union-attr]
        print(f"  axis {axis} ({duel.AXES[0 if axis == 'energy' else 1].masker})")
        print(f"    {'level':>5} {'n':>4} {'event':>9} {'donor OLD':>10} {'donor NEW':>10} "
              f"{'donor-event OLD':>16} {'donor-event NEW':>16}")
        for i, level in enumerate(range(len(block["n"]))):  # type: ignore[arg-type]
            print(
                f"    {level:>5} {block['n'][i]:>4} {block['event_audibility_db'][i]:>+9.2f} "
                f"{block['old_donor_audibility_db'][i]:>+10.2f} {block['new_donor_audibility_db'][i]:>+10.2f} "
                f"{block['old_donor_level_vs_event_db'][i]:>+16.2f} "
                f"{block['new_donor_level_vs_event_db'][i]:>+16.2f}"
            )
        print(
            f"    span across levels: OLD {block['old_donor_audibility_db_span']:.1f} dB, "
            f"NEW {block['new_donor_audibility_db_span']:.1f} dB"
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
