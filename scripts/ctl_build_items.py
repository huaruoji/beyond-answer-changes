"""Build CTL items: sample the corpus, render both calibration arms, write a manifest.

Produces, under ``--output-dir``::

    audio/<item_id>__<calibration>__<masker>__<condition>.wav
    manifest.jsonl      one row per recording
    build_audit.json    I7 (rendered == audited), sham no-op, per-cell axis stats

Only four conditions are rendered, not the seven of the mainline protocol: ``original``, ``diff``, ``same``,
``sham``. ``silence``/``nuisance``/``bg`` are not part of the MIG estimand.

Masker pair defaults to the one probe 1b selected by measurement
(``low_pass_400`` vs ``event_matched``), not to a guess.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctl import (  # noqa: E402
    CTL_GRID,
    audit_bit_identity,
    audit_sham,
    build_item,
    render_item,
    write_manifest,
)
from src.maskers import MaskerSpec  # noqa: E402

MASKER_PAIR = ("low_pass_400", "event_matched")


def pick_donor(
    foreground_dir: Path,
    query_class: str,
    rng: random.Random,
    *,
    same_class: bool,
    exclude: set[Path] | None = None,
) -> tuple[Path, str]:
    """A donor from the query class (``same_class=True``) or from another class.

    The first version of this function only ever returned a different-class donor and
    documented that "same-class donors are handled by the caller" — the caller never
    handled them, so the `same` arm was rendered with the different-class donor and the
    two edited conditions came out byte-identical. MIG was then zero by construction.
    Both branches now exist here so the caller cannot forget one.
    """

    classes = sorted(p.name for p in foreground_dir.iterdir() if p.is_dir())
    if same_class:
        candidates = [query_class]
    else:
        if len(classes) < 2:
            raise SystemExit(f"need at least two classes under {foreground_dir}")
        candidates = [c for c in classes if c != query_class]
    rng.shuffle(candidates)
    exclude = exclude or set()
    for klass in candidates:
        files = [f for f in sorted((foreground_dir / klass).glob("*.wav")) if f not in exclude]
        if files:
            return rng.choice(files), klass
    raise SystemExit(
        f"could not find a {'same' if same_class else 'different'}-class donor for {query_class!r}"
    )


def donor_event_span(path: Path, sample_rate: int, rng: random.Random) -> tuple[np.ndarray, int]:
    """The donor's own samples, trimmed to a single event-sized span (invariant I1).

    The harvested FSD50K files are already single events, so the whole file is the
    span; the point of returning the length explicitly is that no code path may treat
    an arbitrary span of a file as an event.
    """

    from src.synth import read_audio_mono

    wave, _ = read_audio_mono(path, sample_rate)
    return wave, int(wave.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/processed/synth/background"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/synth/vocab_confirm.txt"))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--items-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctl/pilot"))
    args = parser.parse_args()

    labels = [ln.strip() for ln in args.vocab.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(
        p for d in sorted(args.background_dir.iterdir()) if d.is_dir() for p in d.glob("*.wav")
    )
    if not labels or not backgrounds:
        print("missing corpus", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    chosen = rng.sample(labels, min(args.classes, len(labels)))
    specs = {
        "low_pass_400": MaskerSpec(kind="low_pass", rms=0.05, seed=0, cutoff_hz=400.0),
        "event_matched": MaskerSpec(kind="event_matched", rms=0.05, seed=0),
    }

    rows: list[dict[str, object]] = []
    audit: dict[str, object] = {"items": [], "masker_pair": list(MASKER_PAIR)}
    for label in chosen:
        files = sorted((args.foreground_dir / label).glob("*.wav"))
        if not files:
            continue
        for index in range(min(args.items_per_class, len(files))):
            event_path = files[index]
            item_id = f"ctl-{label.replace(' ', '_')[:20]}-{index:02d}"
            seed = rng.randrange(1, 10**6)
            background_path = rng.choice(backgrounds)
            donor_path, donor_class = pick_donor(
                args.foreground_dir, label, rng, same_class=False, exclude={event_path}
            )
            same_path, same_class = pick_donor(
                args.foreground_dir, label, rng, same_class=True, exclude={event_path, donor_path}
            )

            item = build_item(
                item_id=item_id,
                query_class=label,
                donor_class=donor_class,
                same_donor_class=same_class,
                event_source=event_path,
                background_source=background_path,
                donor_source=donor_path,
                maskers=specs,
                seed=seed,
            )
            donor_wave, donor_span = donor_event_span(donor_path, item.sample_rate, rng)
            same_wave, same_span = donor_event_span(same_path, item.sample_rate, rng)

            item_audit: dict[str, object] = {
                "item_id": item_id,
                "class": label,
                "baseline_margin_db": item.baseline_margin_db,
                "margin_drop_nominal_db": item.metadata.get("margin_drop_nominal_db"),
                "margin_drop_realised_db": item.metadata.get("margin_drop_realised_db"),
                "target_margin_db": item.metadata.get("target_margin_db"),
                "cells": {},
            }
            for calibration in CTL_GRID["calibration"]:
                for masker in MASKER_PAIR:
                    rendered = render_item(
                        item,
                        args.output_dir,
                        calibration=calibration,
                        masker=masker,
                        donor=donor_wave,
                        donor_span_samples=donor_span,
                        same_donor=same_wave,
                        same_donor_span_samples=same_span,
                    )
                    rows.extend(rendered["rows"])  # type: ignore[arg-type]

                    # the two edited arms MUST differ, and must come from different
                    # classes; this is the assertion whose absence let the identical
                    # diff/same pair ship and drive every MIG to zero
                    diff_b, _ = sf.read(rendered["audio"]["diff"], dtype="float32")  # type: ignore[index]
                    same_b, _ = sf.read(rendered["audio"]["same"], dtype="float32")  # type: ignore[index]
                    if np.array_equal(diff_b, same_b):
                        raise AssertionError(
                            f"{item_id}/{calibration}/{masker}: diff and same are identical, "
                            f"which makes MIG zero by construction"
                        )
                    if donor_class == same_class:
                        raise AssertionError(
                            f"{item_id}: donor class {donor_class!r} equals same-class "
                            f"donor class; the arms are not different classes"
                        )

                    scene = rendered["scene"]
                    i7 = audit_bit_identity(
                        rendered["audio"]["original"], scene, item.sample_rate  # type: ignore[index]
                    )
                    sham, _ = sf.read(rendered["audio"]["sham"], dtype="float32", always_2d=False)  # type: ignore[index]
                    delta = float(i7["max_abs_delta"])
                    sham_check = audit_sham(scene, sham, item.query_span)
                    item_audit["cells"][f"{calibration}__{masker}"] = {  # type: ignore[index]
                        "audibility_db": item.audibility(calibration, masker),
                        "energy_snr_db": item.energy_snr(calibration, masker),
                        "masker_rms": item.level_of(calibration, masker),
                        "i7_max_abs_delta": delta,
                        "i7_within_one_lsb": bool(i7["within_one_lsb"]),
                        "sham_max_abs_delta": sham_check["max_abs_delta"],
                        "peak": rendered["peak"],
                        "diff_same_max_abs_delta": float(
                            np.abs(diff_b - same_b).max()  # type: ignore[operator]
                        ),
                        "donor_class": donor_class,
                        "same_donor_class": same_class,
                    }
                    print(
                        f"{item_id:26} {calibration:16} {masker:14} "
                        f"aud {item.audibility(calibration, masker):+7.2f} dB  "
                        f"snr {item.energy_snr(calibration, masker):+7.2f} dB  "
                        f"I7 delta {delta:.2e}"
                    )
            audit["items"].append(item_audit)  # type: ignore[attr-defined]

    write_manifest(rows, args.output_dir / "manifest.jsonl")
    (args.output_dir / "build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # per-cell axis summary: this is gates 5 and 6 on the built items
    summary: dict[str, object] = {}
    for calibration in CTL_GRID["calibration"]:
        a, b = MASKER_PAIR
        spreads, snr_spreads = [], []
        for entry in audit["items"]:  # type: ignore[union-attr]
            cells = entry["cells"]
            try:
                va = cells[f"{calibration}__{a}"]["audibility_db"]
                vb = cells[f"{calibration}__{b}"]["audibility_db"]
                sa = cells[f"{calibration}__{a}"]["energy_snr_db"]
                sb = cells[f"{calibration}__{b}"]["energy_snr_db"]
            except KeyError:
                continue
            spreads.append(abs(va - vb))
            snr_spreads.append(abs(sa - sb))
        summary[calibration] = {
            "n_items": len(spreads),
            "audibility_gap_median_db": float(np.median(spreads)) if spreads else None,
            "audibility_gap_min_db": float(np.min(spreads)) if spreads else None,
            "energy_gap_median_db": float(np.median(snr_spreads)) if snr_spreads else None,
        }
    summary["gate5_equal_energy_separates"] = bool(
        (summary["equal_energy"]["audibility_gap_median_db"] or 0.0) >= 10.0  # type: ignore[index]
    )
    summary["gate6_equal_audibility_matches"] = bool(
        (summary["equal_audibility"]["audibility_gap_median_db"] or 99.0) <= 1.0  # type: ignore[index]
    )
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nrows={len(rows)}  -> {args.output_dir / 'manifest.jsonl'}")
    for calibration, block in summary.items():
        if isinstance(block, dict):
            print(f"  {calibration:16} n={block['n_items']}  aud gap med {block['audibility_gap_median_db']} dB  "
                  f"energy gap med {block['energy_gap_median_db']} dB")
    print(f"  gate 5 (equal energy separates)  : {summary['gate5_equal_energy_separates']}")
    print(f"  gate 6 (equal audibility matches): {summary['gate6_equal_audibility_matches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
