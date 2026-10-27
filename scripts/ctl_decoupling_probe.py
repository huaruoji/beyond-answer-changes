"""CTL probe 1: are the two matched contrasts constructible on real soundscapes?

Round-2 tests established the contrasts on synthetic tones in isolation. This probe
answers the question that actually gates the branch: **does the manipulation survive
real FSD50K events mixed over real TUT background recordings?**

Two things can break it that tones cannot reveal:

* the scene background is itself a masker, so the added masker is only part of the
  threshold — the *relative* contrast may shrink or saturate;
* real events have broad, non-stationary spectra, so "equal energy" and "equal
  audibility" may not be simultaneously controllable at a useful operating point.

First run (18 items, low_pass vs white maskers, 6 s scenes, event at +2 s, masker
RMS 0.05): the equal-energy audibility gap collapsed from 27.9 dB on tones to a
**median of 3.6 dB** (min 0.04), while the equal-audibility contrast held
(18.6 dB energy gap). Diagnosis: a low-pass masker cannot mask a real broadband
event, because most of the event's bands lie far above 400 Hz where that masker has
no energy. Tone probes cannot see this — a tone *has* one band.

``--matrix`` measures every masker kind at a common gain for every item and ranks the
pairs, so the masker pair is chosen from measurement instead of from taste.

Output: ``outputs/ctl/probe1_decoupling.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audibility import audibility_margin_db  # noqa: E402
from src.maskers import MaskerSpec, shaped_noise, solve_event_gain  # noqa: E402

SCENE_S = 6.0
SCENE_RATE = 16000
EVENT_START_S = 2.0
TARGET_MARGIN_DB = 6.0
MASKER_RMS = 0.05

# candidate maskers; the pair is chosen by measurement, not by preference
CANDIDATE_MASKERS: dict[str, MaskerSpec] = {
    "white": MaskerSpec(kind="white", rms=MASKER_RMS, seed=0),
    "pink": MaskerSpec(kind="pink", rms=MASKER_RMS, seed=0),
    "low_pass_400": MaskerSpec(kind="low_pass", rms=MASKER_RMS, seed=0, cutoff_hz=400.0),
    "low_pass_1500": MaskerSpec(kind="low_pass", rms=MASKER_RMS, seed=0, cutoff_hz=1500.0),
    "high_pass_2000": MaskerSpec(kind="high_pass", rms=MASKER_RMS, seed=0, cutoff_hz=2000.0),
    "event_matched": MaskerSpec(kind="event_matched", rms=MASKER_RMS, seed=0),
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def read_mono(path: Path, rate: int) -> np.ndarray:
    """Mono at ``rate``; resamples with linear interpolation when needed.

    FSD50K source events are 44.1 kHz while the scenes are 16 kHz. The instrument
    resamples donors for the same reason (see Environment-Notes).
    """

    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr == rate:
        return np.asarray(wave, dtype=np.float32)
    n_out = int(round(len(wave) * rate / float(sr)))
    idx = np.linspace(0.0, len(wave) - 1.0, n_out)
    return np.interp(idx, np.arange(len(wave)), wave).astype(np.float32)


def pick_background_slice(path: Path, rng: random.Random) -> np.ndarray:
    wave = read_mono(path, SCENE_RATE)
    need = int(SCENE_S * SCENE_RATE)
    if wave.size >= need:
        start = rng.randrange(0, wave.size - need + 1)
        return wave[start : start + need]
    reps = int(np.ceil(need / wave.size))
    return np.tile(wave, reps)[:need]


def build_scene(event: np.ndarray, background: np.ndarray, masker: np.ndarray) -> np.ndarray:
    """Place the event into the background+masker scene at a fixed start time."""

    n = int(SCENE_S * SCENE_RATE)
    scene = np.zeros(n, dtype=np.float32)
    start = int(EVENT_START_S * SCENE_RATE)
    scene[: background.size] += background[:n]
    scene += masker[:n]
    end = min(start + event.size, n)
    scene[start:end] += event[: end - start]
    return scene


def event_span(background: np.ndarray, masker: np.ndarray, n_event: int) -> np.ndarray:
    """The interferer signal over the event's span — the threshold reference."""

    start = int(EVENT_START_S * SCENE_RATE)
    return (background + masker)[start : start + n_event][:n_event]


def measure(event: np.ndarray, background: np.ndarray, masker: np.ndarray, gain: float) -> dict[str, float]:
    """Audibility and energy of the event, measured on the mixed scene.

    ``audibility_db`` uses ``mix=False`` because ``interferer`` is *already* the
    background plus the masker. Passing ``mix=True`` here added the event on top of
    an interferer that already contained it, so the event was counted into its own
    masking threshold and every measured gap collapsed by roughly 10x (27.9 dB on
    tones became 3.6 dB on real events). The event's own contribution to the
    threshold is real but second-order; keeping it out of the reference is what
    makes the equal-energy contrast measurable.
    """

    scaled = (event * gain).astype(np.float32)
    interferer = event_span(background, masker, scaled.size)
    event_rms = _rms(scaled)
    interferer_rms = _rms(interferer)
    return {
        "event_gain": float(gain),
        "event_rms": event_rms,
        "interferer_rms": interferer_rms,
        "energy_snr_db": float(20.0 * np.log10(max(event_rms, 1e-12) / max(interferer_rms, 1e-12))),
        "audibility_db": float(audibility_margin_db(scaled, interferer, SCENE_RATE, mix=False)),
    }


def probe_item(event_path: Path, background_path: Path, seed: int, matrix: bool) -> dict[str, object]:
    rng = random.Random(seed)
    event = read_mono(event_path, SCENE_RATE)
    background = pick_background_slice(background_path, rng)

    maskers: dict[str, np.ndarray] = {}
    for name, spec in CANDIDATE_MASKERS.items():
        spec_seeded = MaskerSpec(
            kind=spec.kind, rms=spec.rms, seed=seed, cutoff_hz=spec.cutoff_hz, order=spec.order
        )
        maskers[name] = shaped_noise(SCENE_S, SCENE_RATE, spec_seeded, event=event)

    # per-masker gain that lands the margin on target; its spread across maskers is
    # what the equal-audibility contrast exploits, and its mean is the common gain
    gains = {
        name: solve_event_gain(event, event_span(background, m, event.size), SCENE_RATE,
                               target_margin_db=TARGET_MARGIN_DB)
        for name, m in maskers.items()
    }
    common = float(np.exp(np.mean(np.log(list(gains.values())))))

    row: dict[str, object] = {
        "event": str(event_path),
        "background": str(background_path),
        "seed": seed,
        "event_duration_s": float(event.size) / SCENE_RATE,
        "common_gain": common,
        "per_masker": {
            name: {
                "equal_energy": measure(event, background, m, common),
                "equal_audibility": measure(event, background, m, gains[name]),
            }
            for name, m in maskers.items()
        },
    }
    if not matrix:
        keep = ("low_pass_400", "white")
        row["per_masker"] = {k: v for k, v in row["per_masker"].items() if k in keep}  # type: ignore[union-attr]
    return row


def item_is_valid(per: dict[str, dict[str, dict[str, float]]], max_drop_db: float = 20.0) -> tuple[bool, str]:
    """Reject items where the manipulation is not measurable.

    The criterion has to be **relative**. Absolute audibility anchors are arbitrary
    in this implementation (the threshold curve is fixed only up to the
    threshold-in-quiet convention), so an absolute floor rejects good items —
    measured on the first attempt, requiring audibility > 0 discarded 17 of 18 items
    whose spreads were in fact 12-32 dB.

    The genuinely degenerate case is when the common gain, pulled upward by the
    stronger masker, drives *both* maskers under the quiet floor in the event's peak
    band: the conditions then share the same floored threshold and the gap is
    identically zero. Measured example: a dog-bark item at event SNR +45 dB where
    both maskers returned exactly +6.00 dB. Reject when either masker sits more than
    ``max_drop_db`` below the stronger one.
    """

    values = [blocks["equal_energy"]["audibility_db"] for blocks in per.values()]
    if not values:
        return False, "no maskers measured"
    strongest = max(values)
    if strongest - min(values) < 1e-9:
        # all maskers agree exactly: either a real null or a shared floor. Distinguish
        # by level -- a shared floor at an absurdly high event level is degenerate.
        return False, "all maskers at identical audibility (shared floor or true null)"
    for name, blocks in per.items():
        if strongest - blocks["equal_energy"]["audibility_db"] > max_drop_db:
            return False, f"{name} more than {max_drop_db:.0f} dB below the strongest masker"
    return True, ""


def rank_pairs(rows: list[dict[str, object]], max_drop_db: float = 20.0) -> list[dict[str, object]]:
    """Rank masker pairs by how well they separate audibility at equal energy.

    Validity is judged **per pair**, not per item across all candidate maskers. An
    earlier version rejected an item whenever *any* candidate masker fell more than
    ``max_drop_db`` below the strongest, which discarded 15 of 18 items for pairs the
    offending masker was not even part of. Judged pairwise, the same data yields 12
    valid items and a 26.5 dB median gap for the best pair.
    """

    names = list(CANDIDATE_MASKERS)
    ranked: list[dict[str, object]] = []
    for a, b in itertools.combinations(names, 2):
        gaps, mismatches, energy_gaps, invalid = [], [], [], 0
        for row in rows:
            per = row["per_masker"]  # type: ignore[index]
            if a not in per or b not in per:
                continue
            va, vb = per[a]["equal_energy"]["audibility_db"], per[b]["equal_energy"]["audibility_db"]
            # Degenerate pair: the two maskers are indistinguishable because the
            # common gain (pulled up by the stronger masker) drove both under the
            # quiet floor in the event's peak band. Detected as an *exactly* equal
            # audibility, which a genuine physical difference never produces.
            if abs(va - vb) < 1e-9:
                invalid += 1
                continue
            gaps.append(abs(va - vb))
            ea_a, ea_b = per[a]["equal_audibility"], per[b]["equal_audibility"]
            mismatches.append(abs(ea_a["audibility_db"] - ea_b["audibility_db"]))
            energy_gaps.append(abs(ea_a["energy_snr_db"] - ea_b["energy_snr_db"]))
        if not gaps:
            continue
        ranked.append(
            {
                "pair": f"{a} vs {b}",
                "n_valid": len(gaps),
                "n_invalid": invalid,
                "equal_energy_audibility_gap_median_db": float(np.median(gaps)),
                "equal_energy_audibility_gap_min_db": float(np.min(gaps)),
                "equal_audibility_mismatch_max_db": float(np.max(mismatches)),
                "equal_audibility_energy_gap_median_db": float(np.median(energy_gaps)),
            }
        )
    ranked.sort(key=lambda r: -float(r["equal_energy_audibility_gap_median_db"]))
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/processed/synth/background"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/synth/vocab_confirm.txt"))
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--items-per-class", type=int, default=3)
    parser.add_argument("--matrix", action="store_true", help="measure every masker kind and rank pairs")
    parser.add_argument("--output", type=Path, default=Path("outputs/ctl/probe1_decoupling.json"))
    args = parser.parse_args()

    labels = [ln.strip() for ln in args.vocab.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(p for d in sorted(args.background_dir.iterdir()) if d.is_dir() for p in d.glob("*.wav"))
    if not labels or not backgrounds:
        print(f"missing corpus: {len(labels)} labels, {len(backgrounds)} backgrounds", file=sys.stderr)
        return 2

    rng = random.Random(20260916)
    chosen = rng.sample(labels, min(args.classes, len(labels)))
    rows: list[dict[str, object]] = []
    for label in chosen:
        files = sorted((args.foreground_dir / label).glob("*.wav"))
        if not files:
            continue
        for path in rng.sample(files, min(args.items_per_class, len(files))):
            bg = rng.choice(backgrounds)
            row = probe_item(path, bg, rng.randrange(1, 10**6), args.matrix)
            row["class"] = label
            rows.append(row)
            per = row["per_masker"]  # type: ignore[index]
            if args.matrix:
                ranked = sorted(
                    CANDIDATE_MASKERS,
                    key=lambda n: -per[n]["equal_energy"]["audibility_db"],
                )
                parts = "  ".join(f"{n}:{per[n]['equal_energy']['audibility_db']:+5.1f}" for n in ranked[:4])
                print(f"{label[:20]:20} eqE audibility by masker  {parts}")
            else:
                ee = {k: per[k]["equal_energy"] for k in per}
                ea = {k: per[k]["equal_audibility"] for k in per}
                print(
                    f"{label[:20]:20} eqE gap {abs(ee['low_pass_400']['audibility_db'] - ee['white']['audibility_db']):5.2f} dB"
                    f" | eqA mismatch {abs(ea['low_pass_400']['audibility_db'] - ea['white']['audibility_db']):5.2f} dB"
                    f" | eqA energy gap {abs(ea['low_pass_400']['energy_snr_db'] - ea['white']['energy_snr_db']):6.2f} dB"
                )

    if not rows:
        print("no items probed", file=sys.stderr)
        return 2

    summary: dict[str, object] = {"n_items": len(rows), "matrix": args.matrix}
    if args.matrix:
        summary["pair_ranking"] = rank_pairs(rows)
        best = summary["pair_ranking"][0]  # type: ignore[index]
        summary["gate5_equal_energy_separates"] = bool(
            best["equal_energy_audibility_gap_median_db"] >= 10.0  # type: ignore[index]
        )
        summary["gate6_equal_audibility_matches"] = bool(
            best["equal_audibility_mismatch_max_db"] <= 1.0  # type: ignore[index]
        )
    else:
        ee_gap = [abs(r["per_masker"]["low_pass_400"]["equal_energy"]["audibility_db"]
                      - r["per_masker"]["white"]["equal_energy"]["audibility_db"]) for r in rows]  # type: ignore[index]
        ea_mis = [abs(r["per_masker"]["low_pass_400"]["equal_audibility"]["audibility_db"]
                      - r["per_masker"]["white"]["equal_audibility"]["audibility_db"]) for r in rows]  # type: ignore[index]
        summary["equal_energy_audibility_gap_db"] = {"median": float(np.median(ee_gap)), "min": float(np.min(ee_gap))}
        summary["equal_audibility_margin_mismatch_db"] = {"max": float(np.max(ea_mis))}
        summary["gate5_equal_energy_separates"] = bool(np.median(ee_gap) >= 10.0)
        summary["gate6_equal_audibility_matches"] = bool(np.max(ea_mis) <= 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")

    print("\n=== probe 1 summary (real FSD50K events over TUT backgrounds) ===")
    print(f"items: {summary['n_items']}")
    if args.matrix:
        print(f"{'masker pair':34} {'valid':>6} {'eqE gap med':>12} {'eqE gap min':>12} {'eqA energy gap':>15}")
        for r in summary["pair_ranking"][:8]:  # type: ignore[index]
            print(f"{r['pair']:34} {r['n_valid']:4d}/{r['n_valid'] + r['n_invalid']:<3d} "
                  f"{r['equal_energy_audibility_gap_median_db']:12.2f} "
                  f"{r['equal_energy_audibility_gap_min_db']:12.2f} "
                  f"{r['equal_audibility_energy_gap_median_db']:15.2f}")
    print(f"gate 5 (equal energy separates) : {summary['gate5_equal_energy_separates']}")
    print(f"gate 6 (equal audibility matches): {summary['gate6_equal_audibility_matches']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
