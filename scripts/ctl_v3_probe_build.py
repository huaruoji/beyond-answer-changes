"""v3 probe builder: identity vs change-detection, with a magnitude-matched same arm.

Reading (B) says the model only notices *that* the queried span was rewritten, and that
a different-class donor rewrites it "more" than a same-class donor. The main run's
``same`` arm was never magnitude-matched to its ``diff`` arm, so (B) survives it.

This builder therefore renders, for 40 items of the ``equal_energy__low_pass_400`` cell:

    original    exact re-render of the run-2 recording (byte-verified below)
    diff        exact re-render of the run-2 recording (byte-verified below)
    same        same-class donor chosen so that its edit distance (log-spectral distance
                between the original span and the fitted replacement) is >= the diff
                donor's, and as close to it as possible -- the magnitude-matched arm
    same_loud   that same-class donor at +10 dB inside the span
    diff_loud   the diff-class donor at +10 dB inside the span

and asks three questions per (item, arm):

    2afc   the existing two-alternative question (A = queried class, B = donor class)
    presq  Is '<queried class>' present?   (A = yes, B = no)  -- direct presence question
    presa  Is '<donor class>' present?     (A = yes, B = no)  -- control: the yes/no
           readout must be able to say yes to *something*, else "no" is uninformative

The yes/no presence question is encoded with A/B labels only because
``scripts/evaluate_manifest.py::parse_choice`` extracts A-D letters; the question names
exactly one class and is answered yes/no, which is what isolates identity from the
two-alternative format.

Nothing under src/ or scripts/ is modified; the renderer used is the project's own
``src.synth.apply_condition`` via ``src.ctl``'s fit function, so the stimuli differ from
the run-2 ones only through the donor choice and the level of the two `_loud` arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audibility import audibility_margin_db  # noqa: E402
from src.ctl import (  # noqa: E402
    CTL_GRID,
    DEFAULT_CONDITIONS,
    PEAK_CEILING,
    build_item,
    question_for,
)
from src.maskers import MaskerSpec  # noqa: E402
from src.synth import _fit_donor, crossfade_replace, read_audio_mono  # noqa: E402

CELL = ("equal_energy", "low_pass_400")
ARMS = ("original", "diff", "same", "same_loud", "diff_loud")
QUESTION_KINDS = ("2afc", "presq", "presa")
LOUD_DB = 10.0
MAX_SAME_CANDIDATES = 12
FADE_MS = 10.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lsd_db(original: np.ndarray, edited: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    n = min(original.size, edited.size)
    window = np.hanning(n_fft).astype(np.float64)
    diffs = []
    for start in range(0, max(1, n - n_fft + 1), hop):
        a = np.fft.rfft(original[start : start + n_fft] * window)
        b = np.fft.rfft(edited[start : start + n_fft] * window)
        la = 20.0 * np.log10(np.maximum(np.abs(a), 1e-10))
        lb = 20.0 * np.log10(np.maximum(np.abs(b), 1e-10))
        diffs.append(np.sqrt(np.mean((la - lb) ** 2)))
    return float(np.mean(diffs)) if diffs else float("nan")


def presence_question(target: str) -> str:
    """One class, yes/no answer, A/B labels for the existing parser."""

    return (
        f"Listen to the audio. Is '{target}' present in it? "
        "Option A: 'yes'. Option B: 'no'. "
        "Respond with exactly Answer: A or Answer: B."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/ctl/sample60_v2"))
    parser.add_argument("--foreground-dir", type=Path, default=Path("data/processed/synth/foreground_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctl/v3_probe/probe"))
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    calibration, masker = CELL
    source_rows = [
        row
        for row in load_jsonl(args.run_dir / "manifest.jsonl")
        if row["condition"] == "original" and row["calibration"] == calibration and row["masker"] == masker
    ]
    source_rows.sort(key=lambda row: row["item_id"])
    if len(source_rows) < args.items:
        raise SystemExit(f"only {len(source_rows)} items available in {CELL}")
    pick = sorted({int(round(i)) for i in np.linspace(0, len(source_rows) - 1, args.items)})
    if len(pick) != args.items:
        raise SystemExit(f"stride selection produced {len(pick)} items, wanted {args.items}")
    chosen = [source_rows[i] for i in pick]

    specs = {
        "low_pass_400": MaskerSpec(kind="low_pass", rms=0.05, seed=0, cutoff_hz=400.0),
        "event_matched": MaskerSpec(kind="event_matched", rms=0.05, seed=0),
    }

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    audit: list[dict] = []
    rng = random.Random(args.seed)
    fade_samples = int(round(FADE_MS * 16000 / 1000.0))

    for source in chosen:
        item_id = source["item_id"]
        item = build_item(
            item_id=item_id,
            query_class=source["query_class"],
            donor_class=source["absent_class"],
            event_source=Path(source["meta_event_source"]),
            background_source=Path(source["meta_background_source"]),
            donor_source=Path(source["meta_donor_source"]),
            same_donor_class=source["query_class"],
            margin_drop_db=float(source["meta_margin_drop_nominal_db"]),
            maskers=specs,
            seed=int(source["seed"]),
        )
        scene = item.scene_for(calibration, masker)
        peak = float(np.abs(scene).max())
        gain = 1.0
        if peak > PEAK_CEILING:
            gain = PEAK_CEILING / peak
            scene = (scene * gain).astype(np.float32)
        start, end = item.query_span
        reference_span = scene[start:end].copy()

        diff_wave, _ = read_audio_mono(Path(source["meta_donor_source"]), item.sample_rate)
        diff_span_samples = int(diff_wave.size)
        fitted_diff = _fit_donor(diff_wave, diff_span_samples, end - start, reference_span)
        lsd_diff = lsd_db(reference_span.astype(np.float64), fitted_diff.astype(np.float64))

        # magnitude-matched same-class donor: LSD >= the diff donor's, as close as possible
        event_path = Path(source["meta_event_source"])
        donor_path = Path(source["meta_donor_source"])
        candidates = [
            path
            for path in sorted((args.foreground_dir / source["query_class"]).glob("*.wav"))
            if path != event_path and path != donor_path
        ]
        rng.shuffle(candidates)
        candidates = candidates[:MAX_SAME_CANDIDATES]
        if not candidates:
            raise SystemExit(f"no same-class donor candidates for {item_id}")
        scored = []
        for path in candidates:
            wave, _ = read_audio_mono(path, item.sample_rate)
            fitted = _fit_donor(wave, int(wave.size), end - start, reference_span)
            scored.append((lsd_db(reference_span.astype(np.float64), fitted.astype(np.float64)), path, fitted))
        above = [(lsd, path, fitted) for lsd, path, fitted in scored if lsd >= lsd_diff]
        pool = above or scored
        lsd_same, same_path, fitted_same = min(pool, key=lambda entry: (abs(entry[0] - lsd_diff), str(entry[1])))
        same_wave = fitted_same

        # render the byte-identical arms through the project's own condition code path
        rendered: dict[str, Path] = {}
        for arm, donor in (("original", None), ("diff", fitted_diff), ("same", same_wave)):
            if arm == "original":
                out = scene.copy()
            else:
                out = crossfade_replace(scene, start, end, donor, fade_samples)  # type: ignore[arg-type]
            path = audio_dir / f"{item_id}__equal_energy__low_pass_400__{arm}.wav"
            sf.write(path, out.astype(np.float32), item.sample_rate, subtype="PCM_16")
            rendered[arm] = path

        # the two +10 dB arms: amplify the fitted donor inside the span, keep everything
        # else at one shared (headroom) gain so that only the donor/surround ratio moves
        loud_gain = 10.0 ** (LOUD_DB / 20.0)
        for arm, donor in (("same_loud", same_wave), ("diff_loud", fitted_diff)):
            span_loud = donor * loud_gain
            outside = scene.copy()
            outside[start:end] = 0.0
            shared = min(1.0, PEAK_CEILING / max(float(np.abs(outside).max()), float(np.abs(span_loud).max())))
            out = crossfade_replace((scene * shared).astype(np.float32), start, end, (span_loud * shared), fade_samples)
            if float(np.abs(out).max()) > 1.0:
                raise AssertionError(f"{item_id}/{arm}: peak {np.abs(out).max():.3f} > 1.0")
            path = audio_dir / f"{item_id}__equal_energy__low_pass_400__{arm}.wav"
            sf.write(path, out.astype(np.float32), item.sample_rate, subtype="PCM_16")
            rendered[arm] = path

        # byte-identity check against run 2 where the stimulus must be identical
        checks = {}
        for arm in ("original", "diff"):
            want = Path(source["audio"]) if arm == "original" else Path(
                source["audio"].replace("__original.wav", "__diff.wav")
            )
            got_bytes = rendered[arm].read_bytes()
            want_bytes = want.read_bytes()
            checks[arm] = bool(got_bytes == want_bytes)
        if not all(checks.values()):
            raise AssertionError(f"{item_id}: re-render is not byte-identical to run 2: {checks}")

        for arm in ARMS:
            for kind in QUESTION_KINDS:
                if kind == "2afc":
                    question = question_for(item.query_class, item.donor_class)
                    options = {"A": item.query_class, "B": item.donor_class}
                    truth = "B" if arm.startswith("diff") else "A"
                elif kind == "presq":
                    question = presence_question(item.query_class)
                    options = {"A": "yes", "B": "no"}
                    truth = "B" if arm.startswith("diff") else "A"
                else:
                    question = presence_question(item.donor_class)
                    options = {"A": "yes", "B": "no"}
                    truth = "A" if arm.startswith("diff") else "B"
                source_pair_id = f"{item_id}__equal_energy__low_pass_400__{arm}"
                rows.append(
                    {
                        "item_id": item_id,
                        "pair_id": f"{source_pair_id}__{kind}",
                        "source_pair_id": source_pair_id,
                        "condition": arm,
                        "audio": str(rendered[arm]),
                        "question": question,
                        "question_kind": kind,
                        "option_events": options,
                        "surface_order": 0,
                        "intervention_role": arm,
                        "ground_truth": truth,
                        "calibration": calibration,
                        "masker": masker,
                        "audibility_db": item.audibility(calibration, masker),
                        "energy_snr_db": item.energy_snr(calibration, masker),
                        "query_class": item.query_class,
                        "absent_class": item.donor_class,
                        "same_class": item.query_class,
                        "donor_used": (
                            item.query_class if arm in {"same", "same_loud"}
                            else item.donor_class if arm in {"diff", "diff_loud"} else None
                        ),
                        "query_start_s": start / item.sample_rate,
                        "query_end_s": end / item.sample_rate,
                        "sample_rate": item.sample_rate,
                        "seed": item.seed,
                        "split": "ctl",
                        "shared_gain": gain,
                    }
                )

        delta_same = (fitted_same * 0.0)
        audit.append(
            {
                "item_id": item_id,
                "query_class": item.query_class,
                "absent_class": item.donor_class,
                "replication_byte_identical": checks,
                "diff_donor": str(source["meta_donor_source"]),
                "same_donor": str(same_path),
                "n_same_candidates_scored": len(scored),
                "lsd_diff_db": lsd_diff,
                "lsd_same_db": lsd_same,
                "lsd_same_minus_diff_db": lsd_same - lsd_diff,
                "lsd_same_candidates_median_db": float(np.median([s[0] for s in scored])),
                "change_audibility_diff_db": float(
                    audibility_margin_db(
                        (fitted_diff - reference_span).astype(np.float32), reference_span.astype(np.float32), item.sample_rate
                    )
                ),
                "change_audibility_same_db": float(
                    audibility_margin_db(
                        (fitted_same - reference_span).astype(np.float32), reference_span.astype(np.float32), item.sample_rate
                    )
                ),
                "same_arm_is_magnitude_matched": bool(lsd_same >= lsd_diff),
                "peak_scene": peak,
                "shared_gain_loud_arms_applied": True,
                "delta_unused": float(np.abs(delta_same).sum()),
            }
        )
        print(
            f"{item_id:26} lsd diff {lsd_diff:6.2f} same {lsd_same:6.2f} "
            f"({lsd_same - lsd_diff:+5.2f})  matched={lsd_same >= lsd_diff}  donors={len(scored)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    remote_rows = []
    for row in rows:
        remote = dict(row)
        remote["audio"] = f"audio/{Path(row['audio']).name}"
        remote_rows.append(remote)
    (args.output_dir / "manifest.remote.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remote_rows), encoding="utf-8"
    )
    digest = {}
    for path in sorted(audio_dir.glob("*.wav")):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        digest[path.name] = h.hexdigest()
    (args.output_dir / "audio_digests.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    matched = sum(1 for entry in audit if entry["same_arm_is_magnitude_matched"])
    report = {
        "cell": f"{calibration}__{masker}",
        "n_items": len(chosen),
        "n_records": len(rows),
        "arms": list(ARMS),
        "question_kinds": list(QUESTION_KINDS),
        "n_audio_files": len(digest),
        "replication_byte_identical_original": sum(1 for e in audit if e["replication_byte_identical"]["original"]),
        "replication_byte_identical_diff": sum(1 for e in audit if e["replication_byte_identical"]["diff"]),
        "same_arm_magnitude_matched": matched,
        "lsd_same_minus_diff_db_median": float(np.median([e["lsd_same_minus_diff_db"] for e in audit])),
        "items": audit,
    }
    (args.output_dir / "build_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\nrows={len(rows)}  audio={len(digest)}  byte-identical original/diff="
        f"{report['replication_byte_identical_original']}/{report['replication_byte_identical_diff']}  "
        f"same arm magnitude-matched={matched}/{len(audit)}  "
        f"median LSD(same-diff)={report['lsd_same_minus_diff_db_median']:+.2f} dB"
    )
    print(f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
