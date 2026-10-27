#!/usr/bin/env python3
"""Check whether the natural route's donors sound like the class they are named.

The natural transfer came out opposite to the synthetic grid: models are *more*
accurate on real recordings yet barely move when the annotated evidence is
replaced. One flat explanation would be that the replacement is not a valid
instance of the option it is supposed to present. The synthetic route's donors
come from FSD-MIX-SED, whose clips each carry a single human-validated FSD50K
leaf label. The natural route's donors come from TEMPO, whose event descriptions
are free-text sentences that need not describe what is actually audible.

The test is direct: take each item's donor, find the clip that donor was cut
from, and ask the model the same question about *that* clip. The donor's class is
genuinely present there, so a model that can hear it answers the donor option. If
it cannot, the natural donors are unreliable and the transfer result is about the
donor, not about the models.

    python scripts/check_natural_donors.py \
        --manifest data/processed/natural.jsonl \
        --parquets data/raw/tempo/data/audio_grounding/evaluation/part-0000{0,1}.parquet \
        --audio-dir data/processed/natural_donor_audio \
        --output data/processed/natural_donor.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.feasibility_scan import (  # noqa: E402
    INTERVAL_RE,
    _decode_duration,
    _extract_event,
)
from src.synth import DEFAULT_SAMPLE_RATE, DEFAULT_TARGET_PEAK, read_audio_mono  # noqa: E402

QUESTION = (
    "Listen to the audio. Which of these two sounds is present in it? "
    "Option A: '{a}'. Option B: '{b}'. "
    "Respond with exactly Answer: A or Answer: B."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquets", nargs="+", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    payload: dict[str, bytes] = {}
    occurrences: dict[str, list[str]] = defaultdict(list)
    for parquet in args.parquets:
        import pyarrow.parquet as pq

        for row in pq.read_table(
            parquet, columns=["audio_key", "question", "answer", "audio"]
        ).to_pylist():
            key = row.get("audio_key")
            data = (row.get("audio") or {}).get("bytes")
            if key and data:
                payload.setdefault(key, data)
            intervals = INTERVAL_RE.findall(row.get("answer") or "")
            description = _extract_event(row.get("question") or "")
            if description and len(intervals) == 1:
                occurrences[description.casefold()].append(key)

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    items: dict[str, dict] = {}
    for row in rows:
        if row["condition"] == "original":
            items[row["source_pair_id"]] = row

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    skipped = 0
    if args.max_items:
        items = dict(list(sorted(items.items()))[: args.max_items])
    for item_id, row in sorted(items.items()):
        donor_class = row["absent_class"]
        target_class = row["query_class"]
        home_key = next(
            (key for key in occurrences.get(donor_class, []) if key != row["source_audio_key"]),
            None,
        )
        if home_key is None or home_key not in payload:
            skipped += 1
            continue
        audio_path = args.audio_dir / f"donorhome-{item_id}.wav"
        if not audio_path.exists():
            waveform, rate = read_audio_mono(
                _materialise(payload[home_key], home_key, args.audio_dir), DEFAULT_SAMPLE_RATE
            )
            peak = float(np.abs(waveform).max()) if waveform.size else 0.0
            if peak > 0:
                waveform = (waveform * (DEFAULT_TARGET_PEAK / peak)).astype(np.float32)
            sf.write(audio_path, np.clip(waveform, -1.0, 1.0), rate, subtype="PCM_16")
        for order in (0, 1):
            options = (
                {"A": target_class, "B": donor_class}
                if order == 0
                else {"A": donor_class, "B": target_class}
            )
            # The donor's class is genuinely present in its home clip, so the
            # donor option is the correct answer here.
            out.append(
                {
                    "pair_id": f"donorhome-{item_id}__o{order}",
                    "source_pair_id": f"donorhome-{item_id}",
                    "surface_order": order,
                    "condition": "original",
                    "intervention_role": "none",
                    "audio": str(audio_path),
                    "question": QUESTION.format(a=options["A"], b=options["B"]),
                    "ground_truth": "A" if options["A"] == donor_class else "B",
                    "option_events": options,
                    "evidence": [],
                    "query_class": donor_class,
                    "absent_class": target_class,
                    "nuisance_classes": [],
                    "split": "donor_home",
                    "seed": 0,
                    "n_nuisance": 0,
                    "source_audio_key": home_key,
                    "origin_item": item_id,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(
        json.dumps(
            {
                "items": len(out) // 2,
                "records": len(out),
                "skipped_no_home_clip": skipped,
                "output": str(args.output),
            },
            indent=2,
        )
    )


def _materialise(data: bytes, key: str, directory: Path) -> Path:
    suffix = ".wav" if data[:4] == b"RIFF" else ".mp3"
    path = directory / f"__clip_{Path(key).stem}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


if __name__ == "__main__":
    main()
