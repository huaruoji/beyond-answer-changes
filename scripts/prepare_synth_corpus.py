#!/usr/bin/env python3
"""Turn the raw event library and scene recordings into a scaper corpus.

Two conversions happen here.

**Class names become question text.** The event library is organised by FSD50K
leaf id, and the vocabulary file gives ontology names like ``Chirp_and_tweet``
or ``Organ``. Those are not question text. Normalisation is rule-based with an
explicit override table for the names that are ambiguous or ungrammatical on
their own, because a wrong phrase makes an item ill-posed rather than merely
awkward: ``Organ`` alone could mean a body organ, ``Bark`` a tree's, ``Run`` is
not a sound. No language model is involved, per ADR-010, and the mapping is
frozen in ``labels.json`` so the question text can never drift from the label.

**Scene recordings become scaper backgrounds.** TUT ships as a parquet of 10 s
recordings; scaper wants ``background/<scene>/*.wav``. Decoding happens here.

The excluded list is a judgement call and is recorded rather than hidden:
ambient and scene-like classes (wind, subway) and the speech classes are dropped
because they plausibly occur *inside* an urban background recording, which would
make "which of these two is present" ambiguous for reasons that have nothing to
do with the model.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

#: Explicit question text for names that are ambiguous or ungrammatical alone.
LABEL_OVERRIDES = {
    "Chirp_and_tweet": "birds chirping and tweeting",
    "Shatter": "something shattering",
    "Crack": "a cracking sound",
    "Cutlery_and_silverware": "cutlery clinking",
    "Hiss": "a hissing sound",
    "Splash_and_splatter": "splashing liquid",
    "Typewriter": "a typewriter typing",
    "Bass_drum": "a bass drum",
    "Cupboard_open_or_close": "a cupboard opening or closing",
    "Keys_jangling": "keys jangling",
    "Writing": "writing with a pen on paper",
    "Dishes_and_pots_and_pans": "dishes and pots clattering",
    "Burping_and_eructation": "burping",
    "Run": "running footsteps",
    "Ringtone": "a phone ringtone",
    "Fireworks": "fireworks",
    "Knock": "knocking on a door",
    "Meow": "a cat meowing",
    "Microwave_oven": "a microwave oven",
    "Slam": "a door slamming",
    "Packing_tape_and_duct_tape": "packing tape being pulled",
    "Thump_and_thud": "a heavy thumping sound",
    "Scratching_(performance_technique)": "scratching a record",
    "Hi-hat": "a hi-hat cymbal",
    "Squeak": "a squeaking sound",
    "Walk_and_footsteps": "walking footsteps",
    "Rattle_(instrument)": "a rattle instrument",
    "Electric_guitar": "an electric guitar",
    "Hammer": "a hammer striking",
    "Gunshot_and_gunfire": "a gunshot",
    "Bass_guitar": "a bass guitar",
    "Gasp": "gasping",
    "Ratchet_and_pawl": "a ratchet clicking",
    "Crumpling_and_crinkling": "paper crumpling",
    "Clapping": "clapping",
    "Screaming": "screaming",
    "Finger_snapping": "finger snapping",
    "Drawer_open_or_close": "a drawer opening or closing",
    "Bicycle_bell": "a bicycle bell",
    "Yell": "yelling",
    "Whoosh_and_swoosh_and_swish": "a whooshing sound",
    "Tearing": "paper tearing",
    "Harmonica": "a harmonica",
    "Camera": "a camera shutter",
    "Sigh": "sighing",
    "Trumpet": "a trumpet",
    "Cowbell": "a cowbell",
    "Chink_and_clink": "a clinking sound",
    "Skateboard": "a skateboard rolling",
    "Fart": "flatulence",
    "Acoustic_guitar": "an acoustic guitar",
    "Piano": "a piano",
    "Cough": "coughing",
    "Snare_drum": "a snare drum",
    "Sneeze": "sneezing",
    "Bowed_string_instrument": "a bowed string instrument",
    "Zipper_(clothing)": "a zipper",
    "Coin_(dropping)": "a coin dropping",
    "Organ": "a pipe organ",
    "Bark": "a dog barking",
    "Tabla": "tabla drums",
    "Scissors": "scissors cutting",
    "Rattle": "a rattling sound",
    "Harp": "a harp",
    "Tap": "tapping",
    "Female_singing": "a woman singing",
    "Gong": "a gong",
    "Boom": "a booming sound",
    "Male_singing": "a man singing",
    "Tambourine": "a tambourine",
    "Crushing": "something being crushed",
    "Crash_cymbal": "a crash cymbal",
    "Giggle": "giggling",
    "Drum_kit": "a drum kit",
    "Printer": "a printer printing",
    "Speech_synthesizer": "synthesized speech",
    "Doorbell": "a doorbell",
    "Computer_keyboard": "a computer keyboard typing",
    "Accordion": "an accordion",
    "Drill": "a power drill",
    "Gurgling": "gurgling",
    "Drip": "dripping water",
    "Whispering": "whispering",
    "Crackle": "a crackling sound",
}

#: Dropped because they plausibly occur inside an urban background recording,
#: which would make the presence question ambiguous independently of the model.
EXCLUDED = {
    "Wind",
    "Subway_and_metro_and_underground",
    "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking",
}


def normalise(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    text = re.sub(r"\(([^)]*)\)", r"\1", name)
    text = text.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument(
        "--tut-parquet",
        type=Path,
        nargs="+",
        required=True,
        help="one or more TUT shards; each contributes its own scene recordings",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-sample-rate", type=int, default=16000)
    parser.add_argument("--max-backgrounds", type=int, default=0)
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text())
    foreground_root = args.fsd_root / "foreground"

    # Group event files by (split, class id).
    by_split: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(foreground_root.rglob("*.wav")):
        class_id = path.parent.name
        split = path.parent.parent.name
        if class_id.isdigit():
            by_split[split][int(class_id)].append(path)

    labels: list[dict] = []
    for class_id, name in enumerate(vocab):
        record = {
            "class_id": class_id,
            "ontology_name": name,
            "label": normalise(name),
            "excluded": name in EXCLUDED,
            "counts": {},
        }
        for split, classes in by_split.items():
            record["counts"][split] = len(classes.get(class_id, []))
        labels.append(record)

    kept = [record for record in labels if not record["excluded"]]
    print(
        f"classes: {len(labels)} total, {len(kept)} kept, "
        f"{len(labels) - len(kept)} excluded as background-confusable"
    )
    thin = [r["label"] for r in kept if sum(r["counts"].values()) < 30]
    if thin:
        print(f"  note: {len(thin)} kept classes have fewer than 30 events: {thin[:5]}")

    for split, classes in by_split.items():
        target = args.output / f"foreground_{split}"
        target.mkdir(parents=True, exist_ok=True)
        linked = 0
        for record in kept:
            sources = classes.get(record["class_id"], [])
            if not sources:
                continue
            label_dir = target / record["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            for source in sources:
                destination = label_dir / source.name
                if not destination.exists():
                    destination.symlink_to(source.resolve())
                linked += 1
        print(f"  foreground_{split}: {linked} events, {len(list(target.iterdir()))} labels")

    # Measurement views. The pilot grid and the confirmatory split must not share
    # event instances, so pilot reads train+test and confirm reads val. Screening
    # reads everything, because it only asks whether a class can be named at all.
    view_specs = {
        "pilot": ["train", "test"],
        "confirm": ["val"],
    }
    for view, splits in view_specs.items():
        target = args.output / f"foreground_{view}"
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.iterdir():
            for item in stale.iterdir():
                item.unlink()
        linked = 0
        for split in splits:
            for record in kept:
                sources = by_split.get(split, {}).get(record["class_id"], [])
                if not sources:
                    continue
                label_dir = target / record["label"]
                label_dir.mkdir(parents=True, exist_ok=True)
                for source in sources:
                    destination = label_dir / f"{split}_{source.name}"
                    if not destination.exists():
                        destination.symlink_to(source.resolve())
                    linked += 1
        print(f"  foreground_{view} ({'+'.join(splits)}): {linked} events, {len(list(target.iterdir()))} labels")

    # A union view, for vocabulary screening only. Screening asks whether the
    # model can name a class at all, so it should see every class available
    # rather than the subset that happens to sit in the split the grid will use.
    # The pilot grid reads foreground_train and the confirmatory split reads
    # foreground_val, so no event instance is shared between the measurement
    # splits.
    union = args.output / "foreground_all"
    union.mkdir(parents=True, exist_ok=True)
    linked = 0
    for label_dir in sorted(union.iterdir()):
        for stale in label_dir.iterdir():
            stale.unlink()
    for split, classes in by_split.items():
        for record in kept:
            sources = classes.get(record["class_id"], [])
            if not sources:
                continue
            label_dir = union / record["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            for source in sources:
                destination = label_dir / f"{split}_{source.name}"
                if not destination.exists():
                    destination.symlink_to(source.resolve())
                linked += 1
    print(f"  foreground_all: {linked} events, {len(list(union.iterdir()))} labels")

    # Backgrounds: decode the scene recordings to 16 kHz mono wav.
    rows = []
    for parquet in args.tut_parquet:
        rows.extend(pq.read_table(parquet).to_pylist())
    if args.max_backgrounds:
        rows = rows[: args.max_backgrounds]
    background_root = args.output / "background"
    background_root.mkdir(parents=True, exist_ok=True)
    scene_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        payload = row.get("audio") or {}
        data = payload.get("bytes") if isinstance(payload, dict) else None
        if not data:
            continue
        waveform, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if rate != args.background_sample_rate:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(args.background_sample_rate, rate)
            waveform = resample_poly(
                waveform,
                args.background_sample_rate // divisor,
                rate // divisor,
            ).astype(np.float32)
        scene = str(row.get("label", "unknown"))
        scene_dir = background_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            scene_dir / f"{scene}_{index:03d}.wav",
            np.clip(waveform, -1.0, 1.0),
            args.background_sample_rate,
            subtype="PCM_16",
        )
        scene_counts[scene] += 1
    print(f"  backgrounds: {sum(scene_counts.values())} files, scenes {dict(scene_counts)}")

    summary = {
        "fsd_root": str(args.fsd_root),
        "tut_parquets": [str(p) for p in args.tut_parquet],
        "vocab_entries": len(vocab),
        "classes_kept": len(kept),
        "classes_excluded": sorted(EXCLUDED),
        "events_per_split": {split: sum(len(v) for v in classes.values()) for split, classes in by_split.items()},
        "backgrounds": dict(scene_counts),
        "labels": labels,
    }
    (args.output / "corpus.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "vocab_labels.txt").write_text(
        "\n".join(record["label"] for record in kept) + "\n"
    )
    print(f"wrote {args.output / 'corpus.json'} and vocab_labels.txt")


if __name__ == "__main__":
    main()
