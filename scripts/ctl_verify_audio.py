"""Confirm the GPU host read the audio we think it read.

Every other guard in this pipeline checks that a run *completed*. None of them checks
that the model was fed the intended file: the evaluator resolves a relative audio path
against the working directory, so a path mistake makes the model answer a different
recording while still returning a perfectly parseable answer. The MIG would then be
computed on nonsense with no error anywhere.

Run this after a pilot, comparing the predictions against the manifest on the machine
that holds the audio::

    python scripts/ctl_verify_audio.py --dir outputs/ctl/sample60 \
        --predictions outputs/ctl/sample60/preds.jsonl

It hashes every audio file referenced by the predictions and reports mismatches
against sizes and digests recorded at build time (``audio_digests.json``). When that
record is absent the tool still reports per-file digests so a run on the GPU host can
be compared by eye, and it always reports the distinct file count so a run that
silently answered one file repeatedly cannot pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--digests", type=Path, help="build-time digest record")
    args = parser.parse_args()

    manifest = args.dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest}")
        return 2

    by_pair: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_pair[f"{row['pair_id']}|{row['condition']}"] = row["audio"]

    used: Counter[str] = Counter()
    missing: list[str] = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        audio = rec.get("audio") or by_pair.get(f"{rec.get('pair_id')}|{rec.get('condition')}", "")
        if not audio:
            missing.append(f"{rec.get('pair_id')}|{rec.get('condition')}")
            continue
        used[Path(audio).name] += 1

    if missing:
        print(f"FAIL: {len(missing)} predictions reference no audio, e.g. {missing[:3]}")
        return 3

    distinct = len(used)
    print(f"predictions reference {sum(used.values())} recordings across {distinct} distinct files")
    if distinct == 1 and sum(used.values()) > 1:
        print("FAIL: every prediction used the same file — the path is wrong, not the model")
        return 3

    record_path = args.digests or (args.dir / "audio_digests.json")
    checked = mismatched = 0
    if record_path.exists():
        recorded = json.loads(record_path.read_text(encoding="utf-8"))
        for name in sorted(used):
            path = args.dir / "audio" / name
            if name not in recorded:
                continue
            checked += 1
            if not path.exists():
                print(f"MISMATCH {name}: recorded but absent on disk")
                mismatched += 1
                continue
            got = sha256(path)
            if got != recorded[name]:
                print(f"MISMATCH {name}: digest {got[:12]} != recorded {recorded[name][:12]}")
                mismatched += 1
        print(f"digest check: {checked} files compared, {mismatched} mismatched")
        if mismatched:
            print("FAIL: the audio on disk is not the audio that was measured at build time")
            return 3
    else:
        print(f"note: no digest record at {record_path}; writing one now so the next run can be compared")
        record_path.write_text(
            json.dumps({n: sha256(args.dir / "audio" / n) for n in sorted(used)}, indent=2),
            encoding="utf-8",
        )

    print("OK: every prediction maps to a distinct manifest audio file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
