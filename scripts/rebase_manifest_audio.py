#!/usr/bin/env python3
"""Rewrite a manifest's audio paths so it can run on another machine.

Manifests record absolute local paths, which do not exist on the GPU host. This
replaces the directory with a portable prefix and keeps only the basename, so a
run works as long as the audio sits in one directory relative to where the
evaluator is launched.

    python scripts/rebase_manifest_audio.py screen.jsonl \
        --prefix audio --output screen.remote.jsonl

Anchoring on the basename is deliberate: the generator writes one file per
(item, condition), so basenames are unique and the mapping is unambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prefix", required=True, help="e.g. 'audio' or '.'")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-dir", type=Path, help="check every file exists here")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    missing: list[str] = []
    for row in rows:
        name = Path(row["audio"]).name
        row["audio"] = name if args.prefix in ("", ".") else f"{args.prefix.rstrip('/')}/{name}"
        row["audio_source"] = "synthetic"
        if args.verify_dir is not None and not (args.verify_dir / name).exists():
            missing.append(name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    unique = {Path(row["audio"]).name for row in rows}
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "output": str(args.output),
                "records": len(rows),
                "unique_audio": len(unique),
                "missing_audio": len(missing),
                "missing_examples": missing[:5],
            },
            indent=2,
        )
    )
    if missing:
        raise SystemExit(f"{len(missing)} audio files not found under {args.verify_dir}")


if __name__ == "__main__":
    main()
