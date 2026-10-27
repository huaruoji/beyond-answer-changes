#!/usr/bin/env python3
"""Merge prediction shards, rejecting duplicate pair/condition records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITION_ORDER = {
    name: index
    for index, name in enumerate(
        [
            "original",
            "silence",
            "temporal_swap",
            "evidence_removed",
            "irrelevant_removed",
            "evidence_replaced",
        ]
    )
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    seen: set[tuple[str, str]] = set()
    for shard in args.shards:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["pair_id"]), str(row["condition"]))
            if key in seen:
                raise ValueError(f"duplicate prediction record: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("source_pair_id", row["pair_id"])),
            int(row.get("surface_order", 0)),
            CONDITION_ORDER.get(str(row["condition"]), len(CONDITION_ORDER)),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    print(json.dumps({"shards": len(args.shards), "rows": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
