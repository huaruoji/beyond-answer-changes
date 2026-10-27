#!/usr/bin/env python3
"""Emit a compact schema/statistics audit for one dataset parquet shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def compact(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"kind": "bytes", "length": len(value), "prefix_hex": value[:12].hex()}
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value[:10]]
    text = str(value)
    return text if len(text) <= 400 else text[:400] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = pq.ParquetFile(args.parquet)
    table = source.read().slice(0, args.rows)
    report = {
        "path": str(args.parquet.resolve()),
        "num_rows": source.metadata.num_rows,
        "num_row_groups": source.metadata.num_row_groups,
        "schema": str(source.schema_arrow),
        "examples": [compact(row) for row in table.to_pylist()],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

