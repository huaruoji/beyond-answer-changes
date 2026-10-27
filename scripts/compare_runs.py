"""Side-by-side comparison of two runs on the same grid.

Written for the confirm_v2 re-render: the replacement now keeps the clip's
background instead of overwriting it, so the donor is masked the way the original
event was. The expectation is that the diff arm gets *harder* -- the artifact that
was inflating its accuracy (+16 pt on high-artifact items) is gone -- while the
original and sham arms, whose audio did not change, should be unchanged.

Reports the same quantities the paper uses, grouped by ``pair_id`` so both surface
orders survive.

    python compare_runs.py old.jsonl new.jsonl --label-old "pre-fix" --label-new "post-fix"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ORIG = "original"
DIFF = "target_replaced_diff"
SAME = "target_replaced_same"
SHAM = "sham"
CONDS = ("original", DIFF, SAME, "nuisance_removed", "background_removed", SHAM, "silence")


def load(path: Path, split: str = "confirm") -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        views[row["pair_id"]][row["condition"]] = row
    return dict(views)


def analyse(views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    usable = [
        c for c in views.values()
        if all(k in c for k in (ORIG, DIFF, SAME))
    ]
    n = len(usable)
    out: dict[str, Any] = {"n_views": n}

    def acc(cond: str) -> float:
        rows = [c[cond] for c in usable if cond in c]
        return sum(1 for r in rows if r.get("correct")) / len(rows) if rows else float("nan")

    out["accuracy"] = {c: acc(c) for c in CONDS if any(c in v for v in usable)}

    def changed(c: dict[str, Any], cond: str) -> bool:
        a = c[ORIG].get("parsed_choice")
        b = c[cond].get("parsed_choice")
        return a is not None and b is not None and b != a

    pd = sum(changed(c, DIFF) for c in usable) / n
    ps = sum(changed(c, SAME) for c in usable) / n
    out["P_change_diff"] = pd
    out["P_change_same"] = ps
    out["selectivity"] = pd - ps
    out["sham_change"] = (
        sum(changed(c, SHAM) for c in usable) / n
        if all(SHAM in c for c in usable) else float("nan")
    )

    def opt(row: dict[str, Any], klass: str | None) -> str | None:
        for letter, name in (row.get("option_events") or {}).items():
            if name == klass:
                return letter
        return None

    recs = []
    for c in usable:
        df = c[DIFF]
        recs.append(dict(
            co=bool(c[ORIG].get("correct")),
            cs=bool(c[SAME].get("correct")),
            cd=bool(df.get("correct")),
            csh=bool(c[SHAM].get("correct")) if SHAM in c else False,
            follow=(df.get("parsed_choice") == opt(df, df.get("absent_class"))),
        ))
    for name, cond in (
        ("all", lambda r: True),
        ("E1", lambda r: r["co"]),
        ("E2", lambda r: r["co"] and r["cs"]),
        ("E3", lambda r: r["co"] and r["cs"] and r["csh"]),
    ):
        sub = [r for r in recs if cond(r)]
        out[f"follow_{name}"] = (
            sum(r["follow"] for r in sub) / len(sub) if sub else float("nan")
        )
        out[f"follow_{name}_n"] = len(sub)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--split", default="confirm")
    parser.add_argument("--label-old", default="old")
    parser.add_argument("--label-new", default="new")
    args = parser.parse_args()

    a = analyse(load(args.old, args.split))
    b = analyse(load(args.new, args.split))

    print(f"{'':26s} {args.label_old:>12s} {args.label_new:>12s} {'delta':>10s}")
    print(f"{'views':26s} {a['n_views']:12d} {b['n_views']:12d} {b['n_views']-a['n_views']:10d}")
    print()
    keys = sorted(set(a["accuracy"]) | set(b["accuracy"]))
    for c in keys:
        x, y = a["accuracy"].get(c, float("nan")), b["accuracy"].get(c, float("nan"))
        print(f"  acc {c:22s} {x:12.4f} {y:12.4f} {y-x:+10.4f}")
    print()
    for k in ("P_change_diff", "P_change_same", "selectivity", "sham_change"):
        x, y = a[k], b[k]
        print(f"  {k:24s} {x:12.4f} {y:12.4f} {y-x:+10.4f}")
    print()
    for k in ("follow_all", "follow_E1", "follow_E2", "follow_E3"):
        x, y = a[k], b[k]
        n = b[f"{k}_n"]
        print(f"  {k:24s} {x:12.4f} {y:12.4f} {y-x:+10.4f}   (n={n})")


if __name__ == "__main__":
    main()
