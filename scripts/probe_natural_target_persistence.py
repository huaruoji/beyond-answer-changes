#!/usr/bin/env python3
"""Ask whether the queried sound is still present after its evidence was replaced.

The natural transfer came out inverted: models are more accurate on real
recordings yet three times less sensitive to replacing the annotated evidence,
and five flat explanations have been ruled out by measurement. One candidate
remains that the natural route can test directly.

TEMPO ships positive labels only. Replacing the *annotated* occurrence of a sound
does not remove the class from the recording if the same kind of sound also
occurs somewhere unannotated. In that case the model is not failing to update --
it is correctly reporting that the sound is still there, and the low A_diff says
something about the corpus rather than about the model.

This probe asks the question directly, in the same audio, with a yes/no answer:

    Is '<the queried class>' present?   A: yes   B: no

Asked about the original clip it should be mostly "yes" -- that is the baseline
that shows the format works. Asked about the clip whose annotated evidence was
replaced, the yes-rate says which of the two worlds we are in:

* yes-rate stays high  -> the class persists elsewhere; the low A_diff is a
  property of positive-label corpora, and the synthetic result is untouched
* yes-rate collapses   -> the sound really is gone and the model simply does not
  update its answer, which is a finding about the model and a more interesting
  one than anything the synthetic grid can show on its own

    python scripts/probe_natural_target_persistence.py \
        --manifest data/processed/natural.jsonl \
        --audio-root data/processed/natural_audio \
        --output data/processed/natural_persistence.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("original", "target_replaced_diff")

QUESTION = (
    "Listen to the audio. Is '{target}' present in it? "
    "Option A: 'yes'. Option B: 'no'. "
    "Respond with exactly Answer: A or Answer: B."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_item[row["source_pair_id"]][row["condition"]] = row

    items = sorted(by_item)
    if args.limit:
        items = items[: args.limit]

    out: list[dict] = []
    missing = 0
    for item in items:
        conditions = by_item[item]
        if not all(condition in conditions for condition in CONDITIONS):
            continue
        target = str(conditions["original"]["query_class"])
        for condition in CONDITIONS:
            row = conditions[condition]
            audio = args.audio_root / Path(row["audio"]).name
            if not audio.exists():
                missing += 1
                continue
            out.append(
                {
                    "pair_id": f"persist-{item}__{condition}",
                    "source_pair_id": f"persist-{item}",
                    "surface_order": 0,
                    "condition": condition,
                    "intervention_role": "none",
                    "audio": str(audio),
                    "question": QUESTION.format(target=target),
                    # No ground truth exists for a yes/no presence probe: whether
                    # the class persists is exactly what is being measured. "A" is
                    # a placeholder so downstream tooling sees a valid label; the
                    # analysis reads the yes-rate, not accuracy.
                    "ground_truth": "A",
                    "option_events": {"A": "yes", "B": "no"},
                    "evidence": row.get("evidence", []),
                    "query_class": target,
                    "absent_class": row.get("absent_class"),
                    "nuisance_classes": row.get("nuisance_classes", []),
                    "edited_span_s": row.get("edited_span_s"),
                    "split": "persistence",
                    "seed": 0,
                    "n_nuisance": row.get("n_nuisance", 0),
                    "source_audio_key": row.get("source_audio_key"),
                    "probe_of": condition,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(
        json.dumps(
            {
                "items": len({r["source_pair_id"] for r in out}),
                "records": len(out),
                "missing_audio": missing,
                "output": str(args.output),
                "note": "one record per (item, condition); yes/no so order is not counterbalanced",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
