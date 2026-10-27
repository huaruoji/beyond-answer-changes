#!/usr/bin/env python3
"""Batch-evaluate an audio-language model on a paired causal manifest."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Manifest fields that must survive into the prediction file. Analysis scripts
# group by these; silently dropping one collapses stratified analyses (for
# example the difficulty sweep) into a single pool without any error.
PROVENANCE_FIELDS = (
    "source_pair_id",
    "surface_order",
    "option_events",
    "intervened_event",
    "intervened_option",
    "difficulty",
    "noise_snr_db",
    "measured_snr_db",
    # Synthetic-grid provenance. Without these the screening analysis cannot see
    # which class an item queried, and the attribution analysis cannot check the
    # span the generator recorded.
    "query_class",
    "absent_class",
    "nuisance_classes",
    "edited_span_s",
    "query_span_s",
    "query_start_s",
    "snr_db",
    "background_class",
    "split",
    "n_nuisance",
)


def parse_choice(text: str) -> str | None:
    """Extract an A/B(/C/D) answer without treating prose as a label."""
    # Some chat-model decoders return the full system/user prompt followed by
    # the assistant response.  Never parse answer tokens from the echoed user
    # question (e.g. "Answer A or B").
    assistant_sections = re.split(r"(?im)^\s*assistant\s*$", text)
    completion = assistant_sections[-1].strip()
    marked = re.findall(
        r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is\s*)?[:.)-]?\s*([ABCD])\b",
        completion,
    )
    if marked:
        return marked[-1].upper()
    verbose_marker = re.findall(
        r"(?im)^\s*(?:final\s+)?answer[^\n]{0,60}:\s*([ABCD])\s*$",
        completion,
    )
    if verbose_marker:
        return verbose_marker[-1].upper()
    # Accept an uppercase choice at the start of a line ("A", "B.", etc.).
    # Deliberately avoid IGNORECASE here: otherwise the English article "a"
    # inside free-form reasoning is repeatedly mistaken for option A.
    matches = re.findall(r"(?m)^\s*([ABCD])(?:\s*[\).,:-]|\s*$)", completion)
    if matches:
        return matches[-1].upper()
    lowered = completion.casefold()
    if re.search(r"\b(first|earlier|before)\b", lowered):
        return "A"
    if re.search(r"\b(second|later|after)\b", lowered):
        return "B"
    return None


def style_question(question: str, style: str) -> str:
    if style == "direct":
        return question
    if style == "cot":
        return question + " Briefly reason from the audio, then end with exactly one line: Answer: A or Answer: B."
    if style == "timestamp":
        return question + " First state the approximate timestamps you used, then end with exactly one line: Answer: A or Answer: B."
    raise ValueError(style)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=["qwen25-omni-3b", "audio-flamingo-3", "fun-audio-chat-8b"],
        default="qwen25-omni-3b",
    )
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--pair-shard", type=int, default=0, help="zero-based shard index")
    parser.add_argument("--pair-shards", type=int, default=1, help="number of deterministic pair shards")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="conditions to evaluate. Omitted means every condition present in "
        "the manifest: a hardcoded default silently dropped five of the seven "
        "synthetic conditions on a 10,080-record run because their names did not "
        "match the older corpus, and the run still reported success",
    )
    parser.add_argument("--prompt-style", choices=["direct", "cot", "timestamp"], default="direct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.model_family == "qwen25-omni-3b":
        from src.models.qwen25_omni import Qwen25Omni

        model = Qwen25Omni(model_id=str(args.model_path), revision=None, max_new_tokens=args.max_new_tokens)
    elif args.model_family == "fun-audio-chat-8b":
        from src.models.funaudiochat import FunAudioChat

        model = FunAudioChat(model_id=str(args.model_path), revision=None, max_new_tokens=args.max_new_tokens)
    else:
        from src.models.audio_flamingo3 import AudioFlamingo3

        model = AudioFlamingo3(model_id=str(args.model_path), revision=None, max_new_tokens=args.max_new_tokens)

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.conditions is None:
        allowed = {row.get("condition") for row in rows}
        print(
            json.dumps(
                {
                    "conditions_inferred_from_manifest": sorted(
                        str(name) for name in allowed
                    )
                }
            ),
            flush=True,
        )
    else:
        allowed = set(args.conditions)
    rows = [row for row in rows if row.get("condition") in allowed]
    if not rows:
        raise SystemExit(
            f"no rows left after filtering to conditions {sorted(allowed)}"
        )
    pair_order = list(dict.fromkeys(row["pair_id"] for row in rows))
    if args.limit_pairs is not None:
        pair_order = pair_order[: args.limit_pairs]
    if not (0 <= args.pair_shard < args.pair_shards):
        raise ValueError("pair-shard must be in [0, pair-shards)")
    selected_pairs = {
        pair_id for index, pair_id in enumerate(pair_order)
        if index % args.pair_shards == args.pair_shard
    }
    rows = [row for row in rows if row["pair_id"] in selected_pairs]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume safely after an interrupted remote window.
    completed: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                old = json.loads(line)
                completed.add((old.get("pair_id"), old.get("condition")))

    started = time.time()
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode) as handle:
        for index, row in enumerate(rows, start=1):
            key = (row["pair_id"], row["condition"])
            if key in completed:
                continue
            question = style_question(row["question"], args.prompt_style)
            record = {
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "intervention_role": row.get("intervention_role"),
                "audio": row["audio"],
                "question": question,
                "ground_truth": row["ground_truth"],
                "model": model.model_id,
                "model_revision": model.revision,
                "prompt_style": args.prompt_style,
                "seed": args.seed,
                "index": index,
            }
            for provenance_key in PROVENANCE_FIELDS:
                if provenance_key in row:
                    record[provenance_key] = row[provenance_key]
            try:
                prediction = model.generate(_resolve_audio(Path(row["audio"]), args.manifest), question)
                record["prediction"] = prediction
                record["parsed_choice"] = parse_choice(prediction)
                record["correct"] = record["parsed_choice"] == row["ground_truth"]
            except Exception as exc:  # preserve a resumable error record
                record["error"] = f"{type(exc).__name__}: {exc}"
                if args.stop_on_error:
                    raise
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"done": index, "total": len(rows), "pair_id": row["pair_id"], "condition": row["condition"], "correct": record.get("correct"), "elapsed_s": round(time.time() - started, 2)}), flush=True)

    print(json.dumps({"records_requested": len(rows), "records_completed": len(completed), "output": str(args.output.resolve()), "python": platform.python_version()}))


def _resolve_audio(path: Path, manifest: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest.resolve().parents[2] / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


if __name__ == "__main__":
    main()
