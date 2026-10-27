#!/usr/bin/env python3
"""Lock the paper's numbers into one frozen, regenerable evidence table.

WHY THIS EXISTS
---------------
The confirm-split results are spread across several analysis JSONs produced by
different scripts. This script reads every source file, records its SHA256, and
emits the paper's tables under explicit names so the numbers cannot be quoted
from memory or mixed across aggregation conventions.

Two aggregation conventions appear, and they are labelled in the output:

  * the **pooled two-order view** (n=1200), which `synth_attribution.py` and,
    since 2026-09-21, `synth_counterfactual_follow.py` both use -- this feeds
    the seven-condition selectivity table and the follow(diff) table;
  * the **pair-mean view**, which averages within a source pair then over pairs
    and is the estimator that removes the surface-order effect.

An earlier revision of this script described the follow table as a "one order per
item (n=600)" view. That was the pre-fix behaviour of
`synth_counterfactual_follow.py`, which silently used one surface order per item
and thereby lost half the data; the defect is fixed and this script now reads the
corrected artifact.

Re-run it after any upstream analysis changes; never hand-edit its output.

USAGE
    python scripts/consolidate_evidence.py
    python scripts/consolidate_evidence.py --output-dir outputs/evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# model key -> display name, family, and the two source files it appears in.
MODELS: dict[str, dict[str, str]] = {
    "r2_qwen3b": {
        "display": "Qwen2.5-Omni-3B",
        "family": "Qwen2.5-Omni",
        "attribution": "outputs/round2/confirm_qwen3b.json",
        "follow": "outputs/round2/r2v2_qwen3b.jsonl",
    },
    "r2_qwen7b": {
        "display": "Qwen2.5-Omni-7B",
        "family": "Qwen2.5-Omni",
        "attribution": "outputs/round2/confirm_qwen7b.json",
        "follow": "outputs/round2/r2v2_qwen7b.jsonl",
    },
    "r2_af3": {
        "display": "Audio Flamingo 3",
        "family": "NVIDIA/UMD",
        "attribution": "outputs/round2/confirm_af3.json",
        "follow": "",
    },
    "fun_audio_chat_8b": {
        "display": "Fun-Audio-Chat-8B",
        "family": "FunAudioLLM",
        "attribution": "outputs/p0_fun_audio_chat/confirm_fun_audio_chat_8b.json",
        "follow": "outputs/p0_fun_audio_chat/fun_audio_chat_8b.jsonl",
    },
}

CONFOUND = "outputs/round2/confirm_confound.json"
CONFOUND_VIEW = "outputs/round2/confirm_confound_view.json"
CLASS_PROFILE = "outputs/round2/confirm_class_profile.json"
FOLLOW_3MODELS = "outputs/p0_fun_audio_chat/counterfactual_follow_3models.json"

CONDITION_KEYS = (
    "original",
    "target_replaced_diff",
    "target_replaced_same",
    "nuisance_removed",
    "background_removed",
    "silence",
    "sham",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_conditions(path: Path, split: str = "confirm") -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        items[row["pair_id"]][row["condition"]] = row
    return items


def sham_change_rate(path: Path) -> dict[str, float]:
    """P(answer changes | no-op rewrite), the editing-pipeline artefact floor.

    Computed on the same pooled two-order views the follow table uses, so the
    sham column and the follow column share one denominator convention.

    The loader groups by ``pair_id``: keying by ``source_pair_id`` would let the
    second surface order overwrite the first, silently halving the sample and
    picking an order arbitrarily. That defect was found and fixed in the analyses
    this script reads; do not reintroduce it here.
    """

    items = load_conditions(path)
    n = changed = 0
    for conds in items.values():
        if "original" not in conds or "sham" not in conds:
            continue
        original = conds["original"].get("parsed_choice")
        if original is None:
            continue
        n += 1
        sham = conds["sham"].get("parsed_choice")
        if sham is not None and sham != original:
            changed += 1
    return {"n": n, "rate": (changed / n) if n else float("nan")}


def attribution_block(model: dict[str, str], sources: dict[str, str]) -> dict[str, Any] | None:
    path = model.get("attribution")
    if not path:
        return None
    data = load_json(ROOT / path)
    metrics = data["metrics"]
    boot = data.get("bootstrap_95", {})
    gates = data.get("gates", {})
    return {
        "source": path,
        "views": data.get("views"),
        "records": data.get("records"),
        "records_dropped_unparsed": data.get("records_dropped_unparsed"),
        "clips": data.get("clips"),
        "original_accuracy": data["accuracy_by_condition"]["original"],
        "accuracy_by_condition": {k: data["accuracy_by_condition"][k] for k in CONDITION_KEYS},
        "A_diff": metrics["A_diff"],
        "A_same": metrics["A_same"],
        "A_nuisance_removed": metrics["A_nuisance_removed"],
        "A_background_removed": metrics["A_background_removed"],
        "A_sham": metrics["A_sham"],
        "A_silence": metrics["A_silence"],
        "selectivity": metrics["selectivity"],
        "selectivity_ci95": boot.get("selectivity"),
        "competence_gate": gates.get("G_competence"),
        "sham_gate": gates.get("G_sham"),
        "verdict": data.get("verdict"),
    }


def follow_block(key: str, model: dict[str, str]) -> dict[str, Any] | None:
    """Read the capability-conditioned follow table from the corrected artifact.

    ``key`` is the table's own model key and is used to look the model up in the
    artifact. It must NOT be re-derived from the prediction filename: the Qwen
    rows run on the v2 grid (``r2v2_qwen3b.jsonl``) while both artifacts key the
    model by identity (``r2_qwen3b``). Deriving the key from the stem made this
    lookup depend on a coincidence between two independent naming choices.

    The artifact schema changed on 2026-09-21: the per-subset numbers moved under
    ``by_aggregation.pooled_both_orders`` (an earlier revision had them at the top
    level and, before that, silently used one surface order per item). The
    ``pair_mean`` block is also carried, because it is the estimator that actually
    removes the surface-order effect.
    """

    path = model.get("follow")
    if not path:
        return None
    data = load_json(ROOT / FOLLOW_3MODELS)
    if key not in data["models"]:
        raise SystemExit(
            f"{FOLLOW_3MODELS} has no model key {key!r}; it has "
            f"{sorted(data['models'])}. Regenerate it with "
            f"`synth_counterfactual_follow.py ... --keys {' '.join(MODELS)}`."
        )
    block = data["models"][key]
    pooled = block["by_aggregation"]["pooled_both_orders"]
    subsets = {
        name: {
            "n": pooled[name]["n"],
            "follow_diff": pooled[name]["follow_diff"],
            "preserve_same": pooled[name]["preserve_same"],
            "accuracy_diff": pooled[name]["accuracy_diff"],
        }
        for name in ("all", "E1_orig_correct", "E2_orig_and_same", "E3_all_three")
    }
    return {
        "source": FOLLOW_3MODELS,
        "predictions": path,
        "aggregation": "pooled_both_orders",
        "n_views": block["n_views"],
        "n_source_pairs": block["n_source_pairs"],
        "original_accuracy": pooled["accuracy_original"],
        "subsets": subsets,
        "pair_mean": {
            name: {
                "n_source_pairs": block["pair_mean"][name]["n_source_pairs"],
                "follow_diff": block["pair_mean"][name]["follow_diff"],
                "preserve_same": block["pair_mean"][name]["preserve_same"],
            }
            for name in ("all", "E1_orig_correct", "E2_orig_and_same", "E3_all_three")
        },
        "raw_selectivity": pooled["raw_change_rate"]["selectivity"],
        "null_single_a": pooled["null_check"]["single_a"],
        "null_per_condition": pooled["null_check"]["per_condition"],
        "independence_diagnostic": pooled["independence_diagnostic"],
        "metric_identity": pooled["metric_identity"],
        "sham_change_rate": pooled["sham_floor"],
    }


def build() -> dict[str, Any]:
    sources = [
        Path(m["attribution"]) for m in MODELS.values() if m.get("attribution")
    ] + [Path(m["follow"]) for m in MODELS.values() if m.get("follow")] + [
        Path(FOLLOW_3MODELS),
        Path(CONFOUND),
        Path(CONFOUND_VIEW),
        Path(CLASS_PROFILE),
    ]

    models: dict[str, Any] = {}
    for key, model in MODELS.items():
        models[key] = {
            "display": model["display"],
            "family": model["family"],
            "attribution": attribution_block(model, {}),
            "follow": follow_block(key, model),
        }

    confound = load_json(ROOT / CONFOUND)
    confound_view = load_json(ROOT / CONFOUND_VIEW)
    class_profile = load_json(ROOT / CLASS_PROFILE)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "confirm",
        "definitions": {
            "two_order_view": (
                "synth_attribution.py counts both surface option orders per item as "
                "separate views (n=1200). Feeds the seven-condition selectivity table "
                "and, since 2026-09-21, the follow(diff) table too."
            ),
            "one_order_view": (
                "single_order_balanced in synth_counterfactual_follow.py: one order "
                "per source pair, exactly balanced. Avoids double-counting the shared "
                "waveform but still samples the order effect once per pair."
            ),
            "pair_mean_view": (
                "Averages within a source pair then over pairs. This is the estimator "
                "that removes the surface-order effect, and the one to prefer."
            ),
            "selectivity": "A_diff - A_same: P(answer changes | class change) - P(answer changes | matched in-place rewrite).",
            "follow_diff": "P(answer = donor's class | different-class replacement). Identically accuracy(diff): the different-class condition's ground truth is the donor class.",
            "null_model": "(2a-1)^2: selectivity a two-alternative model with no evidence use still produces. Reported both with a single a and per-condition; the independence premise does not hold, so it is an upper bound.",
            "E2": "original correct and same-class correct.",
            "sham_change_rate": "P(answer changes | no-op rewrite), the editing-pipeline artefact floor.",
            "confidence_intervals": "Cluster bootstrap resampling source pairs, not views, because the two orders of a pair share one waveform.",
        },
        "sources_sha256": {str(p): sha256(ROOT / p) for p in sources},
        "models": models,
        "confound_matched": {
            "band": confound["band"],
            "items_with_distances": confound["items_with_distances"],
            "models": {
                k: {
                    "A_diff": v["A_diff"],
                    "A_same": v["A_same"],
                    "selectivity": v["selectivity"],
                    "median_distance_diff": v["median_distance_diff"],
                    "median_distance_same": v["median_distance_same"],
                    "matched_items": v["matched_items"],
                    "matched_selectivity": v["matched_selectivity"],
                    "matched_bootstrap_95": v["matched_bootstrap_95"],
                    "by_distance_quintile": v["by_distance_quintile"],
                }
                for k, v in confound["models"].items()
            },
        },
        "confound_matched_view": {
            "aggregation": confound_view["aggregation"],
            "band": confound_view["band"],
            "items_with_distances": confound_view["items_with_distances"],
            "models": {
                k: {
                    "matched_items": v["matched_items"],
                    "matched_trials": v["matched_views"],
                    "matched_selectivity": v["matched_selectivity"],
                    "matched_bootstrap_95": v["matched_bootstrap_95"],
                }
                for k, v in confound_view["models"].items()
            },
        },
        "class_profile": {
            "min_items": class_profile["min_items"],
            "models": {
                k: {
                    "classes": v["classes"],
                    "accuracy_spread": v["accuracy_spread"],
                    "selectivity_spread": v["selectivity_spread"],
                    "correlations_across_classes": v["correlations_across_classes"],
                    "flagged_competent_but_low_selectivity": v[
                        "flagged_competent_but_low_selectivity"
                    ],
                    "largest_negative_selectivity_residuals": v[
                        "largest_negative_selectivity_residuals"
                    ],
                }
                for k, v in class_profile["models"].items()
            },
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(table: dict[str, Any]) -> str:
    m = table["models"]
    order = ["r2_qwen7b", "r2_qwen3b", "r2_af3", "fun_audio_chat_8b"]
    lines: list[str] = []
    lines.append("# Frozen evidence table (confirm split)")
    lines.append("")
    lines.append(f"Generated {table['generated_utc']} from the JSONs whose SHA256 are")
    lines.append("recorded in `evidence_table.json`. Regenerate with")
    lines.append("`python scripts/consolidate_evidence.py`; do not hand-edit.")
    lines.append("")
    lines.append(
        "All rows are the **v2** grid (the replacement preserves the clip's "
        "background). Audio Flamingo 3 is the exception: it fails the competence "
        "gate, its selectivity is not read, and its attribution is the pre-fix run "
        "(informational only)."
    )
    lines.append("")

    lines.append("## Table A -- seven-condition selectivity (two-order view, n=1200)")
    lines.append("")
    lines.append(
        "| Model | family | original acc. | A_diff | A_same | A_nuis | A_bg | A_silence | A_sham | selectivity (95% CI) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for key in order:
        a = m[key]["attribution"]
        if not a:
            continue
        ci = a["selectivity_ci95"]
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--"
        lines.append(
            f"| {m[key]['display']} | {m[key]['family']} | {fmt(a['original_accuracy'])} | "
            f"{fmt(a['A_diff'])} | {fmt(a['A_same'])} | {fmt(a['A_nuisance_removed'])} | "
            f"{fmt(a['A_background_removed'])} | {fmt(a['A_silence'])} | {fmt(a['A_sham'])} | "
            f"**{fmt(a['selectivity'])}** {ci_s} |"
        )
    lines.append("")
    lines.append(
        "Audio Flamingo 3 fails the competence gate (original acc. < 0.75), so its "
        "selectivity is not read."
    )
    lines.append("")

    lines.append("## Table B -- capability-conditioned follow(diff) (pooled two-order view, n=1200)")
    lines.append("")
    lines.append(
        "| Model | original acc. | follow(diff) all | follow(diff) E2 | pair-mean E2 | null share of raw selectivity | sham floor |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in order:
        f = m[key]["follow"]
        if not f:
            continue
        all_ci = f["subsets"]["all"]["follow_diff"]["ci95"]
        e2 = f["subsets"]["E2_orig_and_same"]
        pm_e2 = f["pair_mean"]["E2_orig_and_same"]["follow_diff"]
        null_share = f["null_per_condition"]["fraction_of_observed_explained"]
        lines.append(
            f"| {m[key]['display']} | {fmt(f['original_accuracy'])} | "
            f"{fmt(f['subsets']['all']['follow_diff']['rate'])} "
            f"[{all_ci[0]:.3f}, {all_ci[1]:.3f}] | "
            f"{fmt(e2['follow_diff']['rate'])} (n={e2['n']}) | "
            f"{fmt(pm_e2)} | "
            f"{100 * null_share:.0f}% | "
            f"{fmt(f['sham_change_rate']['P_change_sham'], 4)} |"
        )
    lines.append("")
    lines.append(
        "`preserve(same)` is 1.000 on E2/E3 by construction and is not a finding. "
        "`follow(diff)` is identically `accuracy(diff)` -- the different-class "
        "condition's ground truth is the donor class -- so it is a directional name "
        "for per-condition accuracy, not a second measurement. The null share is "
        "the per-condition parameterisation; its independence premise does not hold, "
        "so it is an upper bound. The pair-mean column is the surface-order-free "
        "estimator and is the one to quote."
    )
    lines.append("")

    lines.append("## Table C -- acoustic-distance-matched contrast")
    lines.append("")
    lines.append("| Model | median d(diff) | median d(same) | matched n | matched selectivity (95% CI) |")
    lines.append("|---|---:|---:|---:|---|")
    for key, c in table["confound_matched"]["models"].items():
        ci = c["matched_bootstrap_95"]
        lines.append(
            f"| {m[key]['display']} | {fmt(c['median_distance_diff'])} | "
            f"{fmt(c['median_distance_same'])} | {c['matched_items']} | "
            f"**{fmt(c['matched_selectivity'])}** [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.append("")

    lines.append("## Table D -- class-level dissociation test")
    lines.append("")
    lines.append("| Model | classes | acc. spread (min/med/max) | acc. vs selectivity | acc. vs A_same |")
    lines.append("|---|---:|---|---:|---:|")
    for key, c in table["class_profile"]["models"].items():
        s = c["accuracy_spread"]
        corr = c["correlations_across_classes"]
        lines.append(
            f"| {m[key]['display']} | {c['classes']} | "
            f"{fmt(s['min'])} / {fmt(s['median'])} / {fmt(s['max'])} | "
            f"{fmt(corr['accuracy_vs_selectivity'])} | {fmt(corr['accuracy_vs_A_same'])} |"
        )
    lines.append("")

    view = table["confound_matched_view"]
    lines.append("## Table E -- acoustic-distance contrast by option-order trial")
    lines.append("")
    lines.append(f"Aggregation: {view['aggregation']}")
    lines.append("")
    lines.append("| Model | matched soundscapes | matched trials | selective accuracy (95% CI) |")
    lines.append("|---|---:|---:|---|")
    for key, values in view["models"].items():
        ci = values["matched_bootstrap_95"]
        lines.append(
            f"| {m[key]['display']} | {values['matched_items']} | "
            f"{values['matched_trials']} | **{fmt(values['matched_selectivity'])}** "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evidence")
    args = parser.parse_args()

    table = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evidence_table.json"
    md_path = args.output_dir / "evidence_table.md"
    json_path.write_text(json.dumps(table, indent=2, default=float), encoding="utf-8")
    md_path.write_text(render_markdown(table), encoding="utf-8")
    print(f"written: {json_path}")
    print(f"written: {md_path}")


if __name__ == "__main__":
    main()
