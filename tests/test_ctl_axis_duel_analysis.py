"""Tests for the axis-duel analysis: accuracy, P(change), dose-response, matched contrast.

The analysis script exists because the duel manifest had no analysis path at all (CTL-Plan
§3.4j), so its numbers were never read by anything. These tests run it on a synthetic
manifest whose answers are chosen by hand, so every reported number can be checked against
arithmetic rather than against the script's own output:

* three items, two axes, two levels, two arms;
* one item always answers A — the degenerate strategy the 0.5 floor exists to expose;
* the other two follow the edit only on the easy cells;
* the two axes carry different realised values, so the matched comparison has to do real
  matching and the sign convention is checked.

The synthetic manifest is deliberately shaped like the real one (``span_audibility_db`` /
``span_energy_snr_db`` per row, ``pair_id`` join to the predictions), because the whole
point of those fields is that a reader can fit accuracy against the cue that is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import ctl_axis_duel_analyse as analyse_mod

#: (axis, level) -> (realised span energy SNR, realised span audibility)
CELLS = {
    ("energy", 0): (5.0, 2.0),
    ("energy", 1): (-5.0, -8.0),
    ("audibility", 0): (1.0, 10.0),
    ("audibility", 1): (-9.0, 0.0),
}
#: which cells the two non-degenerate items follow the edit on. Deliberately *not* the
#: same set on both axes, and deliberately not the easy cells of the energy axis: the
#: matched comparison then has to do real matching and its sign is checkable.
FOLLOWS = {("energy", 0): False, ("energy", 1): False, ("audibility", 0): True, ("audibility", 1): False}
ITEMS = ("item-0", "item-1", "item-2")


def _rows() -> list[dict]:
    rows = []
    for item_id in ITEMS:
        for (axis, level), (energy, audibility) in CELLS.items():
            for condition in ("original", "diff"):
                rows.append(
                    {
                        "item_id": item_id,
                        "pair_id": f"{item_id}__{axis}_L{level}__{axis}__{condition}",
                        "condition": condition,
                        "axis": axis,
                        "level": level,
                        "masker": "white" if axis == "energy" else "threshold_shaped",
                        "target_value": energy if axis == "energy" else audibility,
                        "ground_truth": "A" if condition == "original" else "B",
                        "audibility_db": audibility + 1.0,
                        "energy_snr_db": energy + 1.0,
                        "span_audibility_db": audibility,
                        "span_energy_snr_db": energy,
                    }
                )
    return rows


def _choices() -> dict[str, str]:
    """item-0 always answers A; item-1/2 answer B on diff only where FOLLOWS is set."""

    out = {}
    for item_id in ITEMS:
        for (axis, level), _ in CELLS.items():
            for condition in ("original", "diff"):
                pair_id = f"{item_id}__{axis}_L{level}__{axis}__{condition}"
                if item_id == "item-0":
                    choice = "A"
                elif condition == "original":
                    choice = "A"
                else:
                    choice = "B" if FOLLOWS[(axis, level)] else "A"
                out[pair_id] = choice
    return out


@pytest.fixture()
def artifact(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n", encoding="utf-8"
    )
    predictions = tmp_path / "pred_3b.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"pair_id": pair_id, "parsed_choice": choice})
            for pair_id, choice in _choices().items()
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, predictions


def test_analysis_matches_hand_computed_numbers(artifact) -> None:
    manifest, predictions = artifact
    results = analyse_mod.analyse(manifest, predictions, n_boot=200, seed=1)

    assert results["coverage"]["n_items"] == 3
    assert results["coverage"]["span_fields_present"] is True
    assert results["coverage"]["n_pairs_without_prediction"] == 0

    # the always-A floor is reported unconditionally
    assert results["always_a_floor"]["pooled_accuracy"] == 0.5
    assert results["always_a_floor"]["by_construction"] is True

    # energy L0: nobody follows the edit -> diff 0, pooled 0.5, P(change|diff) 0
    energy_easy = results["per_axis_level"]["energy_L0"]
    assert energy_easy["accuracy"]["original"] == pytest.approx(1.0)
    assert energy_easy["accuracy"]["diff"] == pytest.approx(0.0)
    assert energy_easy["accuracy"]["pooled"] == pytest.approx(0.5)
    assert energy_easy["P_change_given_diff"] == pytest.approx(0.0)
    # audibility L0: item-0 always answers A (wrong on diff), the other two answer B
    # (right) -> diff 2/3, pooled 5/6, P(change|diff) 2/3
    audibility_easy = results["per_axis_level"]["audibility_L0"]
    assert audibility_easy["accuracy"]["original"] == pytest.approx(1.0)
    assert audibility_easy["accuracy"]["diff"] == pytest.approx(2 / 3)
    assert audibility_easy["accuracy"]["pooled"] == pytest.approx(5 / 6)
    assert audibility_easy["P_change_given_diff"] == pytest.approx(2 / 3)
    # the realised values are the span ones, not the event-referenced ones (+1 dB here)
    assert energy_easy["realised"]["original_span_energy_snr_db_median"] == pytest.approx(5.0)
    assert energy_easy["realised"]["original_span_audibility_db_median"] == pytest.approx(2.0)
    assert energy_easy["realised"]["original_event_energy_snr_db_median"] == pytest.approx(6.0)
    assert energy_easy["always_a_floor"] == 0.5

    # dose-response: the energy axis is flat (both levels 0.5 pooled) and the audibility
    # axis moves 5/6 -> 1/2 = 33.3 points pooled, 2/3 = 66.7 points on the diff arm
    energy_response = results["dose_response"]["energy"]
    assert energy_response["accuracy_span_pooled_points"] == pytest.approx(0.0)
    assert energy_response["per_level_accuracy_diff"] == pytest.approx([0.0, 0.0])
    response = results["dose_response"]["audibility"]
    assert response["accuracy_span_pooled_points"] == pytest.approx(100 * (5 / 6 - 0.5))
    assert response["accuracy_span_original_points"] == pytest.approx(0.0)
    assert response["accuracy_span_diff_points"] == pytest.approx(100 * 2 / 3)
    assert response["per_level_accuracy_diff"] == pytest.approx([2 / 3, 0.0])
    assert response["always_a_floor"] == 0.5
    lo, hi = response["accuracy_span_pooled_ci95_points"]
    assert lo <= response["accuracy_span_pooled_points"] <= hi

    # matched comparisons, energy minus audibility. Energy L0 (+5 dB) matches audibility
    # L0 (+1 dB): 0.5 pooled against 5/6 -> -1/3; energy L1 matches audibility L1 and both
    # are 0.5 -> 0. Per item: item-0 is 0.5 everywhere (mean 0), item-1/2 mean -1/6, so the
    # item-mean is -1/9 pooled and -2/9 on the diff arm. The original arm is 1.0 everywhere.
    at_energy = results["matched_comparison"]["at_matched_realised_energy"]
    assert at_energy["owner_axis"] == "energy"
    assert at_energy["n_pairs"] == 6
    assert at_energy["realised_mismatch_median_db"] == pytest.approx(4.0)
    assert at_energy["mean_difference_pooled_points"] == pytest.approx(-100 / 6)
    assert at_energy["mean_difference_diff_points"] == pytest.approx(-100 / 3)
    assert at_energy["mean_difference_original_points"] == pytest.approx(0.0)
    assert at_energy["always_a_floor"] == 0.5
    lo, hi = at_energy["mean_difference_pooled_ci95_points"]
    assert lo <= at_energy["mean_difference_pooled_points"] <= hi
    # the audibility-matched comparison must carry the same sign convention even though
    # the audibility axis owns the matching: energy minus audibility
    at_audibility = results["matched_comparison"]["at_matched_realised_audibility"]
    assert at_audibility["owner_axis"] == "audibility"
    assert at_audibility["realised_mismatch_median_db"] == pytest.approx(5.0)
    assert at_audibility["mean_difference_pooled_points"] == pytest.approx(-100 / 6)
    assert at_audibility["mean_difference_diff_points"] == pytest.approx(-100 / 3)
    assert at_audibility["mean_difference_original_points"] == pytest.approx(0.0)


def test_analysis_uses_the_predictions_prediction_field_as_a_fallback(artifact, tmp_path: Path) -> None:
    manifest, _ = artifact
    predictions = tmp_path / "pred_fallback.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"pair_id": pair_id, "prediction": choice})
            for pair_id, choice in _choices().items()
        )
        + "\n",
        encoding="utf-8",
    )
    results = analyse_mod.analyse(manifest, predictions, n_boot=50, seed=1)
    assert results["coverage"]["n_unparsed"] == 0
    assert results["per_axis_level"]["audibility_L0"]["accuracy"]["diff"] == pytest.approx(2 / 3)


def test_analysis_reports_a_manifest_without_span_fields(tmp_path: Path) -> None:
    """An older manifest must not silently look like it carries the realised span level."""

    manifest = tmp_path / "manifest.jsonl"
    rows = _rows()
    for row in rows:
        row.pop("span_audibility_db")
        row.pop("span_energy_snr_db")
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"pair_id": pair_id, "parsed_choice": choice})
            for pair_id, choice in _choices().items()
        )
        + "\n",
        encoding="utf-8",
    )
    results = analyse_mod.analyse(manifest, predictions, n_boot=20, seed=1)
    assert results["coverage"]["span_fields_present"] is False
    assert "predates the span fields" in results["coverage"]["note"]
    # it falls back to the event-referenced values rather than crashing
    assert results["per_axis_level"]["energy_L0"]["realised"][
        "original_span_energy_snr_db_median"
    ] == pytest.approx(6.0)


def test_analysis_cli_writes_json_next_to_the_manifest(artifact, capsys, monkeypatch) -> None:
    manifest, predictions = artifact
    monkeypatch.setattr(
        "sys.argv",
        [
            "ctl_axis_duel_analyse.py",
            "--dir",
            str(manifest.parent),
            "--predictions",
            str(predictions),
            "--bootstrap",
            "50",
        ],
    )
    assert analyse_mod.main() == 0
    output = manifest.parent / f"duel_analysis_{predictions.stem}.json"
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coverage"]["n_items"] == 3
    assert payload["always_a_floor"]["pooled_accuracy"] == 0.5
    # each axis is fitted against the realised value of its own calibrated quantity
    assert payload["dose_response"]["energy"]["dose_response_field"] == "span_energy_snr_db"
    assert payload["dose_response"]["audibility"]["dose_response_field"] == "span_audibility_db"
    printed = capsys.readouterr().out
    assert "ALWAYS-A FLOOR" in printed


# --- the committed artifact, with predictions whose answers are known ---------

ARTIFACT = Path("outputs/ctl/axis_duel")


@pytest.mark.skipif(
    not (ARTIFACT / "manifest.jsonl").exists(), reason="axis_duel build not present"
)
@pytest.mark.parametrize("strategy", ["always_a", "oracle"])
def test_analysis_on_the_committed_artifact(tmp_path: Path, strategy: str) -> None:
    """Read the real manifest with a synthetic model whose accuracy is known exactly.

    ``always_a`` is the degenerate strategy the 0.5 floor exists to expose: it must come
    out at exactly 0.5 pooled on every cell, with zero dose-response on both axes and a
    zero matched difference. ``oracle`` answers the ground truth everywhere: 1.0 pooled,
    ``P(change|diff) = 1``, zero span. Any deviation is a bug in the analysis, not in the
    model.
    """

    manifest = ARTIFACT / "manifest.jsonl"
    rows = analyse_mod.load_manifest(manifest)
    predictions = tmp_path / f"pred_{strategy}.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(
                {
                    "pair_id": row["pair_id"],
                    "parsed_choice": (
                        "A" if strategy == "always_a" else row["ground_truth"]
                    ),
                }
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    results = analyse_mod.analyse(manifest, predictions, n_boot=200, seed=7)
    assert results["coverage"]["n_items"] == 100
    assert results["coverage"]["n_pairs_without_prediction"] == 0
    assert results["coverage"]["span_fields_present"] is True
    for block in results["per_axis_level"].values():
        if strategy == "always_a":
            assert block["accuracy"]["pooled"] == pytest.approx(0.5)
            assert block["accuracy"]["original"] == pytest.approx(1.0)
            assert block["accuracy"]["diff"] == pytest.approx(0.0)
            assert block["P_change_given_diff"] == pytest.approx(0.0)
        else:
            assert block["accuracy"]["pooled"] == pytest.approx(1.0)
            assert block["P_change_given_diff"] == pytest.approx(1.0)
    for block in results["dose_response"].values():
        assert block["accuracy_span_pooled_points"] == pytest.approx(0.0, abs=1e-9)
        # and the realised values are still reported next to the flat curve
        assert all(
            value is not None for value in block["per_level_accuracy_pooled"]
        )
    for block in results["matched_comparison"].values():
        assert block["mean_difference_pooled_points"] == pytest.approx(0.0, abs=1e-9)
        assert block["realised_mismatch_median_db"] is not None
    # the realised axis values are the span ones and they do move across levels
    for axis in ("energy", "audibility"):
        values = [
            results["per_axis_level"][f"{axis}_L{level}"]["realised"][
                "original_span_audibility_db_median"
            ]
            for level in range(5)
        ]
        assert max(values) - min(values) > 15.0, (axis, values)
