"""Guard the difficulty-knob table (ADR-008) against the artifacts.

`README.md` quotes the SNR/polyphony gradient, and ADR-008 makes it load-bearing:
the manipulation must be shown to move accuracy *before* selectivity is read
across it. The numbers were previously hand-written with no generating script, so
nothing checked them. `scripts/difficulty_knobs.py` now produces them, and these
tests pin the README to its output.

Also pins the structural point that makes the table easy to get wrong: the `k_*`
accuracy-knob splits carry only `original` and `silence`, so they cannot yield a
selectivity. Only the `snr*_full` splits carry the matched diff/same pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qwen3b():
    return _load(ROOT / "outputs" / "round2" / "difficulty_qwen3b.json")


def test_readme_snr_deltas_match_artifact(qwen3b):
    """The two SNR deltas README quotes must match the generated table."""

    text = README.read_text(encoding="utf-8")
    knobs = qwen3b["accuracy_knobs"]
    for split in ("k_snr0", "k_snrm5"):
        delta = knobs[split]["delta_pt"]
        assert f"{delta:+.1f} pt" in text or f"{delta:.1f} pt" in text, (
            f"README should quote {split} delta as {delta:+.1f} pt"
        )


def test_readme_polyphony_deltas_match_artifact(qwen3b):
    text = README.read_text(encoding="utf-8")
    knobs = qwen3b["accuracy_knobs"]
    for split in ("k_n2", "k_n4"):
        delta = knobs[split]["delta_pt"]
        assert f"{delta:+.1f} pt" in text or f"{delta:.1f} pt" in text, (
            f"README should quote {split} delta as {delta:+.1f} pt"
        )


def test_baseline_is_the_pilot_split(qwen3b):
    """The baseline must come from the seven-condition pilot grid at +10 dB."""

    base = qwen3b["baseline"]
    assert base["split"] == "pilot"
    assert base["n"] == 800
    assert base["accuracy"] == pytest.approx(0.8275, abs=1e-4)


def test_accuracy_knobs_cannot_yield_selectivity():
    """The k_* splits must not be used to compute an SNR selectivity.

    This is the trap: they look like SNR splits, but they carry only
    `original` and `silence`, so the matched diff/same contrast does not exist in
    them. The script must read the `snr*_full` splits for that, and this test
    records why.
    """

    path = ROOT / "outputs" / "round2" / "r2_qwen3b.jsonl"
    if not path.exists():
        pytest.skip("predictions not present")
    conditions: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split", "")
        if split.startswith("k_"):
            conditions.setdefault(split, set()).add(row["condition"])
    assert conditions, "expected k_* splits in the confirm predictions"
    for split, conds in conditions.items():
        assert conds == {"original", "silence"}, (
            f"{split} unexpectedly carries {sorted(conds)}; if it now has the "
            "matched pair, the selectivity note in difficulty_knobs.py is stale"
        )


def test_snr_selectivity_is_reproducible(qwen3b):
    """The SNR selectivity table must come from the snr*_full splits."""

    block = qwen3b.get("selectivity_across_snr")
    assert block, "the artifact should carry a selectivity-across-SNR section"
    for split, values in block.items():
        assert values["n"] > 0
        assert values["selectivity"] is not None
        lo, hi = values["ci95"]
        assert lo < values["selectivity"] < hi


def test_readme_records_the_snr_decision(qwen3b):
    """ADR-008's verdict must be stated: SNR moves accuracy, polyphony does not."""

    text = README.read_text(encoding="utf-8")
    assert "SNR is a real knob" in text
    assert "polyphony is not" in text
    # and the claim must be true of the artifact
    knobs = qwen3b["accuracy_knobs"]
    assert abs(knobs["k_snrm5"]["delta_pt"]) >= 5.0, "SNR -5 dB should move >=5 pt"
    assert abs(knobs["k_n4"]["delta_pt"]) < 5.0, "polyphony 4 should move <5 pt"
