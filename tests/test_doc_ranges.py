"""Guard the prose ranges in the docs against the committed artifacts.

The audit that motivated this file found several ranges in `README.md` and
`docs/handoff.md` that had gone stale when a model row was added: `A_diff` was
quoted as "73-76%" when the three competent models span 69.25-76.00%, and the
sham range as "0.005-0.013" when it is 0.0017-0.0092. These are hand-written
prose, so nothing else catches them.

Each test recomputes the range from the analysis JSONs and asserts the docs state
it correctly. If a number moves upstream, the test fails and the prose must be
updated deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The three models that clear the competence gate. AF3 fails it, so prose about
# "where the models are competent" must not include it.
COMPETENT = [
    ROOT / "outputs" / "round2" / "confirm_qwen7b.json",
    ROOT / "outputs" / "round2" / "confirm_qwen3b.json",
    ROOT / "outputs" / "p0_fun_audio_chat" / "confirm_fun_audio_chat_8b.json",
]


def _metrics(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def _span(values) -> tuple[float, float]:
    seq = list(values)
    return min(seq), max(seq)


@pytest.fixture(scope="module")
def competent_spans():
    blocks = [_metrics(p) for p in COMPETENT]
    return {
        "A_diff": _span(b["A_diff"] for b in blocks),
        "A_same": _span(b["A_same"] for b in blocks),
        "A_nuisance_removed": _span(b["A_nuisance_removed"] for b in blocks),
        "A_sham": _span(b["A_sham"] for b in blocks),
    }


def _claim_text(path: Path) -> str:
    """Doc text excluding the defect register, which quotes superseded values.

    The changelog in `docs/handoff.md` deliberately records what the old numbers
    were ("the sham range 0.005-0.013 is stale"), so scanning the whole file
    would flag the correction itself. Table rows in that register start with `|`.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("|"))


def test_readme_adiff_range_matches_artifacts(competent_spans):
    text = _claim_text(ROOT / "README.md")
    lo, hi = competent_spans["A_diff"]
    expected = f"{lo * 100:.0f}-{hi * 100:.0f}%"
    assert expected in text, (
        f"README should state the A_diff range as {expected} "
        f"(actual {lo:.4f}-{hi:.4f})"
    )
    assert "73-76%" not in text, "the stale A_diff range is still present"


def test_readme_asame_range_matches_artifacts(competent_spans):
    text = _claim_text(ROOT / "README.md")
    lo, hi = competent_spans["A_same"]
    expected = f"{lo * 100:.0f}-{hi * 100:.0f}%"
    assert expected in text, (
        f"README should state the A_same range as {expected} "
        f"(actual {lo:.4f}-{hi:.4f})"
    )
    assert "11-12%" not in text, "the stale A_same range is still present"


def test_readme_sham_range_matches_artifacts(competent_spans):
    text = _claim_text(ROOT / "README.md")
    lo, hi = competent_spans["A_sham"]
    # README quotes sham to one decimal place, e.g. "0.2-0.9%"
    expected = f"{lo * 100:.1f}-{hi * 100:.1f}%"
    assert expected in text, (
        f"README should state the sham range as {expected} "
        f"(actual {lo:.4f}-{hi:.4f})"
    )
    assert "0.6-0.9%" not in text, "the stale sham range is still present"


def test_handoff_sham_range_matches_artifacts(competent_spans):
    text = _claim_text(ROOT / "docs" / "handoff.md")
    lo, hi = competent_spans["A_sham"]
    expected = f"{lo:.4f}\u2013{hi:.4f}"
    assert expected in text, (
        f"handoff should state the sham range as {expected} "
        f"(actual {lo:.4f}-{hi:.4f})"
    )
    assert "0.005\u20130.013" not in text, "the stale sham range is still present"


def test_bias_wording_is_consistent_across_docs():
    """Both bias figures must appear with one value everywhere they are quoted.

    Scans claim text only: the changelog quotes the superseded approximate forms
    on purpose when recording what was fixed.
    """

    files = [
        ROOT / "README.md",
        ROOT / "docs" / "models.md",
        ROOT / "docs" / "handoff.md",
        ROOT / "scripts" / "synth_counterfactual_follow.py",
    ]
    texts = {f: _claim_text(f) for f in files}

    # pooled-vs-pair-mean delta, and single-order-vs-pair-mean delta
    for value in ("0.37", "0.66"):
        assert any(value in t for t in texts.values()), f"{value} pt not stated anywhere"
    # no approximate restatements of the same quantities
    for stale in ("~0.4 pt", "~0.5 pt", "~0.6 pt", "~0.7 pt", "≤0.4 pt", "≤0.7 pt"):
        for name, t in texts.items():
            assert stale not in t, f"{name.name} still uses the approximate form {stale!r}"


def test_readme_class_correlations_match_the_profile():
    """The per-class correlations went stale once (0.78/0.77 -> 0.668/0.672).

    They are hand-written prose in README.md, so only this guard catches a move.
    """

    path = ROOT / "outputs" / "round2" / "confirm_class_profile.json"
    if not path.exists():
        pytest.skip("class profile not present")
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    corr = {k: v["correlations_across_classes"] for k, v in models.items()}
    assert set(corr) == {"r2_qwen7b", "r2_qwen3b"}, set(corr)

    text = _claim_text(ROOT / "README.md")
    for key in ("r2_qwen7b", "r2_qwen3b"):
        for field in ("accuracy_vs_selectivity", "accuracy_vs_A_same"):
            value = corr[key][field]
            stated = f"{value:.3f}"
            assert stated in text, (
                f"README should state {key}.{field} as {stated} (actual {value!r})"
            )
    for stale in ("**0.78** (3B) and **0.77**", "**-0.74** and **-0.70**"):
        assert stale not in text, f"the stale class correlation {stale!r} is still present"


def test_readme_gind_matches_the_marginals():
    """`G_ind = (2a_o-1)(a_d+a_s-1)` must be quoted as computed, not approximated.

    It replaced the old "null share" column, whose premise the manuscript drops.
    """

    path = ROOT / "outputs" / "evidence" / "evidence_table.json"
    if not path.exists():
        pytest.skip("evidence table not present")
    models = json.loads(path.read_text(encoding="utf-8"))["models"]

    # Raw text, not _claim_text: the G_ind values live in a Markdown table.
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, model in models.items():
        follow = model.get("follow")
        if not follow:
            continue
        a_o = model["attribution"]["original_accuracy"]
        a_d = follow["subsets"]["all"]["follow_diff"]["rate"]
        a_s = follow["subsets"]["all"]["preserve_same"]["rate"]
        g_ind = (2 * a_o - 1) * (a_d + a_s - 1)
        stated = f"{g_ind:.3f}"
        assert stated in text, (
            f"README should state G_ind for {key} as {stated} (actual {g_ind!r})"
        )
    for stale in ("null share", "71-81%"):
        assert stale not in text, f"the superseded null framing {stale!r} is still present"
