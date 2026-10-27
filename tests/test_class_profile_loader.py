"""Regression test for the same view-keying defect in synth_class_profile.py.

The counterfactual-follow analysis keyed views by ``source_pair_id`` and so let
the second surface order overwrite the first. ``synth_class_profile.py`` had the
identical defect, which inflated the per-class accuracy/selectivity correlation
(0.86/0.91 before the fix, 0.78/0.77 after) by silently halving the sample and
selecting one surface order. This test pins the keying.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synth_class_profile.py"


@pytest.fixture(scope="module")
def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_keys_views_by_pair_id_not_source_pair_id(source: str):
    """The loader must group by pair_id so both surface orders survive."""

    assert 'by_item[row["pair_id"]]' in source, (
        "synth_class_profile.py must key views by pair_id; keying by "
        "source_pair_id silently discards one surface order per item"
    )
    assert 'by_item[row["source_pair_id"]]' not in source, (
        "the source_pair_id keying defect has been reintroduced"
    )


def test_comment_records_why(source: str):
    """A future reader must be told why the key choice matters."""

    assert "surface order" in source.lower()
