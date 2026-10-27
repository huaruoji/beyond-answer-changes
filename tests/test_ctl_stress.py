"""Regression test for the stress ladder: the script that §3.4f could not reproduce.

``scripts/ctl_stress_axis.py`` crashed after the donor fix (CTL-Plan §3.4i#1) because it
rendered the ``same`` condition from the different-class donor — the construction that had
produced the void contrast — and ``render_item`` now refuses to do that. It then hit the
post-splice clipping guard, because it never fitted headroom. Both failures are locked here
on one real item rather than by re-running the whole ladder.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import ctl_stress_axis as stress

FOREGROUND = Path("data/processed/synth/foreground_all")
BACKGROUND = Path("data/processed/synth/background")
VOCAB = Path("data/processed/synth/vocab_confirm.txt")

pytestmark = pytest.mark.skipif(
    not FOREGROUND.exists() or not BACKGROUND.exists(),
    reason="CTL corpus not present in this checkout",
)


def test_stress_ladder_renders_three_distinct_conditions(tmp_path: Path) -> None:
    labels = [ln.strip() for ln in VOCAB.read_text(encoding="utf-8").splitlines() if ln.strip()]
    backgrounds = sorted(
        p for d in sorted(BACKGROUND.iterdir()) if d.is_dir() for p in d.glob("*.wav")
    )
    query, donor_label = labels[0], labels[1]
    files = sorted((FOREGROUND / query).glob("*.wav"))
    event_path, same_path = files[0], files[1]
    donor_path = sorted((FOREGROUND / donor_label).glob("*.wav"))[0]

    rng = random.Random(5)
    item = stress.build_ladder_item(
        "stress-test-000",
        query,
        donor_label,
        event_path,
        backgrounds[0],
        donor_path,
        same_path,
        drop_db=stress.LADDER_DB[0],
        rng=rng,
    )
    assert item.same_donor_class == query
    donor_wave, _ = stress.read_audio_mono(donor_path, stress.RATE)
    same_wave, _ = stress.read_audio_mono(same_path, stress.RATE)

    # the headroom fit exists because the post-splice peak is what render_item refuses
    fitted, peak = stress.fit_headroom(item, donor_wave, same_wave)
    assert peak <= stress.PEAK_CEILING, peak
    assert float(fitted.metadata["shared_gain"]) <= 1.0

    out = stress.render_item(
        fitted,
        tmp_path,
        calibration="equal_energy",
        masker=stress.MASKER_KIND,
        conditions=stress.CONDITIONS,
        donor=donor_wave,
        donor_span_samples=donor_wave.size,
        same_donor=same_wave,
        same_donor_span_samples=same_wave.size,
    )
    assert set(out["audio"]) == {"original", "diff", "same"}
    diff, _ = sf.read(out["audio"]["diff"], dtype="float32")
    same, _ = sf.read(out["audio"]["same"], dtype="float32")
    assert not np.array_equal(diff, same), "the stress ladder rendered a void contrast"
    for condition, path in out["audio"].items():
        wave, _ = sf.read(path, dtype="float32")
        assert float(np.abs(wave).max()) <= 1.0, condition
    rows = {str(row["condition"]): row for row in out["rows"]}  # type: ignore[union-attr]
    assert rows["diff"]["donor_used"] == donor_label
    assert rows["same"]["donor_used"] == query
    # the rows carry the realised cue level under both conventions
    for condition, row in rows.items():
        assert np.isfinite(float(row["span_audibility_db"])), condition
        assert np.isfinite(float(row["span_energy_snr_db"])), condition
        assert row["span_cue_signal"] == ("replacement" if condition in {"diff", "same"} else "event")
