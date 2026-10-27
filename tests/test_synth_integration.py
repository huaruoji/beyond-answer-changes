"""End-to-end test of the generator -> audit path on a throwaway corpus.

This is the test that caught two real defects during development: scaper renders
at 44.1 kHz regardless of the input material, so a donor's sample count
disagreed with its span arithmetic; and scaper writes the base at 32-bit while
edits were written at 16-bit, which left ``sham`` not bit-identical to
``original`` and therefore not a no-op.

Skipped, not failed, when the scaper toolchain is unavailable: it needs libsox
on the loader path from the dedicated conda environment (see
``Environment-Notes``). Unit-level invariants are covered by ``test_synth.py``
without that dependency.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from src.synth import DEFAULT_SAMPLE_RATE  # noqa: F401

    HAVE_SCAPER = True
except Exception:  # pragma: no cover - depends on libsox being present
    HAVE_SCAPER = False

pytestmark = pytest.mark.skipif(
    not HAVE_SCAPER, reason="scaper toolchain (libsox) not available"
)

SR = 16000
SR_BACKGROUND = 22050  # deliberately different, to exercise resampling
DURATION = 10.0


def _build_corpus(root: Path) -> Path:
    rng = np.random.default_rng(0)
    classes = {
        "dog_bark": 300.0,
        "siren": 900.0,
        "glass_breaking": 1800.0,
        "footsteps": 150.0,
        "camera_click": 2500.0,
    }
    for label, freq in classes.items():
        directory = root / "foreground" / label
        directory.mkdir(parents=True, exist_ok=True)
        for index, (duration, amplitude) in enumerate(
            [(0.5, 0.35), (0.9, 0.30), (1.3, 0.40)]
        ):
            t = np.arange(int(duration * SR)) / SR
            tone = amplitude * np.sin(2 * np.pi * freq * t)
            tone[:240] *= np.linspace(0, 1, 240)
            tone[-240:] *= np.linspace(1, 0, 240)
            sf.write(directory / f"{label}_{index}.wav", tone.astype(np.float32), SR)
    for index in range(3):
        directory = root / "background" / f"scene{index}"
        directory.mkdir(parents=True, exist_ok=True)
        noise = 0.03 * rng.normal(size=int(12 * SR_BACKGROUND))
        noise = np.convolve(noise, np.ones(80) / 80, mode="same")
        sf.write(
            directory / f"bg_{index}.wav", noise.astype(np.float32), SR_BACKGROUND
        )
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(classes) + "\n")
    return vocab


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *arguments],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("synth_e2e")
    vocab = _build_corpus(root)
    manifest = root / "manifest.jsonl"
    result = _run(
        "build_synth_manifests.py",
        "--vocab", str(vocab),
        "--foreground-dir", str(root / "foreground"),
        "--background-dir", str(root / "background"),
        "--audio-dir", str(root / "audio"),
        "--jams-dir", str(root / "jams"),
        "--output", str(manifest),
        "--split", "e2e",
        "--target", "3",
        "--seed", "20260912",
    )
    assert result.returncode == 0, result.stderr[-3000:]
    summary = json.loads(manifest.with_suffix(".summary.json").read_text())
    audit_result = _run(
        "synth_audit.py",
        str(manifest),
        "--output", str(root / "audit.json"),
        "--audio-root", str(root / "audio"),
    )
    audit = json.loads((root / "audit.json").read_text())
    return {
        "root": root,
        "manifest": manifest,
        "summary": summary,
        "audit": audit,
        "audit_returncode": audit_result.returncode,
    }


def test_generator_produces_the_full_condition_grid(generated) -> None:
    summary = generated["summary"]
    assert summary["items"] == 3
    # 3 items x 7 conditions x 2 option orders
    assert summary["records"] == 3 * 7 * 2
    assert summary["split"] == "e2e"


def test_every_condition_is_balanced_across_option_positions(generated) -> None:
    for condition, counts in generated["summary"]["condition_label_counts"].items():
        assert counts.get("A", 0) == counts.get("B", 0), condition


def test_audit_passes_with_no_errors(generated) -> None:
    audit = generated["audit"]
    assert audit["error_count"] == 0, audit["errors"][:10]
    assert audit["valid"] is True
    assert generated["audit_returncode"] == 0


def test_rendered_clips_use_the_forced_sample_rate(generated) -> None:
    """scaper defaults to 44.1 kHz; the render must override it."""

    files = sorted((generated["root"] / "audio").glob("*__original.wav"))
    assert files
    for path in files:
        info = sf.info(str(path))
        assert info.samplerate == DEFAULT_SAMPLE_RATE
        assert info.frames == int(DURATION * DEFAULT_SAMPLE_RATE)


def test_sham_is_imperceptible(generated) -> None:
    """The sham must not perturb the audio beyond float rounding.

    It is deliberately *not* asserted bit-identical. The sham rewrites a span
    with its own samples through the real crossfade arithmetic, and
    ``(1-w)*x + w*x`` does not always reproduce ``x`` exactly; measured on the
    real corpus the worst case is 1 LSB on about 6 samples out of 160000. That is
    rounding, not an artifact, and the sham exists precisely to keep exercising
    the path that would produce a real artifact — short-circuiting it to an
    identity would make the control vacuous.

    The bound is what matters: the withdrawn pipeline's reversed tail produced
    full-scale jumps, which is roughly 30000 LSB.
    """

    audio = generated["root"] / "audio"
    originals = sorted(audio.glob("*__original.wav"))
    assert originals
    worst = 0
    for original in originals:
        sham = original.with_name(original.name.replace("__original", "__sham"))
        first, _ = sf.read(str(original), dtype="int16")
        second, _ = sf.read(str(sham), dtype="int16")
        worst = max(worst, int(np.abs(first.astype(np.int32) - second.astype(np.int32)).max()))
    assert worst <= 2, f"sham differs by {worst} LSB; that is perturbation, not rounding"


def test_control_audio_leaves_the_query_span_untouched(generated) -> None:
    """I2 at the waveform level, on the real generated files."""

    rows = [
        json.loads(line)
        for line in generated["manifest"].read_text().splitlines()
        if line.strip()
    ]
    by_item: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_item.setdefault(row["source_pair_id"], {})[row["condition"]] = row
    checked = 0
    for item, conditions in by_item.items():
        original = conditions["original"]
        start = int(round(original["evidence"][0]["start"] * DEFAULT_SAMPLE_RATE))
        end = int(round(original["evidence"][0]["end"] * DEFAULT_SAMPLE_RATE))
        base, _ = sf.read(
            str(generated["root"] / "audio" / Path(original["audio"]).name),
            dtype="float32",
        )
        for condition in ("nuisance_removed", "background_removed"):
            edited, _ = sf.read(
                str(
                    generated["root"]
                    / "audio"
                    / Path(conditions[condition]["audio"]).name
                ),
                dtype="float32",
            )
            assert np.allclose(edited[start:end], base[start:end], atol=2e-4), (
                f"{item}/{condition} altered the query span"
            )
            checked += 1
    assert checked >= 6
