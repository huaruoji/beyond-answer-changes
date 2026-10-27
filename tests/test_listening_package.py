"""Guard the listening package: the page must be usable by a remote reviewer.

Two defects motivated these tests, both found when a second listener was asked to
run the check from a fresh clone:

1. The page references `audio/*.wav`, but the wavs were gitignored, so a clone
   produced a page whose every player 404s. A reviewer cannot do a listening
   check on audio they do not have.
2. The page had no way to return results, and the first session (2026-09-18) left
   no per-item record at all — only a prose summary. So a second listener could
   not be compared against the first, which is the entire point of a second
   listener.

These tests pin the fixes: the audio is tracked, the export control exists, and
the export carries what a comparison needs (listener, seed, per-item checks).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "processed" / "synth" / "listening"
#: The confirm-split package is the one whose items the paper reports; the pilot
#: package is what the 2026-09-18 session used. Both must be usable.
PKG_CONFIRM = ROOT / "data" / "processed" / "synth" / "listening_confirm"
HTML = PKG / "index.html"
INDEX = PKG / "index.json"


@pytest.fixture(scope="module")
def page() -> str:
    if not HTML.exists():
        pytest.skip("listening page not generated")
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index():
    if not INDEX.exists():
        pytest.skip("listening index not generated")
    return json.loads(INDEX.read_text(encoding="utf-8"))


@pytest.mark.parametrize("pkg", [PKG, PKG_CONFIRM], ids=["pilot", "confirm"])
def test_package_is_complete_and_tracked(pkg):
    """Each package must exist, reference tracked audio, and carry the export UI.

    Parametrised because both packages ship: a reviewer may be given either, and a
    missing wav or a missing export button is equally fatal in both.
    """

    html = pkg / "index.html"
    if not html.exists():
        pytest.skip(f"{pkg.name} not generated")
    text = html.read_text(encoding="utf-8")

    referenced = sorted(set(re.findall(r'src="audio/([^"]+\.wav)"', text)))
    assert len(referenced) == 100, f"{pkg.name}: expected 100 audio refs"

    missing_disk = [n for n in referenced if not (pkg / "audio" / n).exists()]
    assert not missing_disk, f"{pkg.name}: missing on disk: {missing_disk[:3]}"

    tracked = {
        Path(p).name
        for p in subprocess.run(
            ["git", "ls-files", str(pkg / "audio")],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.split()
    }
    missing_git = [n for n in referenced if n not in tracked]
    assert not missing_git, (
        f"{pkg.name}: {len(missing_git)} referenced wavs are untracked, so a "
        f"fresh clone gets a broken page; first few: {missing_git[:3]}"
    )

    assert "exportResults" in text, f"{pkg.name}: no export control"
    assert 'id="listener"' in text, f"{pkg.name}: no listener field"
    sections = re.findall(r'<section data-item="([^"]+)" data-number="(\d+)"', text)
    assert len(sections) == 20, f"{pkg.name}: expected 20 items"


def test_packages_do_not_share_items():
    """The pilot and confirm packages must be disjoint item sets.

    If they overlapped, a second listener could not be told which split their
    results belong to.
    """

    if not INDEX.exists() or not (PKG_CONFIRM / "index.json").exists():
        pytest.skip("both packages required")
    pilot = {e["item"] for e in json.loads(INDEX.read_text(encoding="utf-8"))}
    confirm = {
        e["item"]
        for e in json.loads((PKG_CONFIRM / "index.json").read_text(encoding="utf-8"))
    }
    assert not (pilot & confirm), f"the two packages share items: {pilot & confirm}"
    assert len(pilot) == len(confirm) == 20


def test_confirm_package_uses_the_confirm_split():
    """The confirm package must actually be drawn from confirm.jsonl."""

    path = PKG_CONFIRM / "index.json"
    if not path.exists():
        pytest.skip("confirm package not generated")
    items = [e["item"] for e in json.loads(path.read_text(encoding="utf-8"))]
    assert all(i.startswith("confirm-") for i in items), (
        f"confirm package contains non-confirm items: "
        f"{[i for i in items if not i.startswith('confirm-')][:3]}"
    )


def test_generator_is_deterministic(tmp_path, index):
    """Re-running with the same seed must pick the same items.

    Without this, a second listener could not be given the same set as the first.
    """

    manifest = ROOT / "data" / "processed" / "synth" / "pilot.jsonl"
    audio = ROOT / "data" / "processed" / "synth" / "pilot_audio"
    if not manifest.exists() or not audio.exists():
        pytest.skip("pilot manifest or audio not present")

    script = ROOT / "scripts" / "make_listening_package.py"
    out = tmp_path / "again"
    proc = subprocess.run(
        ["python3", str(script), "--manifest", str(manifest),
         "--audio-root", str(audio), "--output", str(out),
         "--items", "20", "--seed", "20260916"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    regenerated = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [e["item"] for e in regenerated] == [e["item"] for e in index]
