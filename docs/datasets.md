# Corpus Registry

Last updated: 2026-09-22. This file records what the instrument actually runs on
now, what was audited and rejected, and why. It replaces the pre-2026-09-12
version, which was written around the withdrawn TEMPO-derived MVP.

## In use

| Role | Source | Size | Licence | Notes |
|---|---|---|---|---|
| Foreground events | FSD-MIX-SED source material on [Zenodo 5574135](https://zenodo.org/records/5574135) | 948 MB, 10,296 wavs, 89 FSD50K leaf classes | CC-BY-4.0 | each clip carries a single human-validated leaf label with Present-and-Predominant annotation; this is what makes the donors trustworthy |
| Background | TUT Urban Acoustic Scenes 2018 development, 16-bit mirror on Hugging Face | 4 shards, 1008 recordings of 10 s | mirror declares none; upstream is AFL-3.0 — verify before redistribution | scenes: airport 265, bus 242, metro 261, metro_station 240 |
| Transfer check | TEMPO `audio_grounding` evaluation split, `tacos` source | 269 MB, 2000 clips | see the dataset card | **its numbers are withdrawn** — see below |

Vocabulary: 84 of the 89 FSD50K classes are kept. Five are excluded because they
plausibly occur inside an urban scene recording, which would make "which of these
two is present" ambiguous for reasons unrelated to the model: `Wind`,
`Subway_and_metro_and_underground`, and the three speech classes.

Class names become question text by rule plus an explicit override table in
`scripts/prepare_synth_corpus.py`, never by a language model (ADR-010). The
mapping is frozen in `data/processed/synth/corpus.json` so question text cannot
drift from the label.

Instance splits: the pilot grid reads `foreground_pilot` (train+test), the
confirmatory split reads `foreground_confirm` (val), screening reads
`foreground_all`. No event instance is shared between the two measurement splits.

**Audio provenance for the third-model run.** The GPU host that ran the third
model did not have the confirm audio, so it was rebuilt there on 2026-09-21 from
the same pinned sources: `FSD_MIX_SED.source.tar.gz` (Zenodo 5574135) for the
events, the `test-00000..00003-of-00010` shards of
`wetdog/TUT-urban-acoustic-scenes-2018-development-16bit` for the background, and
`vocab.json` (Zenodo 5574135). `prepare_synth_corpus.py` reproduced the tracked
`corpus.json` with no diff, and the rebuilt grid reproduced all 600 JAMS and all
8400 manifest rows bit-for-bit. Per-file SHA256 of the 4200 rebuilt wavs is in
`outputs/p0_fun_audio_chat/audio_digests.json` (pre-fix) and
`outputs/p0_fun_audio_chat/audio_digests_v2.json` (the corrected `confirm_v2` grid).

The original waveforms were still present on the local host at
`data/processed/synth/confirm_audio/` (4200 wavs, ctime 2026-09-13, hardlinked
from `round2_audio/`). All 4200 digests match those originals exactly (4200
match, 0 mismatch), so the rebuilt audio is sample-identical to what the Qwen
runs consumed. JAMS equality alone would only have proved placement/RNG identity;
the digest comparison closes the waveform question. An earlier revision of this
section claimed no original wav remained, which was incorrect.

## Audited and rejected

**TEMPO `audio_grounding`, four of its five splits.** 14.88 GB total, of which
only the 269 MB `tacos` split admits the design at all. `audioset_strong` and
`synthetic_ag` annotate roughly one event per clip (rows/clip 1.00–1.02), so a
second, temporally separated event — the thing a matched control needs — cannot
exist. Measured with `scripts/feasibility_scan.py`; they yield zero eligible
items. The remaining split gives 168 strict / 238 moderate / 354 lenient clips.

**TEMPO as a whole, for this design.** Three independent structural obstacles, all
about how it is annotated rather than how much of it there is:

1. too few clips carry two separable events;
2. its event descriptions are free text — 3,848 distinct over 2,000 clips, only 92
   appearing in more than one — so a same-class donor is not constructible
   (ADR-013);
3. it ships positive labels only, so replacing the annotated occurrence of a sound
   does not remove the class, and "the truth flips" is not a licensed inference.
   `scripts/probe_natural_target_persistence.py` shows this directly: after
   replacement the models still report the class present 79–88% of the time
   against baselines of 84–91%.

The third is the one that mattered and the one that generalises: **before running
any "remove the evidence so the truth flips" experiment, verify the class is
actually absent afterwards.** Positive-label corpora do not give you that, and the
check costs one yes/no pass.

## Considered, not used

| Source | Why not |
|---|---|
| DCASE2026 Task 5 dev | human QA, but evidence intervals are not promised; metadata only in this checkout |
| AudioMCQ-StrongAC-GeminiCoT | contribution filtering; no strong local boundaries in the visible schema |
| MUSAN | the intended background source, but all three OpenSLR mirrors returned zero bytes on both ranged and plain requests; replaced by TUT |
| TEMPO full collection (64.9 GB) | deferred. Nothing in the current design needs it — the constraint is annotation structure, not volume |
