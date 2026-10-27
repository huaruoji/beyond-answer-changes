# Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence

Code and frozen evidence for a study of how audio-language models attribute local
acoustic evidence. Manuscript under review at ICASSP 2027, paper #13728.

## What the instrument measures

An answer change after a local sound-event edit can reflect the event's class,
sensitivity to waveform rewriting, or both. A whole-clip removal, a silence or a
mask cannot separate the two, because all of them change the class and the
waveform at once.

This instrument replaces the queried event with either another recording of the
**same** class or an event from a **different** class. Both arms use the same
target span, donor-fitting procedure, level estimate and crossfade. The same-class
arm is therefore a matched control: it preserves the correct answer while
rewriting the interval, so the difference between the arms isolates the response
to the class change rather than to the edit.

Seven conditions per item, both option orders, 600 synthetic soundscapes with
annotated event intervals, vocabulary frozen before generation.

## Headline result

`Δ = A_diff − A_same`, the different-class minus same-class answer-change rate.
All rows are the corrected grid, in which a replacement keeps the clip's
background.

| Model | original acc. | A_diff | A_same | A_nuis | A_bg | A_sham | Δ (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-Omni-7B | 0.882 | 0.732 | 0.122 | 0.055 | 0.013 | 0.006 | **0.609** [+0.573, +0.646] |
| Qwen2.5-Omni-3B | 0.853 | 0.703 | 0.125 | 0.071 | 0.015 | 0.009 | **0.578** [+0.540, +0.617] |
| Fun-Audio-Chat-8B | 0.847 | 0.653 | 0.157 | 0.075 | 0.011 | 0.002 | **0.497** [+0.453, +0.539] |
| Audio Flamingo 3 | 0.704 | 0.408 | 0.098 | 0.077 | 0.027 | 0.009 | not read, fails the competence gate |

Nuisance removal, background-window edits and sham edits change answers far less
often than a class change, so the response sits at the queried event rather than
at the editing path.

## Capability-conditioned follow

`follow(diff)` is the rate at which the answer names the donor class, which is the
ground truth of the different-class condition. E2 restricts to trials whose
original and same-class answers are both correct.

| Model | follow(diff), all | follow(diff), E2 | pair-mean E2 | sham floor |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni-7B | 0.845 [0.821, 0.869] | 0.813 (n=971) | 0.819 | 0.0058 |
| Qwen2.5-Omni-3B | 0.838 [0.815, 0.862] | 0.802 (n=941) | 0.801 | 0.0092 |
| Fun-Audio-Chat-8B | 0.800 [0.771, 0.828] | 0.738 (n=891) | 0.737 | 0.0017 |

The pair mean averages within a source pair before averaging over pairs, which
removes the surface-order effect and is the estimator the paper quotes.

## Acoustic-distance control

Among the 236 soundscapes whose different-class and same-class log-mel edit
distances match within 15%, the contrast survives.

| Model | matched soundscapes | matched trials | Δ (95% CI) |
|---|---:|---:|---|
| Qwen2.5-Omni-7B | 236 | 472 | **0.617** [+0.555, +0.678] |
| Qwen2.5-Omni-3B | 236 | 472 | **0.530** [+0.462, +0.597] |

## Layout

```
src/                  instrument, model wrappers, synthesis helpers
scripts/              build, audit, inference and analysis entry points
tests/                invariants and schema guards
configs/              frozen run configuration
docs/                 protocol invariants, corpus registry, synthesis design, models
paper/                manuscript source, figures and the reference checks
outputs/evidence/     the frozen table the manuscript quotes from
```

The manuscript never quotes an analysis file directly. Every number in it comes
from `outputs/evidence/evidence_table.json`, which records the SHA256 of each
source file so a stale table cannot ship.

## Data and artifacts not included

Two artifact sets are too large for a code repository and are available on
request.

- Synthesized soundscapes, about 289 MB, 743 audio files with 5450 Scaper
  annotation files.
- Model predictions, about 116 MB of JSONL across the three competent models.

`scripts/consolidate_evidence.py` regenerates `outputs/evidence/` from those
files. `paper/verify_reference_draft.py` re-derives the manuscript's arithmetic
and checks each source hash, so it needs the same artifacts.

## Environment

Python 3.11 or newer. `pip install -r requirements.txt` covers the analysis path,
and `requirements-models.txt` covers inference. Scaper is never imported
directly. `src/synth.py` owns the environment contract described in
`docs/protocol-invariants.md`.

Corpora, both used under their own terms and neither redistributed here:

- FSD50K event recordings, through the FSD-MIX-SED source release.
- TUT Urban Acoustic Scenes 2018 background recordings.

## Reproduce

1. Build the grid with `scripts/build_synth_manifests.py` and
   `scripts/prepare_synth_corpus.py`.
2. Audit it against the invariants with `scripts/synth_audit.py`, before any
   model sees the audio.
3. Run inference with `scripts/evaluate_manifest.py` and the model wrappers in
   `src/models/`.
4. Analyse with `scripts/synth_attribution.py`,
   `scripts/synth_counterfactual_follow.py`, `scripts/synth_confound_check.py`
   and `scripts/synth_class_profile.py`.
5. Freeze the numbers with `scripts/consolidate_evidence.py`.
6. Build the manuscript from `paper/` with `latexmk -pdf reference-draft.tex`.

## License

MIT, see `LICENSE`.
