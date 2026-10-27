# Synthetic soundscape design for local evidence attribution

Status: design frozen 2026-09-12, before any generation run. Changes require an
amendment section at the bottom, not a silent edit.


## Contents

- [1. Why synthesize](#1-why-synthesize)
- [2. Source material](#2-source-material)
- [3. Toolchain: scaper](#3-toolchain-scaper)
- [4. Rendering and editing specification](#4-rendering-and-editing-specification)
- [5. Item structure](#5-item-structure)
- [6. Conditions](#6-conditions)
- [7. Metrics](#7-metrics)
- [8. Gates](#8-gates)
- [9. Vocabulary screening (before the main grid)](#9-vocabulary-screening-before-the-main-grid)
- [10. Splits and seeds](#10-splits-and-seeds)
- [11. Difficulty knobs](#11-difficulty-knobs)
- [12. Human listening check](#12-human-listening-check)
- [13. Natural-audio transfer check](#13-natural-audio-transfer-check)
- [14. What this design still cannot claim](#14-what-this-design-still-cannot-claim)
- [15. Amendment 1 — corrections found while building the instrument](#15-amendment-1--corrections-found-while-building-the-instrument)
- [16. Amendment 2 — corrections found on the real corpus](#16-amendment-2--corrections-found-on-the-real-corpus)

## 1. Why synthesize

The natural-corpus route failed for a structural reason, not a data-volume one.
`scripts/feasibility_scan.py` measured, over the whole TEMPO `audio_grounding`
task (14.88 GB, five splits), that only the 269 MB `tacos` evaluation split
admits the design at all: 238 clips under the moderate criteria. The other four
sources annotate roughly one event per clip, so a second, temporally separated
event — the thing a matched control needs — cannot exist. They yield zero items.

Synthesis removes that constraint by construction:

| Requirement | Natural corpus | Synthesized |
|---|---|---|
| Event placement known exactly | annotation, 0.1 s rounded | we place it |
| Second event for a control | must already be annotated | we place it |
| Control interval non-overlap | luck (40/80 violated before) | enforced in code |
| Edit length matching | approximate | exact |
| Energy matching | not controlled | a parameter |
| Clip contents fully known | no (only positive labels) | yes |
| Scale | capped at 238 clips | capped by the event library |

The last row matters most. With TEMPO we could never rule out that an "empty"
region contained an unlabelled event, because the corpus only ships positive
labels. In a synthesized soundscape the background is a known file and the
foreground is a list we wrote, so "this window holds no event" is a fact.

## 2. Source material

**Foreground events: FSD-MIX-SED source material** (Zenodo
[5574135](https://zenodo.org/records/5574135), CC-BY-4.0, 948 MB). This is the
release that accompanies FSD-MIX-CLIPS: 10,296 clips from FSD50K, each carrying a
single validated leaf label, with leading and trailing silence trimmed. 89 leaf
classes, with `vocab.json` giving the class names. FSD-MIX-CLIPS used exactly
this material to build 281,039 scaper soundscapes, so it is a known-good
substrate for this kind of synthesis.

**Background: MUSAN noise subset** (OpenSLR [17](https://openslr.org/17/),
CC-BY-4.0). Real recordings, not synthetic noise. The previous attempt's
difficulty knob failed partly because the "noise bed" was 5 ms-smoothed Gaussian
rumble that peak normalisation then attenuated; using real ambience avoids
repeating that.

## 3. Toolchain: scaper

The field-standard tool is [scaper](https://github.com/justinsalamon/scaper)
(Salamon et al., WASPAA 2017), and DESED has used it for DCASE Task 4 since 2019.
We use it, after working around two environment problems.

**sox is missing and cannot be installed system-wide.** `soxbindings`, a scaper
dependency, compiles against `sox.h` and links `-lsox`; neither was present,
`ffmpeg` is absent, and `apt` requires a password we do not have. A dedicated
conda env supplies all three:

```bash
conda create -y -n audiotools -c conda-forge sox          # binary + sox.h + libsox.so
CPATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib \
  uv pip install --python .venv/bin/python soxbindings    # builds against conda sox
```

At run time scaper needs that env on `PATH` and `LD_LIBRARY_PATH`.

**scaper 1.6.5 predates NumPy 2.0.** It calls the removed alias `np.Inf` in four
places in `scaper/core.py`. `src/synth.py` sets `np.Inf = np.inf` immediately
before importing scaper. This is a compatibility shim, not a behavioural change,
and it is the only NumPy-2 incompatibility in scaper, jams or soxbindings.

Consequence: nothing may import `scaper` directly. Everything goes through
`src/synth.py`, which owns the shim and the environment contract. Recorded in
`Environment-Notes.md`.

**Division of labour.** scaper renders the base soundscape: it places events at
requested times, calibrates their gain against the background, handles
resampling, and emits JAMS with exact onsets and durations. Our code in
`src/synth.py` owns the seven conditions, because those overwrite exact sample
spans of an already-rendered soundscape and are not a scaper operation. The
ADR-011 invariants live in that layer.

## 4. Rendering and editing specification

A generated soundscape is 10.0 s. The sample rate is whatever the background
material uses; MUSAN is 16 kHz, which is also what the target models expect, so
scaper resamples the 44.1 kHz FSD50K foregrounds down to match.

**Base soundscape (scaper).**

```python
sc = scaper.Scaper(10.0, foreground_dir, background_dir)
sc.ref_db = <background loudness in LUFS>
sc.add_background(label=('const', bg_class), source_file=('const', bg_path), source_time=('const', 0))
sc.add_event(label=('const', c), source_file=('const', path), source_time=('const', 0),
             event_time=('const', t), event_duration=('const', d), snr=('const', snr_db),
             pitch_shift=('const', 0), time_stretch=('const', 1.0))
```

Every parameter is `const`, never a distribution: we want each item's placement
recorded exactly, not sampled. Onsets land on a 10 ms grid, chosen by our seeded
RNG before the call, and rejected if the event would overlap another or exceed
the clip.

scaper's `snr` is LUFS-based (`ref_db` is the background loudness in LUFS), so
the difficulty knob is perceptual rather than raw RMS. The achieved value is read
back from the JAMS file and stored per event.

**Conditions (our code).** Each condition rewrites an exact sample span of the
rendered soundscape:

- `target_replaced_diff` / `target_replaced_same`: overwrite `Q`'s span with a
  donor event, resampled to the span length and gain-matched to the span's
  original level.
- `nuisance_removed` / `background_removed` / `target_removed`: overwrite the
  span with background. The background is taken from elsewhere in the same clip,
  not from another file, so the replacement shares the clip's ambience.
- `sham`: overwrite `Q`'s span with its own samples, through the same code path.
- `silence`: zero the whole clip.

**Boundaries.** Every rewritten span gets a 10 ms raised-cosine crossfade at each
edge, `(1 - w) * original + w * inserted`, with `w` rising at the head and
falling at the tail. The tail weight direction is covered by a unit test because
the previous implementation in `src/interventions.py` had it reversed.

### Invariants (ADR-011, enforced in code, not audited afterwards)

| ID | Invariant | How it is structural |
|---|---|---|
| I1 | a donor is an event, never a clip | `place_event` takes a source path plus its own `[start, end)`; there is no code path that accepts a whole file without a span |
| I2 | control windows never overlap the target | placement assigns non-overlapping slots and asserts `gap >= min_gap_s` at construction time |
| I3 | edits are duration- and energy-matched | all control edits reuse the target's exact sample span; the mixer records achieved SNR per event and the audit checks the match |
| I4 | no transient at edit boundaries | head and tail crossfades both tested by asserting the first difference stays within 3x the untouched signal's |

## 5. Item structure

An item is one soundscape plus its option pair.

- One **query event** `Q` of class `c_Q`, the only event of that class present.
- One **nuisance event** `N` of class `c_N`, with `c_N` distinct from `c_Q` and
  from `c_X`. It is never mentioned in the question.
- The **option pair** is `{c_Q, c_X}` where `c_X` is a third class that does
  **not** occur in the original soundscape.
- Background fills the whole clip.

Question, rendered in two option orders:

> Listen to the audio. Which of these two sounds is present in it?
> Option A: 'X'. Option B: 'Y'. Respond with exactly Answer: A or Answer: B.

Frozen description strings come from `vocab.json`, lowercased and used verbatim,
so the question text never drifts from the annotation.

`N` exists so that one of the controls removes a real event rather than ambience.
Without it, "irrelevant edit" would always mean "quieter background", which is a
weaker contrast than "removed a different sound".

## 6. Conditions

Seven conditions per item, each rendered in both option orders.

| # | Condition | Operation | Truth | Purpose |
|---|---|---|---|---|
| 1 | `original` | none | `c_Q` | competence |
| 2 | `target_replaced_diff` | `Q`'s exact span overwritten with an instance of `c_X` | `c_X` | relevant, truth-flipping |
| 3 | `target_replaced_same` | `Q`'s exact span overwritten with a different instance of `c_Q` | `c_Q` | same region, same operation, truth preserved |
| 4 | `nuisance_removed` | `N`'s exact span replaced by background | `c_Q` | irrelevant event removed |
| 5 | `background_removed` | a `d_Q`-long event-free window replaced by background | `c_Q` | matched non-event edit |
| 6 | `sham` | `Q`'s span rewritten with its own samples through the same crossfade path | `c_Q` | pipeline artifact control |
| 7 | `silence` | whole clip zeroed | no unique truth | global audio dependency |

Conditions 2 and 3 are the matched pair the previous design lacked. Both touch
the same samples with the same operation and the same boundary treatment; they
differ only in whether the replacement preserves the queried identity. That is
what makes the headline number interpretable.

Condition 6 is the honesty check. Writing a span back with its own samples
should be a no-op; if the model's answer moves, the pipeline is perturbing the
audio and no other number can be trusted.

## 7. Metrics

Answer-change rates, all computed on the model's own prediction so they are
defined at any accuracy level:

```text
A_diff  = P(answer changes | target_replaced_diff)
A_same  = P(answer changes | target_replaced_same)
A_nuis  = P(answer changes | nuisance_removed)
A_bg    = P(answer changes | background_removed)
A_sham  = P(answer changes | sham)

selectivity   = A_diff - A_same      # headline
specificity   = A_same - A_sham      # is the region special, or is any edit visible?
perturbation  = max(A_nuis, A_bg, A_sham)
```

Plus the conventional readouts: `original_accuracy`, `silence_answer_change`,
`option_swap_consistency`, `option_swap_both_correct`.

Intervals come from a bootstrap that resamples **source clips** (ADR-011). Every
clipped item is one clip, so the resampling unit is the item.

## 8. Gates

A run is only interpretable if all three pass. This is ADR-009's three-branch
rule with the missing fourth state named.

| Gate | Criterion | If it fails |
|---|---|---|
| G-competence | `original_accuracy >= 0.75` and `silence_answer_change >= 0.20` | instrument invalid for that model; no claim about evidence use |
| G-sham | `A_sham <= 0.05` | pipeline is perturbing audio; fix before reporting anything |
| G-knob | a difficulty level moves accuracy by >= 5 points | knob is inert; do not read selectivity across it (ADR-008) |

## 9. Vocabulary screening (before the main grid)

The 89 classes are not equally recognisable. A class the model cannot name in
this question format cannot carry a competence floor, and a run at chance says
nothing.

Procedure: build a screening set where the item contains exactly one event and
the two options are the true class and a random other class. Run each target
model. Keep classes with per-class accuracy above a threshold (start at 0.80),
require at least 20 classes to survive, and freeze the surviving list before the
main grid is generated.

The screening set and the confirmatory set use different RNG seeds and disjoint
event instances.

## 10. Splits and seeds

| Split | Items | Seed | Purpose |
|---|---:|---|---|
| `screen` | 89 classes × 20 | 20260912 | vocabulary selection |
| `pilot` | 300 | 20260913 | design checks, gate calibration, listening |
| `confirm` | 600 | 20260914 | reported numbers, generated after the design is frozen |

The confirmatory split is generated once, after the pilot, and is not regenerated
if the numbers disappoint. The previous attempt failed partly by revising items
after seeing results, which turns a confirmatory set into an exploratory one.

Split and condition identities live in the manifest; the analysis script refuses
to pool splits unless told to.

## 11. Difficulty knobs

Two, applied to the `pilot` and `confirm` splits as crossed levels rather than as
new splits:

1. `snr_db` for the query event: `+10, +5, 0, -5`.
2. `n_nuisance`: `1, 2, 4`.

Both are parameters of the mixer, so they move the audio and nothing else. Per
ADR-008 each knob must be shown to move accuracy before any selectivity reading
across it is reported.

## 12. Human listening check

Synthetic ground truth is exact by construction, so listening is not checking
labels. It is checking that the instrument is perceptually sane:

- background audible and not clipping;
- query event clearly audible at the calibration SNR;
- no click or seam at any edit boundary;
- the two option strings are both plausible labels for their sounds;
- the nuisance event is not confusable with either option.

Twenty items from the pilot split, three listeners, disagreements adjudicated.
Any systematic defect found here sends the design back to section 4 rather than
into a caveat.

## 13. Natural-audio transfer check

The 238 TEMPO `tacos` clips that passed the feasibility screen are run through
the same seven conditions and the same metrics, with the same code, by
substituting `background_removed` windows for the absent nuisance annotation
where a clip has fewer than two events.

This is not a second headline result. It answers one question: does whatever the
synthetic instrument measures also appear on real recorded audio with
human-authored annotations. If the two disagree, the synthetic numbers are the
ones that need explaining.

## 14. What this design still cannot claim

- Findings are about models evaluated on *this* synthesis, not about audio in
  general. Real recordings are messier than a background plus placed events.
- scaper's LUFS calibration makes a level perceptually comparable across the
  background, but "-5 dB SNR" is still not guaranteed to be equally *hard* for
  every event class. The knob validation in section 11 is what protects the
  reading, not the calibration.
- Synthesized soundscapes remove the very ambiguity that makes real annotation
  hard, so a positive result here is an upper bound on how well models do on
  natural audio.
- Condition 5's control window is event-free by construction, so it is a
  *cleaner* control than any natural corpus can offer. That is a strength of the
  causal claim and a limitation on its realism, at the same time.

## 15. Amendment 1 — corrections found while building the instrument

Recorded 2026-09-12, before any model was run on generated audio. No result
depends on the superseded text. Each item is a correction to a claim this
document made and got wrong, not a change of design.

**A1.1 Background source: MUSAN -> TUT Urban Acoustic Scenes 2018.**
Section 2 named MUSAN noise. All three OpenSLR mirrors returned zero bytes on
both ranged and plain requests, so the 11 GB tarball is not obtainable from here.
Replacement: `wetdog/TUT-urban-acoustic-scenes-2018-development-16bit` on
Hugging Face, one shard (480 MB) of real 10 s urban scene recordings. This keeps
the property that motivated MUSAN — real recorded ambience rather than synthetic
noise — and is 20x smaller. Scene labels (street, park, metro, ...) are recorded
per item so background type can be checked as a factor rather than assumed away.
Licence: the 16-bit mirror declares none; the upstream TUT 2018 development set
is AFL-3.0. Verify before any redistribution.

**A1.2 scaper does not follow the input sample rate.** Section 4 claimed the
output lands at 16 kHz because the background is 16 kHz. That is false: scaper
sets `self.sr = 44100` on the Scaper object and resamples every source into it.
`render_soundscape` now assigns `scaper_instance.sr` explicitly and takes a
`sample_rate` argument, defaulting to 16 kHz.

**A1.3 Donors must be resampled, not read natively.** Consequence of A1.2 plus
the 44.1 kHz event library: a donor read at its native rate had a sample count
that disagreed with its span in soundscape samples. The first generator run
failed with `donor span 39690 outside donor of 14400 samples` — 0.9 s at 44.1 kHz
versus 0.9 s at 16 kHz. `src/synth.py::read_audio_mono` now resamples to a target
rate, and the generator uses it for every donor.

**A1.4 `original` and `sham` must be bit-identical.** scaper writes the base at
32-bit while edited conditions were written at PCM_16, so pointing `original` at
scaper's file left the reference and the no-op control in different formats and
differing by quantisation. Every condition, including `original`, is now written
through the same PCM_16 path from the same array. Verified bit-identical on all
generated items, and pinned by
`tests/test_synth_integration.py::test_sham_is_bit_identical_to_original`.

A1.2 through A1.4 were found by running the generator end to end on a throwaway
corpus before the real library arrived; that path is now a test
(`tests/test_synth_integration.py`), so they cannot regress silently.

## 16. Amendment 2 — corrections found on the real corpus

Recorded 2026-09-12, still before any model was run on generated audio.

**A2.1 Loud event classes clipped the mix.** scaper's `generate()` defaults
`peak_normalization=False`, and classes such as gunshots drove the render past
full scale: measured at roughly 30% of attempts rejected under the "rendered
audio clips" guard. Turning normalisation on fixes the clipping but scales every
mix to exactly 1.0, which is hotter than the models are usually fed and is
indistinguishable from real clipping to an audit. `render_soundscape` now enables
`peak_normalization` and then applies one gain to land every item at
`DEFAULT_TARGET_PEAK = 0.90`. A single gain leaves the background-to-event ratio
— the only level relationship the design depends on — untouched. Measured after
the change: 150/152 attempts accepted (1.3% rejected, all for donor availability)
and every rendered peak exactly 0.9000.

**A2.2 The sham is not bit-identical, and should not be.** A1.4 said it was, on
the strength of the smoke corpus. On the real corpus it differs by up to **1 LSB
on about 6 samples out of 160000**. That is `(1-w)*x + w*x` not reproducing `x`
exactly in float32, not an artifact. The assertion was too strong and has been
replaced by a bound: `test_sham_is_imperceptible` requires at most 2 LSB, which
separates rounding from perturbation — the withdrawn pipeline's reversed tail
produced roughly 30000 LSB. The sham still runs the real crossfade path on
purpose; short-circuiting it to an identity would make the control vacuous.

**A2.3 Donors are only required by the replacement conditions.** Screening
renders `original` and `silence`, but the generator demanded a duration-matched
donor for every item anyway and rejected those that had none. Screening yield
went from 98.7% to 100% (200/200) once donor selection was gated on the
conditions actually being rendered.

**A2.4 Corpus composition.** Of the 89 FSD50K leaf classes, 84 are kept.
Excluded: `Wind`, `Subway_and_metro_and_underground`, and the three speech
classes, because they plausibly occur *inside* an urban scene recording, which
would make "which of these two is present" ambiguous for reasons unrelated to
the model. Class names are converted to question text by rule plus an explicit
override table (`scripts/prepare_synth_corpus.py`), never by a language model,
per ADR-010; the mapping is frozen in `corpus.json` so question text cannot drift
from the label. Event instances are split: the pilot grid reads
`foreground_train` (56 labels, 5056 events), the confirmatory split reads
`foreground_val` (69 labels, 1944 events), and screening reads `foreground_all`
(84 labels, 9582 events) because screening asks about naming, not about the
items the grid will use.

**A2.5 Background scene coverage is currently a single scene.** The first TUT
shard is entirely `airport` (252 recordings). Further shards are downloading.
Until they land, every item shares one scene, which is a controlled factor rather
than a confound for the within-item contrast but does limit external validity.
The scene is recorded per item so this can be checked rather than assumed.
