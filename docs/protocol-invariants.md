# Protocol invariants

These rules make the reported numbers interpretable. They were fixed before the
confirm-split evaluation, and the tests in `tests/` lock the ones marked I1 to I4.

## 2.1 Estimand and environment

- The estimand is paired at the item level. One source pair contributes two
  option orders of the same waveform, so the pair is the clustering unit and the
  item is the pairing unit.
- No module may `import scaper` directly. Use `src/synth.py`, which owns the
  environment contract, a NumPy 2 shim for `np.Inf` and the sox `PATH` and
  `LD_LIBRARY_PATH` conventions.
- Scaper hardcodes `self.sr = 44100` on the `Scaper` object and does not follow
  the input sample rate. Set it explicitly before `generate()`, and resample
  donors to the target rate.

## 2.2 Hard edit invariants

| ID | Criterion |
|---|---|
| I1 | A donor waveform comes only from its annotated `[start, end)`, never the whole clip |
| I2 | Target and control intervals do not overlap, `gap >= 0.2 s` |
| I3 | Edit duration matches, `abs(c.span - t.span) / t.span <= 0.25`, and the post-edit interval RMS differs by at most 1 dB |
| I4 | Edit boundaries add no transient, first differences stay within three times the original at the same position |

Instrument extension, used by the controllability work:

| ID | Criterion |
|---|---|
| I5 | A masker gain acts on the background only and never changes the sample interval of a foreground event |
| I6 | No condition-dependent peak normalisation after mixing |
| I7 | The array used for the auditory score is bit-identical to the array written for the model |

## 2.3 Level rules

1. Calibrate the level of the target and of the masker separately, before mixing.
2. Do not normalise the mix to a fixed peak. Low-frequency noise then dominates
   the peak and pushes the mid and high frequencies down, so the difficulty knob
   stops working. If clipping must be prevented, apply one common gain to the
   whole mixture. This is the only permitted level correction.
3. `original` and `sham` must be bit-identical. A 32-bit base path and a PCM_16
   edit path once turned the sham into a real edit.

## 2.4 Methodological pre-checks

- Validate the knob before evaluating. Any difficulty knob (SNR, gain, masking)
  must be shown to move accuracy by at least 5 percentage points before it enters
  an evaluation. Otherwise the readings do not support a claim about evidence use.
- Probe the premise. Any design that infers "removing the evidence flips the
  truth" needs a direct yes or no probe first, confirming that the target class
  really is absent from the edited clip. A positively labelled corpus does not
  support that inference on its own.

## 2.5 Statistical discipline

- Bootstrap clusters on the source clip, not on the source pair. The pairs of one
  round came from fewer clips than pairs, and clustering on pairs understates the
  interval width.
- Balance A/B option order inside every condition, and cluster the analysis by
  source pair or base item.
- Parse only the assistant completion. Split on `assistant`, accept an explicit
  answer marker or a leading option letter, and never match free text broadly,
  otherwise an event description such as "a woman ..." is read as option A.
