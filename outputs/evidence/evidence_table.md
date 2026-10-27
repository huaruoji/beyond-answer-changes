# Frozen evidence table (confirm split)

Generated 2026-09-24T10:23:10+00:00 from the JSONs whose SHA256 are
recorded in `evidence_table.json`. Regenerate with
`python scripts/consolidate_evidence.py`; do not hand-edit.

All rows are the **v2** grid (the replacement preserves the clip's background). Audio Flamingo 3 is the exception: it fails the competence gate, its selectivity is not read, and its attribution is the pre-fix run (informational only).

## Table A -- seven-condition selectivity (two-order view, n=1200)

| Model | family | original acc. | A_diff | A_same | A_nuis | A_bg | A_silence | A_sham | selectivity (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-Omni-7B | Qwen2.5-Omni | 0.882 | 0.732 | 0.122 | 0.055 | 0.013 | 0.444 | 0.006 | **0.609** [+0.573, +0.646] |
| Qwen2.5-Omni-3B | Qwen2.5-Omni | 0.853 | 0.703 | 0.125 | 0.071 | 0.015 | 0.427 | 0.009 | **0.578** [+0.540, +0.617] |
| Audio Flamingo 3 | NVIDIA/UMD | 0.704 | 0.408 | 0.098 | 0.077 | 0.027 | 0.214 | 0.009 | **0.311** [+0.279, +0.343] |
| Fun-Audio-Chat-8B | FunAudioLLM | 0.847 | 0.653 | 0.157 | 0.075 | 0.011 | 0.460 | 0.002 | **0.497** [+0.453, +0.539] |

Audio Flamingo 3 fails the competence gate (original acc. < 0.75), so its selectivity is not read.

## Table B -- capability-conditioned follow(diff) (pooled two-order view, n=1200)

| Model | original acc. | follow(diff) all | follow(diff) E2 | pair-mean E2 | null share of raw selectivity | sham floor |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Omni-7B | 0.882 | 0.845 [0.821, 0.869] | 0.813 (n=971) | 0.819 | 81% | 0.0058 |
| Qwen2.5-Omni-3B | 0.853 | 0.838 [0.815, 0.862] | 0.802 (n=941) | 0.801 | 80% | 0.0092 |
| Fun-Audio-Chat-8B | 0.847 | 0.800 [0.771, 0.828] | 0.738 (n=891) | 0.737 | 71% | 0.0017 |

`preserve(same)` is 1.000 on E2/E3 by construction and is not a finding. `follow(diff)` is identically `accuracy(diff)` -- the different-class condition's ground truth is the donor class -- so it is a directional name for per-condition accuracy, not a second measurement. The null share is the per-condition parameterisation; its independence premise does not hold, so it is an upper bound. The pair-mean column is the surface-order-free estimator and is the one to quote.

## Table C -- acoustic-distance-matched contrast

| Model | median d(diff) | median d(same) | matched n | matched selectivity (95% CI) |
|---|---:|---:|---:|---|
| Qwen2.5-Omni-7B | 3.098 | 2.575 | 236 | **0.657** [+0.585, +0.725] |
| Qwen2.5-Omni-3B | 3.098 | 2.575 | 236 | **0.564** [+0.483, +0.640] |

## Table D -- class-level dissociation test

| Model | classes | acc. spread (min/med/max) | acc. vs selectivity | acc. vs A_same |
|---|---:|---|---:|---:|
| Qwen2.5-Omni-7B | 33 | 0.714 / 0.880 / 0.981 | 0.668 | -0.579 |
| Qwen2.5-Omni-3B | 33 | 0.611 / 0.857 / 1.000 | 0.672 | -0.534 |

## Table E -- acoustic-distance contrast by option-order trial

Aggregation: per-option-order view, bootstrap resampled by soundscape

| Model | matched soundscapes | matched trials | selective accuracy (95% CI) |
|---|---:|---:|---|
| Qwen2.5-Omni-7B | 236 | 472 | **0.617** [+0.555, +0.678] |
| Qwen2.5-Omni-3B | 236 | 472 | **0.530** [+0.462, +0.597] |

