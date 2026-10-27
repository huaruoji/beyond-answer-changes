# Model Landscape and Measured Behaviour

Last checked: 2026-09-22. This file records verified deployment facts and, since
the instrument now produces numbers, what each model actually did on it. The
pre-2026-09-12 version was a pure download-feasibility audit for the withdrawn
MVP. Both tables below now aggregate the same way: every surface order counted as
its own view, 1200 views over 600 source pairs (`synth_attribution.py` and
`synth_counterfactual_follow.py`). An earlier revision of this file warned that
their `original accuracy` columns were different quantities; that was a symptom
of a loader defect in `synth_counterfactual_follow.py`, which kept only the last
surface order per item and silently discarded the other. The defect is fixed and
the columns now agree.

## What the deployed models did

| Model | Synthetic confirm split, original accuracy | Passes the competence gate (>=0.75) | Note |
|---|---:|---|---|
| Qwen2.5-Omni-7B | 0.882 | yes | selectivity +0.609 [+0.573, +0.646] |
| Qwen2.5-Omni-3B | 0.853 | yes | selectivity +0.578 [+0.540, +0.617] |
| Audio Flamingo 3 | 0.704 | **no, on every split and SNR level** | also strongly position-biased: A-share 0.782 against 0.58 / 0.61 for the two Qwen models |
| Fun-Audio-Chat-8B | 0.847 | yes | different architecture family; follow(diff) E2 0.738 (v2 grid), sham floor 0.0017 |

AF3 never clears the gate. Its position bias replicates the withdrawn MVP's
finding that AF3 was the most position-bound of the models tested (option-swap
consistency 0.152), which suggests that is a property of the model rather than of
the old pipeline.

## Capability-conditioned counterfactual following (the current headline)

The change rate is confounded by the metric's own construction; a model that
ignores the evidence still diverges `(2a-1)^2` of the time. The directional
alternative, computed by `scripts/synth_counterfactual_follow.py`, is
`follow(diff) = 1[the model answers the donor's class after the replacement]`.
Fun-Audio-Chat-8B is the cross-architecture check: it clears the gate and follows
the counterfactual at the same order of magnitude as the Qwen models.

**`follow(diff)` is per-condition accuracy under a directional name.** The
different-class condition's ground truth is the donor class, so it is the same
event as `accuracy(diff)`; `preserve(same)` is likewise `accuracy(same)`. The
script asserts this at run time (`metric_identity`) and the numbers below should
not be read as a second, independent measurement.

| Model | original acc. | follow(diff), all | follow(diff), E2 | null share of raw selectivity | sham floor |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Omni-3B | 0.853 | 0.838 | 0.802 | 80% | 0.009 |
| Qwen2.5-Omni-7B | 0.882 | 0.845 | 0.813 | 81% | 0.006 |
| Fun-Audio-Chat-8B | 0.847 | 0.800 | 0.738 | 71% | 0.0017 |

Both surface orders pooled, 1200 views over 600 source pairs. `preserve(same)` on
E2 is 1.0 by construction and is not a finding. Two caveats: the null's
independence premise fails (class-correctness agreement 0.85-0.89 vs 0.72-0.80
predicted), so the null share is an upper bound rather than a calibrated figure;
and there is a large surface-order effect, so confidence intervals resample
source pairs, not views. The order-effect-free estimator averages within each
pair then over pairs, giving E2 = 0.801 / 0.819 / 0.737, within 0.37 pt of the
pooled column. All rows are the **v2** grid (background preserved when replacing a
span). Numbers live in
`outputs/p0_fun_audio_chat/counterfactual_follow_3models.json`; the frozen run
config is `configs/p0_fun_audio_chat_frozen.json`.

Per-class accuracy spans 0.61-1.00 within a single model (3B, confirm split,
33 classes with at least 8 items; sd 0.098). Across classes the correlation
between accuracy and selectivity is 0.67 (3B) and 0.67 (7B), so "this class is
recognised but the model does not use it as evidence" is not a separable claim
here — see `scripts/synth_class_profile.py`. Note the correlation is partly
built in, since `selectivity = A_diff - A_same`; treat it as evidence against
dissociation, not as a causal estimate.

## Local constraint

- GPU: NVIDIA RTX 4060 Ti, 16 GB VRAM.
- WSL ext4 free: about 645 GB; C: free: about 35 GB.
- No audio LLM is currently cached.
- First-pass candidates must fit one card using documented quantization or be
  replaced by a smaller family member.

The constrained local card is not used for model weights. A remote GPU host
offered six idle RTX 5880 Ada 48 GB cards and about 1.3 TB of scratch space, so
remote model artifacts stay off the constrained local disk.

## Candidates under audit

| Model | Family / role | Official weights | License | Native general sound? | Timestamp/reasoning? | 16 GB feasibility | Status |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Omni-3B | Qwen / small general omni | `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, 12.0 GB | Qwen Research | Yes; official MMAU 63.30% avg | reasoning prompt; no native evidence output guarantee | 13.3 GiB observed on 48 GB card with talker disabled | wrapper and direct/CoT/timestamp inference complete |
| Qwen2.5-Omni-7B | Qwen / widely used baseline | audit pending | audit pending | Yes | used by Omni-R1, SARI, faithfulness, AHA | quantized only likely | Priority A if 3B behavior is usable |
| Qwen3-Omni | Qwen / strong current baseline | official 30B-A3B Instruct/Thinking weights | Apache-2.0 | Yes | explicit Thinking variant | 30B total is not a credible 16 GB first run | Remote/later only |
| Audio Flamingo 3 | NVIDIA/UMD / distinct family | `7d4bae64ee29878af6504ae6f6bb3e40492838ad`; 16.5 GB base shards plus 1.4 GB Think adapter (HF repo also contains a duplicate 16.5 GB consolidated file) | NVIDIA OneWay Noncommercial | Yes | AF-Think | BF16 does not fit safely; test 4-bit after Qwen | Priority B |
| Audio Flamingo Next | NVIDIA/UMD / distinct family | repository/size still unresolved | non-commercial expected; verify | Yes | Temporal Audio CoT | likely remote/quantized | Hold |
| Kimi-Audio | Moonshot / distinct family | audit pending | audit pending | Yes | general reasoning | likely too large; verify | Priority C |
| MiMo-Audio-7B-Instruct | Xiaomi / current ADQA baseline | `c359441c22c2a1c74be5f99a91e83392680e9cc8`; 16.1 GB | MIT | Yes | few-shot audio QA | custom CUDA>=12/flash-attn path; BF16 unsafe at 16 GB | Priority C |
| Fun-Audio-Chat-8B | FunAudioLLM / voice-chat baseline | `7bf72dc7c705493f817178a0859efff91e9cf73c`; 18.9 GB | Apache-2.0 | Yes | chat | ~18.6 GiB observed on a 24 GB RTX 4090 in BF16 | **run on confirm**; see `outputs/p0_fun_audio_chat/` and `configs/p0_fun_audio_chat_frozen.json`. Needs its own repo code (`register_funaudiochat`) and `transformers==4.52.3`, `numpy<2` |

Selection is deliberately not final until official Hugging Face/GitHub model
cards, revisions, dependencies, and measured peak VRAM are recorded.

## Verified official constraints

- Qwen's card reports 18.38 GB minimum BF16 memory for a 15-second **video**
  input and warns actual usage is usually at least 1.2×. Audio-only text output
  is smaller and `disable_talker()` saves about 2 GB, but fit is an experiment,
  not an assumption.
- Fun-Audio-Chat explicitly reports about 24 GB for inference and 4×80 GB for
  training.
- MiMo requires Python 3.12, CUDA 12+, and FlashAttention 2.7.4 in its official
  reference setup.
- AF3's official HF repository accidentally/temporarily exposes both sharded and
  consolidated copies of the same base weights; selective include/exclude is
  mandatory to avoid a redundant 16.5 GB download.

## Measured MVP deployment

Both AF3 and Qwen2.5-Omni-3B ran in BF16 on one RTX 5880 Ada 48 GB card per
worker. Qwen direct/CoT workers peaked at approximately 13.3 GiB resident
memory. All weights, generated audio and raw prediction JSONL stayed on the remote
scratch disk. The constrained local filesystem holds only code and small metric
artifacts.

## Measured third-model deployment

Fun-Audio-Chat-8B ran in BF16 on one RTX 4090 (24 GB) per shard on the `QRH`
host, one process per card, two shards, complete by resume-friendly sharding.
Observed resident memory 18.6 GiB, close to the card limit, so a single
unshared card is required. Throughput was about 3-4 records/s per card on 10 s
16 kHz clips, so the 8400-record confirm split finished in well under an hour of
wall time on two cards.
