# Scripts

Partitioned so a reader does not have to guess which ones produced anything
trustworthy. The withdrawn pipeline is in [`../withdrawn/`](../withdrawn/); it is
not listed here.

## Live — the current instrument

| Script | What it does |
|---|---|
| `prepare_synth_corpus.py` | builds the scaper corpus from FSD50K single-label events and TUT scene recordings; maps ontology names to question text; instance-disjoint train/val/all views |
| `build_synth_manifests.py` | generates the seven-condition grid; refuses items without a matched control |
| `synth_audit.py` | mechanical audit against the I1–I4 invariants, before any model sees the audio |
| `synth_screen.py` | vocabulary screening gate; writes the frozen class list |
| `synth_attribution.py` | change-rate metrics and the competence-free contrasts; derives the condition set from the data |
| `synth_counterfactual_follow.py` | capability-conditioned `follow(diff)` (which is diff-condition accuracy under a directional name; asserted at run time) plus both parameterisations of the `(2a-1)^2` null, the independence diagnostic, and pair-clustered CIs |
| `synth_confound_check.py` | recomputes the contrast on items matched for acoustic change |
| `synth_class_profile.py` | per-class accuracy/selectivity, plus whether selectivity is just accuracy in disguise |
| `difficulty_knobs.py` | the ADR-008 table: does the SNR / polyphony knob move accuracy before selectivity is read across it. Reads the `k_*` splits for accuracy and the `snr*_full` splits for selectivity — they are different files, see the module docstring |
| `compare_runs.py` | side-by-side diff of two runs on the same grid, grouped by pair_id; used to check that an instrument change touched only the conditions it should |
| `background_artifact.py` | measures the replacement's noise-floor artefact and tests whether models exploit it |
| `consolidate_evidence.py` | freezes the paper's numbers into `outputs/evidence/` with source SHA256 |
| `make_listening_package.py` | the human perceptual check, as a self-contained HTML page |

## Live — the natural-audio route

Kept because its *methodology* is the result, not its numbers: it is the worked
example of a transfer check whose truth premise was false.

| Script | What it does |
|---|---|
| `build_natural_manifests.py` | seven conditions on TEMPO tacos; matched control is `nuisance_replaced` (ADR-013) |
| `check_natural_donors.py` | are the natural donors nameable in their home clips |
| `probe_natural_target_persistence.py` | **the check that caught it**: is the class still present after its evidence was replaced |

## Live — shared infrastructure

| Script | What it does |
|---|---|
| `evaluate_manifest.py` | batch inference; families `qwen25-omni-3b`, `audio-flamingo-3`, `fun-audio-chat-8b`; infers the condition set from the manifest rather than a hardcoded list |
| `remote_eval.sh` | one process per GPU in the `dl-tasks` tmux session, with `START`/`PIPE_RC`/`__DONE__` markers |
| `rebase_manifest_audio.py` | rewrites absolute audio paths for the GPU host |
| `feasibility_scan.py` | annotation-level gate: how many clips can host a local item with a matched control |
| `inspect_dataset.py`, `audit_parquet.py` | dataset reconnaissance |
| `merge_prediction_shards.py` | merge `--pair-shards` output |
| `remote_*_setup.sh`, `remote_*_aria2.sh`, `remote_wait_eval.sh` | environment setup, model download, run monitoring |

## Note on `synth_class_profile.py`

It reports the across-class correlation between accuracy and selectivity, which
came out at 0.78 / 0.77 for the two competent models. That is the script that
shows the class-level dissociation does *not* exist, so it is worth running
before anyone claims a class is "recognised but not used as evidence". (An
earlier revision of this file quoted 0.86 / 0.91; those came from the
`source_pair_id` view-keying defect described in the counterfactual-follow entry
above, which silently used one surface order per item. Treat the correlation as
evidence against dissociation, not as a causal estimate — `selectivity` is
defined as `A_diff - A_same`, so part of the relationship is built in.)
