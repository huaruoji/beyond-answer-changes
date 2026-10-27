# Reference figure data

| Figure | Source | Values | Type | Generator | Outputs |
|---|---|---|---|---|---|
| reference_identity | ../../outputs/evidence/evidence_table.json | models.{r2_qwen7b,r2_qwen3b,fun_audio_chat_8b}.attribution.{A_diff,A_same,A_nuisance_removed,A_background_removed,A_silence,A_sham}, multiplied by 100; the rightmost group is .attribution.selectivity with its .attribution.selectivity_ci95 interval | Real model predictions on synthesized audio; no mock data | ../make_reference_figures.py | reference_identity.pdf, .svg, .png |
| reference_class | Predictions at each model's follow.predictions path in the frozen evidence table | Group confirm trials by query_class; original accuracy versus mean of different-class change minus same-class change | Real model predictions, 33 classes per model, 1,200 trials | ../make_reference_figures.py | reference_class.pdf, .svg, .png |
| reference_distribution | Same frozen predictions and class profiles as reference_class | Empirical cumulative distribution of the 33 class-level gaps, with equal class weights and no smoothing | Real model predictions; no mock data | ../make_reference_figures.py | reference_distribution.pdf, .svg, .png |

Descriptive rates over 1,200 trials per model (600 soundscapes x 2 option orders). Paired trials share a
soundscape. The only interval shown is on the change-rate gap in `reference_identity`, and it resamples
**soundscapes**: a soundscape contributes its original render and all six interventions as one paired unit,
and its two option orders move together. The split reuses 458 background recordings across 600 soundscapes (at most
eight share one) and does not record donor recordings, so the interval is clustered at the soundscape
level and does not account for source-recording reuse across soundscapes.

Both `reference_class` and `reference_distribution` are included in the manuscript. The generator prints
the per-model minimum, median, maximum and negative count so the claim stays checkable, and asserts 33
classes, 1,200 trials, and that the trial-weighted class gap equals `A_diff - A_same` in the frozen table.

Verify source hashes and underlying predictions with ../verify_reference_draft.py. Full model names; Times
New Roman at 10 pt (9 pt single-column legends); 450 dpi PNG; vector PDF/SVG. The bar chart is 7 inches
wide and 2.1 inches tall; the scatter and cumulative distribution are each 3.39 inches wide. Blue, pale blue and muted brick consistently identify models;
marker shapes and line styles distinguish them without hatch fills. The generator checks text bounds before saving.

The generator needs Times New Roman. On this machine it is installed from the Windows font directory
(`cp /mnt/c/Windows/Fonts/times*.ttf ~/.local/share/fonts/ && fc-cache -f`); without it the script falls
back to a metric-compatible serif and says so, which is fine for layout checks but not for submission.
