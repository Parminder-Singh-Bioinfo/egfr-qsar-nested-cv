# EGFR post-model analyses and manuscript visualization

Add this `post_model_release/` directory to the existing modeling repository.
Preserve its existing `single_protein/`, `cell_based/`, `tools/` and root README.
This addition does not replace the original modeling release.

## Data setup

The five small model-performance plotting CSVs are included here. The other
post-model inputs are supplied in `EGFR_POST_MODEL_DATA_AND_RESULTS.zip`.
Extract that archive at the existing repository root: its members already start
with `post_model_release/`. Verify archive hashes before extracting.

Retain the existing `EGFR_QSAR_FROZEN_DATA.zip` on Zenodo. Its four master matrices
are not duplicated in the new archive. From the existing repository root:

```bash
python post_model_release/tools/unpack_frozen_masters.py --archive /path/to/EGFR_QSAR_FROZEN_DATA.zip
```

Use the existing verified Python 3.8.10 environment and requirements. A clean
dependency-install test remains distinct from the successful existing-environment replay.

## Regenerate Figures 2–5 without training

```bash
python post_model_release/visualization/model_performance/render_all.py \
  --input-dir post_model_release/data/figure_inputs/model_performance \
  --output-dir post_model_release/generated_figures/model_performance
```

Figures 3 and 5 use average precision (AP). Their old numeric column names are
retained for compatibility; values were calculated with average_precision_score.

## Repeat the complete saved-result checks

After extracting both data archives, from the existing repository root:

```bash
python post_model_release/verification/run_step4_saved_result_checks.py \
  --staging post_model_release --repository-root .
```

This uses explicit repository/data locations, regenerates plots, recalculates
scaffold and SHAP subgroup summaries and verifies saved ablation and Morgan
results. It does NOT train models, rerun nested CV or calculate new SHAP values.
Outputs go to a new timestamped `post_model_release/verification_runs/` folder.

## Optional computational stages

The ablation refit script retrains only the selected configurations on development
data. The original SHAP generators calculate new explanations and can be costly.
They are not invoked by the saved-result checker. Integrated SHAP/ablation scripts
accept `--repository-root` and `--data-root`; pass them explicitly when running
from this combined repository layout. See `STEP3_INPUT_PATHS.md`.

## Scope and provenance

The staged code passed the saved-result runtime checks recorded in verification/.
The user confirmed visual review. Existing scaffold assignments are the starting
point for scaffold replay; no new scaffold generator is claimed. Historical SHAP
background row IDs are not claimed recovered; future generator runs save sampled
row indices. The existing ten full models are reused from the original repository.
Full nested-CV training and upstream raw-record/feature-generation replay were not
repeated. Figure 3 styling was reconstructed from the supplied manuscript raster.

Final figure numbering for the later matched plots belongs in the manuscript index;
the scripts retain descriptive output names. No new figure numbering is inferred here.
