# Step 3 input map

Paths below are relative to the staging root. No analyses were run.

| Task | Inputs |
|---|---|
| Figures 2–5 | `data/figure_inputs/model_performance/` |
| Scaffold plot | `data/figure_inputs/scaffold/` |
| Scaffold numerical analysis | `data/scaffold/single_protein/`, `data/scaffold/cell_based/`; five-representation prediction roots under the existing repository |
| Ablation comparison | `data/results/ablation/` |
| Optional ablation retraining | `data/modeling/` and original full-model bundles in the existing repository |
| Saved SHAP subgroup analysis | `data/shap/`, `data/scaffold/` and original predictions in the existing repository |
| Both SHAP figures | `data/figure_inputs/shap/` |
| Morgan plotting | `data/chemistry/reference_results/` |
| Morgan verification | `--component-root data/chemistry` |
| Morgan recomputation | `--root data/chemistry/analysis_inputs`, structure file in `data/chemistry/reference_results/input/` |

Adapted ablation and integrated SHAP scripts accept `--repository-root`, `--data-root` and a required new `--output-dir`.
The defaults locate `data/` inside staging and the existing repository beside staging.
The two SHAP generators already accept explicit model/development/external/output paths;
their `--project` must be the corresponding `post_model_analysis/shap/<assay>` folder.
For full external explanations, explicitly use 2524 single-protein or 192 cell-based rows,
100 background rows, seed 42 and two permutations; defaults are smaller pilot settings.
New SHAP runs save row-index provenance. This does not retrospectively recover historical samples.

Frozen masters include all molecular columns; ablation/SHAP use each saved bundle's selected
feature list and ordering. This avoids regenerating 20 representation files during packaging.
Unchanged original staged code is backed up under `STEP3_ORIGINAL_STAGED_CODE/`.
Do not upload that backup as active source code.

Next: Step 4 runtime checks against these inputs. Syntax parsing alone does not establish
that the integrated analysis commands reproduce every output. No publishing yet.
