# EGFR QSAR: Nested Grouped Cross-Validation and External Validation

## Overview

This repository contains reproducibility materials for machine-learning
models developed to classify EGFR inhibitors using single-protein and
cell-based bioactivity datasets.

The release includes nested cross-validation code, trained models,
feature-selection records, external predictions, applicability-domain
analyses, and comparison results.

## Datasets

| Dataset | Development records | External records |
|---|---:|---:|
| Single-protein | 5,093 | 2,524 |
| Cell-based | 3,164 | 192 |

Activity classification:
- Active: pIC50 > 7.0
- Inactive: pIC50 <= 7.0

Development and external cohorts have no overlapping standardized
compound identities.

## Molecular representations

Five representations were evaluated independently:

1. 2D molecular descriptors
2. 3D molecular descriptors
3. Morgan fingerprints
4. Combined 2D and 3D descriptors
5. Combined 2D, 3D, and Morgan fingerprints

## Machine-learning methodology

The analysis used nested stratified compound-grouped cross-validation
with five outer folds and three inner folds.

Seven algorithms were evaluated: logistic regression, SVM-RBF,
random forest, Extra Trees, LightGBM, XGBoost, and KNN.

Feature budgets were 50, 100, 200, 300, and 500.

Model selection was based on Matthews correlation coefficient (MCC).
Final selected configurations were fitted on the development data
and evaluated on source-held-out external datasets.

## Repository organization

- single_protein/scripts/: single-protein modeling and comparison code
- single_protein/results/: saved models and analysis results
- cell_based/scripts/: cell-based modeling and comparison code
- cell_based/results/: saved models and analysis results
- tools/reconstruct_modeling_matrices.py: frozen-data reconstruction
- tools/run_reproducible_nested_qsar.sh: portable modeling runner
- requirements.txt: current verified package versions

The main nested-CV workflow is nested_grouped_cv.py.
The qsar_core.py module provides reusable modeling utilities.
The compare_representations.py script regenerates comparison summaries.

## Software environment

The current verified environment uses Python 3.8.10.

Install dependencies with:

    python -m pip install -r requirements.txt

The recorded versions describe the verified current environment
and are not independently confirmed as the exact original
training environment.

## Frozen modeling datasets

The four master feature matrices are stored separately in:

EGFR_QSAR_FROZEN_DATA.zip

The archive also contains representation-column metadata.
Its SHA-256 checksum should be verified before reconstruction.

Data archive DOI or URL: TO BE ADDED

To inspect reconstruction options:

    python tools/reconstruct_modeling_matrices.py --help

## Running the pipeline

Example input validation without training:

    bash tools/run_reproducible_nested_qsar.sh single_protein /path/to/reconstructed_data /path/to/new_results --dry-run

Use cell_based instead of single_protein for the cell-based analysis.

Removing --dry-run launches computationally expensive nested-CV training.
Full retraining has not been verified using this release.

## Saved models and results

Five final trained models are provided for each assay dataset.

Each representation includes relevant model-selection records,
nested-CV metrics, selected features, strict external predictions,
and validation summaries.

Only load joblib model files from trusted sources.

## Reproducibility verification

The following checks were completed:

- Frozen-data archive and checksums verified
- All 20 modeling matrices reconstructed
- Ten saved-model prediction outputs reproduced
- Both portable runners passed input dry-runs
- All eight comparison tables regenerated successfully
- Python and shell syntax checks passed

Full nested-CV retraining was not repeated for this release.

Reproducibility currently begins from frozen molecular-feature
matrices rather than raw ChEMBL records. The complete upstream
cell-based token-supported preprocessing workflow remains to be
documented and independently verified.

## Data availability

The frozen modeling datasets and post-model analysis results are
available on Zenodo:
[10.5281/zenodo.23886017](https://doi.org/10.5281/zenodo.23886017).

## Citation

Associated manuscript citation will be added upon publication.

## License

The GitHub code license still needs to be selected and added as a LICENSE file. The dataset’s CC BY-SA 4.0 license does not automatically license the code.

## Complete dataset reconstruction

Download EGFR_QSAR_FROZEN_DATA.zip from the associated data
repository (link to be added after deposition).

Verify its integrity using the accompanying checksum file.
Run this command from the directory containing both files:

    sha256sum -c EGFR_QSAR_FROZEN_DATA.zip.sha256

The expected archive SHA-256 is:

    7ca27c87166ef33ba1c5ee74f6136ea125474a96543bf31d753d921c38194fd4

From the root of this code repository, reconstruct all modeling
representations using:

    DATA_ARCHIVE="/path/to/EGFR_QSAR_FROZEN_DATA.zip"
    DATA_ROOT="/path/to/EGFR_QSAR_RECONSTRUCTED"

    for project in single_protein cell_based; do
        for split in development external; do
            python tools/reconstruct_modeling_matrices.py                 --archive "$DATA_ARCHIVE"                 --output "$DATA_ROOT"                 --project "$project"                 --split "$split"                 --representations 2d 3d morgan 2d_3d 2d_3d_morgan
        done
    done

This generates five feature representations for each of the
two assay datasets and their development/external splits
(20 modeling CSV files in total).

The master archive contains four frozen datasets plus
representation_columns.json, which records feature-column
membership and ordering.

After reconstruction, check the modeling inputs:

    bash tools/run_reproducible_nested_qsar.sh         single_protein "$DATA_ROOT" /path/to/new_results --dry-run

    bash tools/run_reproducible_nested_qsar.sh         cell_based "$DATA_ROOT" /path/to/new_results --dry-run

These dry runs check required files without training models.

The complete nested-CV retraining pipeline is available, but
it has not been independently rerun using this release.

Historical run_configuration.json and holdout_ad_manifest.json
files contain redacted local path placeholders. They preserve
the original experiment settings but should not be used as
direct executable input paths.
