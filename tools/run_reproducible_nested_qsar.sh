#!/usr/bin/env bash
set -euo pipefail

# Portable EGFR QSAR nested-CV runner
#
# Usage:
# bash tools/run_reproducible_nested_qsar.sh \
#   PROJECT DATA_ROOT OUTPUT_ROOT [--dry-run]
#
# PROJECT: single_protein or cell_based
# DATA_ROOT: reconstructed modeling datasets
# OUTPUT_ROOT: separate directory for fresh results

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 PROJECT DATA_ROOT OUTPUT_ROOT [--dry-run]"
    exit 1
fi

PROJECT="$1"
DATA_ROOT="$2"
OUTPUT_ROOT="$3"
DRY_RUN=false

if [[ $# -eq 4 ]]; then
    if [[ "$4" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        echo "ERROR: Unknown option $4" >&2
        exit 1
    fi
fi

if [[ "$PROJECT" != "single_protein" &&
      "$PROJECT" != "cell_based" ]]; then
    echo "ERROR: Invalid project: $PROJECT" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_SCRIPTS="$REPO_ROOT/$PROJECT/scripts"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEARCH_ITERATIONS="${SEARCH_ITERATIONS:-12}"
N_JOBS="${N_JOBS:-2}"

RESULTS_ROOT="$OUTPUT_ROOT/$PROJECT/full_cutoff_7"

if [[ "$PROJECT" == "single_protein" ]]; then
    ANALYSIS="primary_full_cutoff_7_nested"
else
    ANALYSIS="cell_primary_full_cutoff_7_nested"
fi

REPS=(2d 3d morgan 2d_3d 2d_3d_morgan)

filename_for_rep() {
    case "$1" in
        2d)
            echo "1_all_2d_descriptors_with_metadata.csv" ;;
        3d)
            echo "2_all_3d_descriptors_with_metadata.csv" ;;
        morgan)
            echo "3_rdkit_morgan_fingerprints_with_metadata.csv" ;;
        2d_3d)
            echo "4_all_2d_3d_descriptors_with_metadata.csv" ;;
        2d_3d_morgan)
            echo "5_all_features_2d_3d_morgan_with_metadata.csv" ;;
    esac
}

echo "========================================"
echo "EGFR QSAR PORTABLE RUNNER"
echo "========================================"
echo "Project:       $PROJECT"
echo "Data root:     $DATA_ROOT"
echo "Output root:   $RESULTS_ROOT"
echo "Search trials: $SEARCH_ITERATIONS"
echo "Parallel jobs: $N_JOBS"
echo "Dry run:       $DRY_RUN"
echo

# Check required scripts
for script in \
    qsar_core.py \
    nested_grouped_cv.py \
    compare_representations.py
do
    if [[ ! -f "$PROJECT_SCRIPTS/$script" ]]; then
        echo "MISSING SCRIPT: $script"
        exit 1
    fi
done

# Verify all input datasets before training
MISSING=0

for REP in "${REPS[@]}"; do
    FILE="$(filename_for_rep "$REP")"

    for SPLIT in development external; do
        INPUT="$DATA_ROOT/$PROJECT/$SPLIT/$FILE"

        if [[ -f "$INPUT" ]]; then
            echo "FOUND: $REP / $SPLIT"
        else
            echo "MISSING: $INPUT"
            MISSING=$((MISSING + 1))
        fi
    done
done

echo
echo "Missing datasets: $MISSING"

if [[ "$MISSING" -gt 0 ]]; then
    echo "Reconstruct the missing modeling matrices first."
    exit 2
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN PASSED"
    echo "All required input files and scripts exist."
    echo "No model training performed."
    exit 0
fi

# Never overwrite existing run results
if [[ -e "$RESULTS_ROOT" ]]; then
    echo "ERROR: Results directory already exists:"
    echo "$RESULTS_ROOT"
    echo "Choose a new output directory."
    exit 1
fi

mkdir -p "$RESULTS_ROOT"

for REP in "${REPS[@]}"; do
    FILE="$(filename_for_rep "$REP")"

    DEV="$DATA_ROOT/$PROJECT/development/$FILE"
    EXT="$DATA_ROOT/$PROJECT/external/$FILE"
    OUT="$RESULTS_ROOT/$REP"

    echo "========================================"
    echo "Training representation: $REP"
    echo "========================================"

    "$PYTHON_BIN" -u \
        "$PROJECT_SCRIPTS/nested_grouped_cv.py" \
        --development "$DEV" \
        --external "$EXT" \
        --output-dir "$OUT" \
        --representation "$REP" \
        --analysis-name "$ANALYSIS" \
        --core-script \
        "$PROJECT_SCRIPTS/qsar_core.py" \
        --budgets 50 100 200 300 500 \
        --models logistic svm_rbf random_forest extra_trees lightgbm xgboost knn \
        --outer-folds 5 \
        --inner-folds 3 \
        --search-iterations "$SEARCH_ITERATIONS" \
        --missing-threshold 0.30 \
        --correlation-threshold 0.90 \
        --fdr-alpha 0.05 \
        --threshold 0.50 \
        --ad-column Inside_AD \
        --bootstrap 1000 \
        --random-state 42 \
        --n-jobs "$N_JOBS" \
        --dpi 300 \
        2>&1 | tee "$RESULTS_ROOT/${REP}_training.log"
done

"$PYTHON_BIN" \
    "$PROJECT_SCRIPTS/compare_representations.py" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$RESULTS_ROOT/comparison" \
    --dpi 300

echo "EGFR QSAR pipeline completed successfully."
