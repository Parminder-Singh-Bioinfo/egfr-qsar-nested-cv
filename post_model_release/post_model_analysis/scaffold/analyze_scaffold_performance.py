
#!/usr/bin/env python3
"""
EGFR QSAR: Scaffold generalization across all 10 final models.

Datasets:
    Single-protein: 5 molecular representations
    Cell-based:     5 molecular representations

Scaffold assignments are compound-level (InChIKey_STD).
Predictions remain observation-level (compound + mutation).

Outputs:
    all_10_scaffold_performance.csv
    all_10_scaffold_ad_performance.csv
    all_10_scaffold_summary.csv
    original_scaffold_reproducibility_check.csv

Python 3.8 compatible.
Original input files are never modified.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

HOME = Path.home()

parser = argparse.ArgumentParser(description="Recalculate scaffold and scaffold-by-AD performance from frozen predictions. No model training.")
parser.add_argument('--single-scaffold-dir', required=True, type=Path)
parser.add_argument('--single-predictions-root', required=True, type=Path)
parser.add_argument('--cell-scaffold-dir', required=True, type=Path)
parser.add_argument('--cell-predictions-root', required=True, type=Path)
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
OUTPUT = args.output_dir.expanduser().resolve()
output_names = ['all_10_scaffold_performance.csv', 'all_10_scaffold_ad_performance.csv',
                'all_10_scaffold_summary.csv', 'original_scaffold_reproducibility_check.csv', 'analysis_manifest.json']
if any((OUTPUT / name).exists() for name in output_names):
    parser.error('An output file already exists. Choose a new output directory.')

REPS = [
    "2d",
    "3d",
    "morgan",
    "2d_3d",
    "2d_3d_morgan",
]

DATASETS = {
    "Single-protein": {
        "scaffold_dir": (
            HOME
            / "projects/egfr_source_holdout/results/"
              "scaffold_analysis/full_cutoff_7"
        ),
        "model_dir": (
            HOME
            / "projects/egfr_source_holdout/results/"
              "final_nested_qsar/full_cutoff_7"
        ),
        "models": {
            "2d": ("SVM-RBF", 300),
            "3d": ("SVM-RBF", 300),
            "morgan": ("Extra Trees", 300),
            "2d_3d": ("SVM-RBF", 500),
            "2d_3d_morgan": ("Extra Trees", 200),
        },
    },
    "Cell-based": {
        "scaffold_dir": (
            HOME
            / "projects/egfr_cell_based_source_holdout/"
              "results/scaffold_analysis/"
              "literature_multisource_token_supported/"
              "full_cutoff_7"
        ),
        "model_dir": (
            HOME
            / "projects/egfr_cell_based_source_holdout/"
              "results/final_nested_qsar/"
              "literature_multisource_token_supported/"
              "full_cutoff_7"
        ),
        "models": {
            "2d": ("Extra Trees", 200),
            "3d": ("SVM-RBF", 500),
            "morgan": ("Extra Trees", 500),
            "2d_3d": ("LightGBM", 300),
            "2d_3d_morgan": ("SVM-RBF", 200),
        },
    },
}

PREDICTION_KEYS = ["InChIKey_STD", "Mutation_normalized"]
DATASETS['Single-protein']['scaffold_dir'] = args.single_scaffold_dir.expanduser().resolve()
DATASETS['Single-protein']['model_dir'] = args.single_predictions_root.expanduser().resolve()
DATASETS['Cell-based']['scaffold_dir'] = args.cell_scaffold_dir.expanduser().resolve()
DATASETS['Cell-based']['model_dir'] = args.cell_predictions_root.expanduser().resolve()
input_files = []
for config in DATASETS.values():
    input_files.append(config['scaffold_dir'] / 'external_scaffold_overlap.csv')
    input_files.extend(config['model_dir'] / rep / 'predictions/strict_external_predictions.csv' for rep in REPS)
    reference = config['scaffold_dir'] / 'scaffold_performance.csv'
    if reference.exists():
        input_files.append(reference)
for path in input_files:
    if not path.is_file():
        parser.error('Missing input: ' + str(path))
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_mutation(value):
    if pd.isna(value):
        raise ValueError("Missing mutation label")

    value = str(value).strip().upper().replace(" ", "")

    if value in (
        "NO_REPORTED_MUTATION",
        "NOREPORTEDMUTATION",
    ):
        return "NO_REPORTED_MUTATION"

    return value


def normalize_scaffold_status(value):
    if pd.isna(value):
        return "Scaffold-less"

    value = str(value).strip().lower()

    if value == "seen":
        return "Seen"

    if value == "unseen":
        return "Unseen"

    if value in (
        "scaffold-less",
        "scaffoldless",
        "scaffold_less",
    ):
        return "Scaffold-less"

    raise ValueError(
        "Unrecognized scaffold status: {}".format(value)
    )


def prepare_identifiers(df, source):
    df = df.copy()

    for col in ["InChIKey_STD", "Mutation_clean"]:
        if col not in df.columns:
            raise ValueError(
                "{} missing column {}".format(source, col)
            )

    if df["InChIKey_STD"].isna().any():
        raise ValueError(
            "Missing InChIKey_STD in {}".format(source)
        )

    df["InChIKey_STD"] = (
        df["InChIKey_STD"].astype(str).str.strip()
    )

    df["Mutation_normalized"] = (
        df["Mutation_clean"].apply(normalize_mutation)
    )

    return df


# ============================================================
# 3. PREPARE COMPOUND-LEVEL SCAFFOLD ASSIGNMENTS
# ============================================================

def prepare_scaffolds(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = prepare_identifiers(
        pd.read_csv(path), str(path)
    )

    required = [
        "Murcko_scaffold",
        "Scaffold_status",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                "{} missing {}".format(path, col)
            )

    df["Murcko_scaffold"] = (
        df["Murcko_scaffold"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Scaffold_status"] = (
        df["Scaffold_status"]
        .apply(normalize_scaffold_status)
    )

    # Empty scaffold structures are scaffold-less.
    empty = df["Murcko_scaffold"].eq("")

    df.loc[empty, "Scaffold_status"] = "Scaffold-less"

    # A compound must have a single scaffold identity/status.
    conflicts = (
        df.groupby("InChIKey_STD")
        .agg(
            n_scaffolds=("Murcko_scaffold", "nunique"),
            n_statuses=("Scaffold_status", "nunique"),
        )
    )

    conflicts = conflicts[
        (conflicts["n_scaffolds"] > 1)
        | (conflicts["n_statuses"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            "Conflicting scaffold assignments:\n{}".format(
                conflicts.head(20).to_string()
            )
        )

    # Retain one scaffold assignment per chemical compound.
    df = df[
        [
            "InChIKey_STD",
            "Murcko_scaffold",
            "Scaffold_status",
        ]
    ].drop_duplicates(subset=["InChIKey_STD"])

    if df["InChIKey_STD"].duplicated().any():
        raise ValueError("Duplicate scaffold lookup keys")

    return df


# ============================================================
# 4. PREPARE OBSERVATION-LEVEL MODEL PREDICTIONS
# ============================================================

def prepare_predictions(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = prepare_identifiers(
        pd.read_csv(path), str(path)
    )

    required = [
        "observed",
        "predicted",
        "probability_active",
        "Inside_AD",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                "{} missing {}".format(path, col)
            )

    for col in [
        "observed",
        "predicted",
        "probability_active",
    ]:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        )

    if df[
        ["observed", "predicted", "probability_active"]
    ].isna().any().any():
        raise ValueError(
            "Missing prediction values in {}".format(path)
        )

    for col in ["observed", "predicted"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(
                "{} must contain binary 0/1 values".format(col)
            )

    if not df["probability_active"].between(0, 1).all():
        raise ValueError(
            "Probabilities outside [0, 1]"
        )

    def parse_ad(value):
        if pd.isna(value):
            raise ValueError("Missing Inside_AD value")

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        value = str(value).strip().lower()

        if value in ("true", "1", "yes"):
            return True

        if value in ("false", "0", "no"):
            return False

        raise ValueError(
            "Invalid Inside_AD: {}".format(value)
        )

    df["Inside_AD"] = df["Inside_AD"].apply(parse_ad)

    # Predictions must represent distinct compound-mutation
    # observations in these final external files.
    duplicates = df.duplicated(
        PREDICTION_KEYS, keep=False
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate compound-mutation prediction keys:\n{}".format(
                df.loc[
                    duplicates, PREDICTION_KEYS
                ].head(20).to_string(index=False)
            )
        )

    return df


# ============================================================
# 5. JOIN PREDICTIONS TO SCAFFOLD ASSIGNMENTS
# ============================================================

def merge_scaffolds(predictions, scaffolds, dataset, rep):

    # IMPORTANT:
    # Join scaffold information using compound identity only.
    # Retain mutation labels and every prediction observation.

    merged = predictions.merge(
        scaffolds,
        on="InChIKey_STD",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"].ne("both")

    if unmatched.any():
        print(
            merged.loc[
                unmatched,
                [
                    "InChIKey_STD",
                    "Mutation_normalized",
                    "_merge",
                ]
            ].head(20).to_string(index=False)
        )

        raise ValueError(
            "{} {}: {} predictions without scaffold assignment".format(
                dataset, rep, int(unmatched.sum())
            )
        )

    if len(merged) != len(predictions):
        raise ValueError(
            "Prediction observation count changed after merge"
        )

    merged = merged.drop(columns=["_merge"])

    if merged["Scaffold_status"].isna().any():
        raise ValueError(
            "Missing scaffold status after merge"
        )

    return merged


# ============================================================
# 6. METRIC CALCULATION
# ============================================================

def calculate_metrics(df):

    n = len(df)

    result = {
        "Observations": n,
        "Unique_compounds": (
            df["InChIKey_STD"].nunique()
        ),
        "Unique_scaffolds": (
            df.loc[
                df["Murcko_scaffold"].ne(""),
                "Murcko_scaffold",
            ].nunique()
        ),
        "Active": int(
            (df["observed"] == 1).sum()
        ),
        "Inactive": int(
            (df["observed"] == 0).sum()
        ),
        "Prevalence": (
            float((df["observed"] == 1).mean())
            if n else np.nan
        ),
        "Inside_AD": int(
            df["Inside_AD"].sum()
        ),
        "MCC": np.nan,
        "ROC_AUC": np.nan,
        "AP": np.nan,
        "Balanced_accuracy": np.nan,
        "Sensitivity": np.nan,
        "Specificity": np.nan,
        "TN": np.nan,
        "FP": np.nan,
        "FN": np.nan,
        "TP": np.nan,
    }

    if n == 0:
        return result

    y = df["observed"].to_numpy(dtype=int)
    yhat = df["predicted"].to_numpy(dtype=int)
    scores = df["probability_active"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y, yhat, labels=[0, 1]
    ).ravel()

    result.update({
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    # Two observed classes are required for ROC-AUC.
    # Single-class groups are marked undefined for
    # consistent comparison across all metrics.
    if len(np.unique(y)) < 2:
        return result

    result["MCC"] = matthews_corrcoef(y, yhat)
    result["ROC_AUC"] = roc_auc_score(y, scores)
    result["AP"] = average_precision_score(y, scores)

    result["Balanced_accuracy"] = (
        balanced_accuracy_score(y, yhat)
    )

    result["Sensitivity"] = (
        tp / (tp + fn)
        if tp + fn else np.nan
    )

    result["Specificity"] = (
        tn / (tn + fp)
        if tn + fp else np.nan
    )

    return result


# ============================================================
# 7. RUN ALL 10 FINAL MODELS
# ============================================================

performance_rows = []
ad_rows = []
summary_rows = []

for dataset, config in DATASETS.items():

    print("\n" + "=" * 75)
    print(dataset.upper())
    print("=" * 75)

    scaffold_path = (
        config["scaffold_dir"]
        / "external_scaffold_overlap.csv"
    )

    scaffolds = prepare_scaffolds(scaffold_path)

    print(
        "Compound-level scaffold assignments:",
        len(scaffolds)
    )

    for rep in REPS:

        algorithm, budget = config["models"][rep]

        pred_path = (
            config["model_dir"]
            / rep
            / "predictions/strict_external_predictions.csv"
        )

        predictions = prepare_predictions(pred_path)

        merged = merge_scaffolds(
            predictions,
            scaffolds,
            dataset,
            rep,
        )

        common = {
            "Dataset": dataset,
            "Representation": rep,
            "Algorithm": algorithm,
            "Budget": budget,
        }

        # ----------------------------------------------
        # Overall and scaffold-specific performance
        # ----------------------------------------------

        for status in [
            "Total",
            "Seen",
            "Unseen",
            "Scaffold-less",
        ]:

            if status == "Total":
                subset = merged
            else:
                subset = merged[
                    merged["Scaffold_status"] == status
                ]

            if status != "Total" and subset.empty:
                continue

            performance_rows.append({
                **common,
                "Scaffold_status": status,
                **calculate_metrics(subset),
            })

        # ----------------------------------------------
        # Scaffold status x applicability domain
        # ----------------------------------------------

        for status in [
            "Seen",
            "Unseen",
            "Scaffold-less",
        ]:

            for ad_value, ad_name in [
                (True, "Inside"),
                (False, "Outside"),
            ]:

                subset = merged[
                    (merged["Scaffold_status"] == status)
                    & (merged["Inside_AD"] == ad_value)
                ]

                if subset.empty:
                    continue

                ad_rows.append({
                    **common,
                    "Scaffold_status": status,
                    "AD_status": ad_name,
                    **calculate_metrics(subset),
                })

        # ----------------------------------------------
        # Scaffold composition summary
        # ----------------------------------------------

        seen = merged[
            merged["Scaffold_status"] == "Seen"
        ]

        unseen = merged[
            merged["Scaffold_status"] == "Unseen"
        ]

        scaffoldless = merged[
            merged["Scaffold_status"] == "Scaffold-less"
        ]

        summary_rows.append({
            **common,
            "External_observations": len(merged),
            "External_compounds": (
                merged["InChIKey_STD"].nunique()
            ),
            "Seen_observations": len(seen),
            "Unseen_observations": len(unseen),
            "Scaffold_less_observations": len(scaffoldless),
            "Seen_compounds": seen["InChIKey_STD"].nunique(),
            "Unseen_compounds": unseen["InChIKey_STD"].nunique(),
            "Unique_seen_scaffolds": (
                seen["Murcko_scaffold"].nunique()
            ),
            "Unique_unseen_scaffolds": (
                unseen["Murcko_scaffold"].nunique()
            ),
            "Inside_AD_observations": int(
                merged["Inside_AD"].sum()
            ),
        })

        print(
            "{} | {} | {} | budget {} | "
            "{} observations | seen {} | unseen {} | "
            "scaffold-less {}".format(
                dataset,
                rep,
                algorithm,
                budget,
                len(merged),
                len(seen),
                len(unseen),
                len(scaffoldless),
            )
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

performance = pd.DataFrame(performance_rows)
ad_performance = pd.DataFrame(ad_rows)
summary = pd.DataFrame(summary_rows)

performance.to_csv(
    OUTPUT / "all_10_scaffold_performance.csv",
    index=False,
)

ad_performance.to_csv(
    OUTPUT / "all_10_scaffold_ad_performance.csv",
    index=False,
)

summary.to_csv(
    OUTPUT / "all_10_scaffold_summary.csv",
    index=False,
)


# ============================================================
# 9. CHECK AGAINST ORIGINAL SCAFFOLD ANALYSIS
# ============================================================

checks = []

verified_models = {
    "Single-protein": "morgan",
    "Cell-based": "3d",
}

for dataset, rep in verified_models.items():

    original_path = (
        DATASETS[dataset]["scaffold_dir"]
        / "scaffold_performance.csv"
    )

    if not original_path.exists():
        print(
            "Original scaffold metrics not found:",
            original_path,
        )
        continue

    original = pd.read_csv(original_path)

    current = performance[
        (performance["Dataset"] == dataset)
        & (performance["Representation"] == rep)
        & (
            performance["Scaffold_status"].isin(
                ["Seen", "Unseen"]
            )
        )
    ]

    for _, row in current.iterrows():

        status = row["Scaffold_status"]

        old_rows = original[
            original["Scaffold_status"] == status
        ]

        if len(old_rows) != 1:
            continue

        old = old_rows.iloc[0]

        for new_col, old_col in [
            ("MCC", "MCC"),
            ("ROC_AUC", "ROC_AUC"),
            ("AP", "PR_AUC"),
            ("Balanced_accuracy", "Balanced_accuracy"),
        ]:

            if old_col not in original.columns:
                continue

            old_value = float(old[old_col])
            new_value = float(row[new_col])

            is_match = bool(
                np.isclose(
                    old_value,
                    new_value,
                    atol=1e-8,
                    rtol=0,
                    equal_nan=True,
                )
            )

            checks.append({
                "Dataset": dataset,
                "Representation": rep,
                "Scaffold_status": status,
                "Metric": new_col,
                "Original": old_value,
                "Recalculated": new_value,
                "Absolute_difference": (
                    abs(old_value - new_value)
                ),
                "Match": is_match,
            })

checks_df = pd.DataFrame(checks)

checks_df.to_csv(
    OUTPUT / "original_scaffold_reproducibility_check.csv",
    index=False,
)


# ============================================================
# 10. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SCAFFOLD GENERALIZATION: ALL 10 MODELS")
print("=" * 75)

display_columns = [
    "Dataset",
    "Representation",
    "Algorithm",
    "Budget",
    "Scaffold_status",
    "Observations",
    "MCC",
    "ROC_AUC",
    "AP",
]

selected = performance[
    performance["Scaffold_status"].isin(
        ["Seen", "Unseen"]
    )
]

print(
    selected[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("ORIGINAL SCAFFOLD REPRODUCIBILITY")
print("=" * 75)

if checks_df.empty:
    print("No original metrics available for comparison.")
else:
    print(checks_df.to_string(index=False))

    print(
        "\nMatching metrics: {}/{}".format(
            int(checks_df["Match"].sum()),
            len(checks_df),
        )
    )

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

for filename in [
    "all_10_scaffold_performance.csv",
    "all_10_scaffold_ad_performance.csv",
    "all_10_scaffold_summary.csv",
    "original_scaffold_reproducibility_check.csv",
]:
    print(OUTPUT / filename)

print("\nAnalysis completed.")
manifest = {
    'method': 'Compound-level scaffold lookup; observation-level performance; no training',
    'adapted_from': 'analyze_all_10_scaffold_models_1.py',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in input_files],
    'reference_checks': len(checks_df),
    'reference_checks_passed': int(checks_df['Match'].sum()) if not checks_df.empty else 0,
    'numpy_version': np.__version__, 'pandas_version': pd.__version__,
    'model_labels': {dataset: config['models'] for dataset, config in DATASETS.items()},
    'note': 'Algorithm and feature-budget labels are the verified original metadata; AP is average precision.',
}
(OUTPUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
