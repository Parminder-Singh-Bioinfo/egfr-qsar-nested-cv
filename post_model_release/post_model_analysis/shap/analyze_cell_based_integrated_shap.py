
#!/usr/bin/env python3
"""
Integrated cell-based EGFR 3D SVM-RBF SHAP error analysis.

Reuses existing molecular SHAP values without recalculating SHAP.

Analyses:
  - Full external TP/FN and TN/FP
  - Inside/outside applicability domain
  - Seen/unseen/scaffold-less categories
  - Combined AD x scaffold groups
  - Signed and absolute SHAP differences
  - 3D descriptor percentile differences

Outputs:
  - Row-level classification CSV
  - Group-count and performance summaries
  - Feature comparison CSVs
  - PNG and PDF figures
  - Validation and analysis manifest

Dependencies: numpy, pandas, matplotlib
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

import argparse
BASE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Subgroup analysis from saved full-model SHAP values; no training or new SHAP calculations.')
parser.add_argument('--data-root', type=Path, default=BASE / 'data')
parser.add_argument('--repository-root', type=Path, default=BASE.parent / 'EGFR_QSAR_PUBLICATION_RELEASE')
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
DATA = args.data_root.expanduser().resolve()
REPO = args.repository_root.expanduser().resolve()
SHAP_DIR = DATA / 'shap/cell_based'
PRED_FILE = REPO / 'cell_based/results/final_nested_qsar/literature_multisource_token_supported/full_cutoff_7/3d/predictions/strict_external_predictions.csv'
SCAFFOLD_FILE = DATA / 'scaffold/cell_based/external_scaffold_predictions.csv'
OUT = args.output_dir.expanduser().resolve()
if OUT.exists() and any(OUT.iterdir()):
    parser.error('Output directory must be new or empty.')

EXPECTED_ROWS = 192
EXPECTED_FEATURES = 438
MIN_GROUP_SIZE = 5
TOP_N = 20
TOL = 1e-8

SCAFFOLD_CATEGORIES = [
    "Seen",
    "Unseen",
    "Scaffold-less",
]

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}"
        )


def normalize_mutation(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


def parse_boolean(series):
    result = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if result.isna().any():
        raise ValueError("Unrecognized AD values")

    return result.astype(bool)


def safe_filename(text):
    return re.sub(
        r"[^A-Za-z0-9_-]+", "_", str(text)
    )


def mean_or_nan(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def median_or_nan(values):
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def plot_difference(
    table,
    column,
    xlabel,
    title,
    output_stem,
    folder,
):
    ranked = table.reindex(
        table[column]
        .abs()
        .sort_values(ascending=False)
        .index
    )

    top = ranked.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))

    colors = [
        "#2166AC" if v >= 0 else "#B2182B"
        for v in top[column]
    ]

    ax.barh(
        top["feature"],
        top[column],
        color=colors,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            folder / f"{output_stem}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 3. LOAD FILES
# ============================================================

print("\nLoading cell-based data...")

pred = load_csv(PRED_FILE)
scaffold = load_csv(SCAFFOLD_FILE)

explained = load_csv(
    SHAP_DIR / "shap_explained_predictions.csv"
)

shap = load_csv(
    SHAP_DIR / "shap_molecular_values.csv"
)

inputs = load_csv(
    SHAP_DIR / "shap_molecular_inputs.csv"
)

require_columns(
    pred,
    [
        "InChIKey_STD",
        "Mutation_clean",
        "Inside_AD",
        "observed",
        "probability_active",
        "predicted",
    ],
    "Final predictions",
)

require_columns(
    scaffold,
    [
        "InChIKey_STD",
        "Mutation_clean",
        "Activity_binary",
        "Scaffold_status",
        "observed",
        "predicted",
        "probability_active",
        "Inside_AD",
    ],
    "Scaffold predictions",
)

require_columns(
    explained,
    [
        "external_row_index",
        "predicted_probability",
    ],
    "SHAP predictions",
)


# ============================================================
# 4. VERIFY SHAP AND PREDICTION ALIGNMENT
# ============================================================

n = len(pred)

if n != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} observations; found {n}"
    )

if not all(
    len(df) == n
    for df in [scaffold, explained, shap, inputs]
):
    raise ValueError(
        "Input files have different observation counts"
    )

if shap.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features; "
        f"found {shap.shape[1]}"
    )

if shap.columns.tolist() != inputs.columns.tolist():
    raise ValueError(
        "SHAP and molecular input columns do not match"
    )

if shap.columns.duplicated().any():
    raise ValueError("Duplicate molecular feature columns")

shap_values = shap.to_numpy(dtype=float)
input_values = inputs.to_numpy(dtype=float)

if not np.isfinite(shap_values).all():
    raise ValueError("Non-finite SHAP values")

if not np.isfinite(input_values).all():
    raise ValueError("Non-finite molecular input values")

indices = explained[
    "external_row_index"
].to_numpy()

if not np.array_equal(indices, np.arange(n)):
    raise ValueError(
        "SHAP external_row_index is not sequential"
    )

prob_error = np.abs(
    explained["predicted_probability"].to_numpy(
        dtype=float
    )
    - pred["probability_active"].to_numpy(
        dtype=float
    )
)

max_prob_error = float(prob_error.max())

if max_prob_error > TOL:
    raise ValueError(
        "SHAP predicted probabilities do not match "
        f"final predictions: {max_prob_error}"
    )

print("SHAP probability alignment verified.")


# ============================================================
# 5. VERIFY SCAFFOLD ROW ALIGNMENT
# ============================================================

pred_keys = (
    pred["InChIKey_STD"].astype(str)
    + "|"
    + normalize_mutation(pred["Mutation_clean"])
)

scaffold_keys = (
    scaffold["InChIKey_STD"].astype(str)
    + "|"
    + normalize_mutation(scaffold["Mutation_clean"])
)

if not np.array_equal(
    pred_keys.to_numpy(),
    scaffold_keys.to_numpy(),
):
    raise ValueError(
        "Scaffold observations are not in prediction order. "
        "Do not join by InChIKey alone."
    )

for column in ["observed", "predicted"]:
    if not np.array_equal(
        pred[column].to_numpy(),
        scaffold[column].to_numpy(),
    ):
        raise ValueError(
            f"Scaffold mismatch in {column}"
        )

if not np.array_equal(
    pred["observed"].to_numpy(),
    scaffold["Activity_binary"].to_numpy(),
):
    raise ValueError(
        "Activity_binary does not match observed labels"
    )

scaffold_prob_error = float(
    np.abs(
        pred["probability_active"].to_numpy(dtype=float)
        - scaffold["probability_active"].to_numpy(
            dtype=float
        )
    ).max()
)

if scaffold_prob_error > TOL:
    raise ValueError(
        "Scaffold prediction probabilities differ"
    )

ad_pred = parse_boolean(pred["Inside_AD"])
ad_scaffold = parse_boolean(scaffold["Inside_AD"])

if not np.array_equal(
    ad_pred.to_numpy(),
    ad_scaffold.to_numpy(),
):
    raise ValueError("AD annotations disagree")

print("Scaffold prediction alignment verified.")


# ============================================================
# 6. CLASSIFY TP / FN / TN / FP
# ============================================================

data = pred.copy()

data["external_row_index"] = np.arange(n)
data["Inside_AD"] = ad_pred

data["Scaffold_status"] = (
    scaffold["Scaffold_status"]
    .astype(str)
    .str.strip()
)

unexpected = sorted(
    set(data["Scaffold_status"])
    - set(SCAFFOLD_CATEGORIES)
)

if unexpected:
    raise ValueError(
        f"Unexpected scaffold categories: {unexpected}"
    )

if not data["observed"].isin([0, 1]).all():
    raise ValueError("Invalid observed class labels")

if not data["predicted"].isin([0, 1]).all():
    raise ValueError("Invalid predicted class labels")

y = data["observed"].to_numpy()
p = data["predicted"].to_numpy()

data["group"] = np.select(
    [
        (y == 1) & (p == 1),
        (y == 1) & (p == 0),
        (y == 0) & (p == 0),
        (y == 0) & (p == 1),
    ],
    ["TP", "FN", "TN", "FP"],
    default="UNKNOWN",
)

if (data["group"] == "UNKNOWN").any():
    raise ValueError("Unclassified observations")

data["AD_group"] = np.where(
    data["Inside_AD"],
    "Inside_AD",
    "Outside_AD",
)

data["AD_scaffold_group"] = (
    data["AD_group"]
    + "__Scaffold_"
    + data["Scaffold_status"]
)

data.to_csv(
    OUT / "external_predictions_with_error_groups.csv",
    index=False,
)

print("\nScaffold categories:")
print(
    data["Scaffold_status"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 7. DEFINE ANALYSIS SUBSETS
# ============================================================

subsets = {
    "Full_External": np.ones(n, dtype=bool)
}

for ad in ["Inside_AD", "Outside_AD"]:
    subsets[ad] = (
        data["AD_group"] == ad
    ).to_numpy()

for sc in SCAFFOLD_CATEGORIES:
    subsets[f"Scaffold_{sc}"] = (
        data["Scaffold_status"] == sc
    ).to_numpy()

for ad in ["Inside_AD", "Outside_AD"]:
    for sc in SCAFFOLD_CATEGORIES:
        name = f"{ad}__Scaffold_{sc}"

        subsets[name] = (
            (data["AD_group"] == ad)
            & (data["Scaffold_status"] == sc)
        ).to_numpy()


# ============================================================
# 8. DESCRIPTOR PERCENTILE RANKS
# ============================================================

# All 192 observations form the reference distribution.
# This avoids comparing raw descriptor scales, which
# differ substantially among 3D descriptor families.

descriptor_percentiles = (
    inputs.rank(
        axis=0,
        pct=True,
        method="average",
    ) * 100
)

percentile_values = (
    descriptor_percentiles.to_numpy(dtype=float)
)

features = shap.columns.to_numpy()


# ============================================================
# 9. GROUP COUNTS AND PERFORMANCE
# ============================================================

summary_rows = []
subset_performance = []

groups = ["TP", "FN", "TN", "FP"]

for subset_name, subset_mask in subsets.items():

    d = data.loc[subset_mask]

    counts = (
        d["group"]
        .value_counts()
        .reindex(groups, fill_value=0)
    )

    tp = int(counts["TP"])
    fn = int(counts["FN"])
    tn = int(counts["TN"])
    fp = int(counts["FP"])

    total = len(d)

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0 else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0 else np.nan
    )

    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if np.isfinite(sensitivity)
        and np.isfinite(specificity)
        else np.nan
    )

    denominator = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn) / denominator
        if denominator > 0 else np.nan
    )

    subset_performance.append({
        "subset": subset_name,
        "n": total,
        "unique_compounds": d["InChIKey_STD"].nunique(),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "accuracy": (
            (tp + tn) / total
            if total else np.nan
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    })

    for group in groups:
        g = d.loc[d["group"] == group]

        probabilities = g[
            "probability_active"
        ].to_numpy(dtype=float)

        summary_rows.append({
            "subset": subset_name,
            "group": group,
            "n": len(g),
            "unique_compounds":
                g["InChIKey_STD"].nunique(),
            "mean_probability_active":
                mean_or_nan(probabilities),
            "median_probability_active":
                median_or_nan(probabilities),
        })

pd.DataFrame(summary_rows).to_csv(
    OUT / "error_group_summary_by_subset.csv",
    index=False,
)

pd.DataFrame(subset_performance).to_csv(
    OUT / "subset_classification_performance.csv",
    index=False,
)


# ============================================================
# 10. TP/FN AND TN/FP FEATURE COMPARISONS
# ============================================================

comparisons = {
    "TP_vs_FN": ("TP", "FN"),
    "TN_vs_FP": ("TN", "FP"),
}

all_results = []
analysis_log = []

group_values = data["group"].to_numpy()

for subset_name, subset_mask in subsets.items():

    for comparison_name, (a, b) in comparisons.items():

        mask_a = (
            subset_mask & (group_values == a)
        )

        mask_b = (
            subset_mask & (group_values == b)
        )

        na = int(mask_a.sum())
        nb = int(mask_b.sum())

        record = {
            "subset": subset_name,
            "comparison": comparison_name,
            "first_group": a,
            "second_group": b,
            "n_first": na,
            "n_second": nb,
            "unique_compounds_first": int(
                data.loc[
                    mask_a, "InChIKey_STD"
                ].nunique()
            ),
            "unique_compounds_second": int(
                data.loc[
                    mask_b, "InChIKey_STD"
                ].nunique()
            ),
        }

        if min(na, nb) < MIN_GROUP_SIZE:
            record["status"] = "SKIPPED_SMALL_GROUP"
            analysis_log.append(record)
            continue

        sa = shap_values[mask_a]
        sb = shap_values[mask_b]

        pa = percentile_values[mask_a]
        pb = percentile_values[mask_b]

        xa = input_values[mask_a]
        xb = input_values[mask_b]

        result = pd.DataFrame({
            "feature": features,

            f"{a}_mean_signed_SHAP":
                sa.mean(axis=0),

            f"{b}_mean_signed_SHAP":
                sb.mean(axis=0),

            f"{a}_mean_abs_SHAP":
                np.abs(sa).mean(axis=0),

            f"{b}_mean_abs_SHAP":
                np.abs(sb).mean(axis=0),

            f"{a}_mean_descriptor_percentile":
                pa.mean(axis=0),

            f"{b}_mean_descriptor_percentile":
                pb.mean(axis=0),

            f"{a}_mean_descriptor_value":
                xa.mean(axis=0),

            f"{b}_mean_descriptor_value":
                xb.mean(axis=0),
        })

        result["signed_SHAP_difference"] = (
            result[f"{a}_mean_signed_SHAP"]
            - result[f"{b}_mean_signed_SHAP"]
        )

        result["absolute_SHAP_difference"] = (
            result[f"{a}_mean_abs_SHAP"]
            - result[f"{b}_mean_abs_SHAP"]
        )

        result["descriptor_percentile_difference"] = (
            result[f"{a}_mean_descriptor_percentile"]
            - result[f"{b}_mean_descriptor_percentile"]
        )

        result["abs_signed_difference"] = (
            result["signed_SHAP_difference"].abs()
        )

        result = result.sort_values(
            "abs_signed_difference",
            ascending=False,
        ).reset_index(drop=True)

        result.insert(0, "subset", subset_name)
        result.insert(1, "comparison", comparison_name)
        result.insert(2, "n_first", na)
        result.insert(3, "n_second", nb)

        all_results.append(result)

        folder = OUT / safe_filename(subset_name)
        folder.mkdir(parents=True, exist_ok=True)

        result.to_csv(
            folder
            / f"{comparison_name}_feature_comparison.csv",
            index=False,
        )

        title = (
            f"{subset_name}: {a} vs {b}\n"
            f"n={na} vs n={nb}"
        )

        plot_difference(
            result,
            "signed_SHAP_difference",
            f"Mean signed SHAP ({a}) - ({b})\n"
            "Contribution to predicted Active probability",
            title,
            f"{comparison_name}_signed_SHAP",
            folder,
        )

        plot_difference(
            result,
            "absolute_SHAP_difference",
            f"Mean absolute SHAP ({a}) - ({b})",
            title,
            f"{comparison_name}_absolute_SHAP",
            folder,
        )

        plot_difference(
            result,
            "descriptor_percentile_difference",
            f"Mean descriptor percentile ({a}) - ({b})",
            title,
            f"{comparison_name}_descriptor_percentile",
            folder,
        )

        record["status"] = "COMPLETED"
        analysis_log.append(record)


# ============================================================
# 11. SAVE COMBINED RESULTS
# ============================================================

if all_results:
    pd.concat(
        all_results,
        ignore_index=True,
    ).to_csv(
        OUT / "all_feature_comparisons.csv",
        index=False,
    )

status = pd.DataFrame(analysis_log)

status.to_csv(
    OUT / "comparison_status.csv",
    index=False,
)


# ============================================================
# 12. SAVE VALIDATION MANIFEST
# ============================================================

manifest = {
    "analysis":
        "Cell-based 3D SVM-RBF SHAP error analysis",

    "external_observations": n,

    "unique_compounds": int(
        data["InChIKey_STD"].nunique()
    ),

    "molecular_features": len(features),

    "max_SHAP_probability_difference":
        max_prob_error,

    "max_scaffold_probability_difference":
        scaffold_prob_error,

    "SHAP_recomputed": False,
    "model_retrained": False,

    "min_group_size": MIN_GROUP_SIZE,

    "scaffold_counts": {
        str(k): int(v)
        for k, v in data[
            "Scaffold_status"
        ].value_counts().items()
    },

    "TP": int((data["group"] == "TP").sum()),
    "FN": int((data["group"] == "FN").sum()),
    "TN": int((data["group"] == "TN").sum()),
    "FP": int((data["group"] == "FP").sum()),

    "comparisons_completed": int(
        (status["status"] == "COMPLETED").sum()
    ),

    "comparisons_skipped": int(
        (status["status"] != "COMPLETED").sum()
    ),

    "sources": {
        "predictions": str(PRED_FILE),
        "scaffolds": str(SCAFFOLD_FILE),
        "SHAP": str(SHAP_DIR),
    },
}

with open(
    OUT / "analysis_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)


# ============================================================
# 13. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CELL-BASED SHAP ERROR ANALYSIS COMPLETED")
print("=" * 60)

print("External observations:", n)

print(
    "Unique compounds:",
    data["InChIKey_STD"].nunique(),
)

print("Molecular features:", len(features))

print(
    "Maximum SHAP probability error:",
    max_prob_error,
)

print("\nClassification groups:")
print(
    data["group"]
    .value_counts()
    .reindex(groups, fill_value=0)
    .to_string()
)

print("\nScaffold groups:")
print(
    data["Scaffold_status"]
    .value_counts()
    .to_string()
)

print("\nCompleted comparisons:",
      manifest["comparisons_completed"])

print("Skipped comparisons:",
      manifest["comparisons_skipped"])

print("\nSkipped subgroup comparisons:")

skipped = status[
    status["status"] != "COMPLETED"
]

if skipped.empty:
    print("None")
else:
    print(
        skipped[
            [
                "subset",
                "comparison",
                "n_first",
                "n_second",
            ]
        ].to_string(index=False)
    )

print("\nOutput directory:")
print(OUT)

print("\nNo SHAP recalculation or model retraining performed.")
