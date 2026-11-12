
#!/usr/bin/env python3
"""
Integrated single-protein Morgan SHAP error analysis.

Reuses existing SHAP values. Does not retrain the model
or recalculate SHAP.

Outputs:
  - Row-level classifications with AD/scaffold labels
  - TP/FN and TN/FP comparison tables
  - Group-count and probability summaries
  - Top-20 signed SHAP difference plots (PNG/PDF)
  - Top-20 Morgan-bit frequency difference plots
  - Analysis manifest and validation report

Requires: pandas, numpy, matplotlib
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
BASE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Subgroup analysis from saved full-model SHAP values; no training or new SHAP calculations.')
parser.add_argument('--data-root', type=Path, default=BASE / 'data')
parser.add_argument('--repository-root', type=Path, default=BASE.parent / 'EGFR_QSAR_PUBLICATION_RELEASE')
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
DATA = args.data_root.expanduser().resolve()
REPO = args.repository_root.expanduser().resolve()
SHAP_DIR = DATA / 'shap/single_protein'
PRED_FILE = REPO / 'single_protein/results/final_nested_qsar/full_cutoff_7/morgan/predictions/strict_external_predictions.csv'
SCAFFOLD_FILE = DATA / 'scaffold/single_protein/external_scaffold_predictions.csv'
OUT = args.output_dir.expanduser().resolve()
if OUT.exists() and any(OUT.iterdir()):
    parser.error('Output directory must be new or empty.')

TOP_N = 20
MIN_GROUP_SIZE = 5
PROBABILITY_TOLERANCE = 1e-8
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)

def require(df, cols, name):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing {missing}")

def normalize_mutation(series):
    return (
        series.astype(str).str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapped = s.astype(str).str.strip().str.lower().map({
        "true": True, "false": False,
        "1": True, "0": False
    })
    if mapped.isna().any():
        raise ValueError("Unrecognized Inside_AD values")
    return mapped.astype(bool)

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)

print("Loading existing results...")

pred = load(PRED_FILE)
scaffold = load(SCAFFOLD_FILE)
explained = load(
    SHAP_DIR / "shap_explained_predictions.csv"
)
shap = load(SHAP_DIR / "shap_molecular_values.csv")
inputs = load(SHAP_DIR / "shap_molecular_inputs.csv")

require(
    pred,
    ["InChIKey_STD", "Mutation_clean", "Inside_AD",
     "observed", "probability_active", "predicted"],
    "Final predictions"
)
require(
    scaffold,
    ["InChIKey_STD", "Mutation_clean",
     "Scaffold_status", "Inside_AD",
     "predicted", "probability_active",
     "Activity_binary"],
    "Scaffold predictions"
)
require(
    explained,
    ["external_row_index", "predicted_probability"],
    "SHAP predictions"
)

n = len(pred)
if not (n == len(explained) == len(shap) == len(inputs)):
    raise ValueError("SHAP and prediction row counts differ")

if n != 2524:
    raise ValueError(f"Expected 2524 observations, got {n}")

if shap.columns.tolist() != inputs.columns.tolist():
    raise ValueError("SHAP/input feature columns differ")

if shap.columns.duplicated().any():
    raise ValueError("Duplicate molecular feature columns")

if shap.isna().any().any():
    raise ValueError("Missing SHAP values detected")

if not np.isfinite(shap.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite SHAP values detected")

# --------------------------------------------------
# 1. VERIFY SHAP-TO-PREDICTION ALIGNMENT
# --------------------------------------------------
indices = explained.external_row_index.to_numpy()

if not np.array_equal(indices, np.arange(n)):
    raise ValueError(
        "SHAP row indices are not 0..2523 in order. "
        "Stop and inspect row alignment before continuing."
    )

prob_error = np.abs(
    explained.predicted_probability.to_numpy()
    - pred.probability_active.to_numpy()
)
max_prob_error = float(prob_error.max())

if max_prob_error > PROBABILITY_TOLERANCE:
    raise ValueError(
        f"SHAP probabilities do not match: {max_prob_error}"
    )

# Use the saved final-model predictions, not a newly
# computed probability threshold.
if not pred.observed.isin([0, 1]).all():
    raise ValueError("Invalid observed labels")
if not pred.predicted.isin([0, 1]).all():
    raise ValueError("Invalid predicted labels")

# --------------------------------------------------
# 2. VERIFY AND JOIN SCAFFOLD ANNOTATIONS
# --------------------------------------------------
pred = pred.copy()
scaffold = scaffold.copy()

pred["_mutation_key"] = normalize_mutation(
    pred.Mutation_clean
)
scaffold["_mutation_key"] = normalize_mutation(
    scaffold.Mutation_clean
)

join_cols = ["InChIKey_STD", "_mutation_key"]

if pred.duplicated(join_cols).any():
    raise ValueError("Duplicate prediction compound/mutation keys")
if scaffold.duplicated(join_cols).any():
    raise ValueError("Duplicate scaffold compound/mutation keys")

annotations = scaffold[
    join_cols + [
        "Scaffold_status",
        "Inside_AD",
        "predicted",
        "probability_active",
        "Activity_binary"
    ]
].rename(columns={
    "Inside_AD": "scaffold_Inside_AD",
    "predicted": "scaffold_predicted",
    "probability_active": "scaffold_probability",
    "Activity_binary": "scaffold_observed"
})

merged = pred.merge(
    annotations,
    on=join_cols,
    how="left",
    validate="one_to_one",
    indicator=True,
    sort=False
)

if not (merged["_merge"] == "both").all():
    raise ValueError(
        "Some predictions could not be matched to scaffolds"
    )

# Merge preserves left order with sort=False.
if not merged.InChIKey_STD.equals(pred.InChIKey_STD):
    raise ValueError("Prediction row order changed")

merged["Inside_AD"] = bool_series(merged.Inside_AD)
merged["scaffold_Inside_AD"] = bool_series(
    merged.scaffold_Inside_AD
)

if not (
    merged.Inside_AD == merged.scaffold_Inside_AD
).all():
    raise ValueError("AD labels disagree")

if not (
    merged.predicted == merged.scaffold_predicted
).all():
    raise ValueError("Predicted classes disagree")

if not (
    merged.observed == merged.scaffold_observed
).all():
    raise ValueError("Observed labels disagree")

scaffold_prob_error = np.abs(
    merged.probability_active
    - merged.scaffold_probability
).max()

if scaffold_prob_error > PROBABILITY_TOLERANCE:
    raise ValueError("Scaffold prediction probabilities disagree")

status = merged.Scaffold_status.astype(str).str.strip()
if not status.isin(["Seen", "Unseen"]).all():
    raise ValueError(
        "Unexpected scaffold statuses: "
        + str(status.unique().tolist())
    )
merged["Scaffold_status"] = status

# --------------------------------------------------
# 3. CLASSIFY TP, FN, TN, FP
# --------------------------------------------------
y = merged.observed.to_numpy()
p = merged.predicted.to_numpy()

merged["group"] = np.select(
    [
        (y == 1) & (p == 1),
        (y == 1) & (p == 0),
        (y == 0) & (p == 0),
        (y == 0) & (p == 1)
    ],
    ["TP", "FN", "TN", "FP"],
    default="UNKNOWN"
)

if (merged.group == "UNKNOWN").any():
    raise ValueError("Unclassified observations")

merged["AD_group"] = np.where(
    merged.Inside_AD, "Inside_AD", "Outside_AD"
)

merged["AD_scaffold_group"] = (
    merged.AD_group + "__" + merged.Scaffold_status
)

merged["external_row_index"] = np.arange(n)

merged.to_csv(
    OUT / "external_predictions_with_error_groups.csv",
    index=False
)

# --------------------------------------------------
# 4. GROUP SUMMARIES
# --------------------------------------------------
subsets = {"Full_External": np.ones(n, dtype=bool)}

for ad in ["Inside_AD", "Outside_AD"]:
    subsets[ad] = (
        merged.AD_group == ad
    ).to_numpy()

for sc in ["Seen", "Unseen"]:
    subsets[f"Scaffold_{sc}"] = (
        merged.Scaffold_status == sc
    ).to_numpy()

for ad in ["Inside_AD", "Outside_AD"]:
    for sc in ["Seen", "Unseen"]:
        name = f"{ad}__Scaffold_{sc}"
        subsets[name] = (
            (merged.AD_group == ad)
            & (merged.Scaffold_status == sc)
        ).to_numpy()

summary_rows = []
for subset_name, mask in subsets.items():
    d = merged.loc[mask]
    for group in ["TP", "FN", "TN", "FP"]:
        g = d.loc[d.group == group]
        summary_rows.append({
            "subset": subset_name,
            "group": group,
            "n": len(g),
            "mean_probability_active":
                g.probability_active.mean(),
            "median_probability_active":
                g.probability_active.median()
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "error_group_summary_by_subset.csv",
    index=False
)

# --------------------------------------------------
# 5. FEATURE COMPARISON
# --------------------------------------------------
shap_array = shap.to_numpy(dtype=float)
input_array = inputs.to_numpy(dtype=float)
features = shap.columns.to_numpy()

comparisons = {
    "TP_vs_FN": ("TP", "FN"),
    "TN_vs_FP": ("TN", "FP")
}

all_results = []
analysis_log = []

def compare(subset_name, subset_mask, title, a, b):
    mask_a = subset_mask & (merged.group.to_numpy() == a)
    mask_b = subset_mask & (merged.group.to_numpy() == b)

    na, nb = int(mask_a.sum()), int(mask_b.sum())

    record = {
        "subset": subset_name,
        "comparison": title,
        "n_first": na,
        "n_second": nb,
        "first_group": a,
        "second_group": b
    }

    if min(na, nb) < MIN_GROUP_SIZE:
        record["status"] = "SKIPPED_SMALL_GROUP"
        analysis_log.append(record)
        return

    sa = shap_array[mask_a]
    sb = shap_array[mask_b]
    xa = input_array[mask_a]
    xb = input_array[mask_b]

    result = pd.DataFrame({
        "feature": features,
        f"{a}_mean_signed_SHAP": sa.mean(axis=0),
        f"{b}_mean_signed_SHAP": sb.mean(axis=0),
        f"{a}_mean_abs_SHAP": np.abs(sa).mean(axis=0),
        f"{b}_mean_abs_SHAP": np.abs(sb).mean(axis=0),
        f"{a}_bit_frequency": xa.mean(axis=0),
        f"{b}_bit_frequency": xb.mean(axis=0)
    })

    result["signed_SHAP_difference"] = (
        result[f"{a}_mean_signed_SHAP"]
        - result[f"{b}_mean_signed_SHAP"]
    )
    result["absolute_SHAP_difference"] = (
        result[f"{a}_mean_abs_SHAP"]
        - result[f"{b}_mean_abs_SHAP"]
    )
    result["bit_frequency_difference"] = (
        result[f"{a}_bit_frequency"]
        - result[f"{b}_bit_frequency"]
    )

    result["abs_signed_difference"] = (
        result.signed_SHAP_difference.abs()
    )

    result = result.sort_values(
        "abs_signed_difference", ascending=False
    )

    result.insert(0, "subset", subset_name)
    result.insert(1, "comparison", title)
    result.insert(2, "n_first", na)
    result.insert(3, "n_second", nb)

    all_results.append(result)

    folder = OUT / safe_name(subset_name)
    folder.mkdir(exist_ok=True)

    result.to_csv(
        folder / f"{title}_feature_comparison.csv",
        index=False
    )

    # Signed SHAP difference
    top = result.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(
        top.feature,
        top.signed_SHAP_difference
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        f"Mean signed SHAP ({a}) - ({b})\n"
        "Contribution to predicted Active probability"
    )
    ax.set_title(
        f"{subset_name}: {a} vs {b}\n"
        f"n={na} vs n={nb}"
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            folder / f"{title}_signed_SHAP.{ext}",
            dpi=300, bbox_inches="tight"
        )
    plt.close(fig)

    # Fingerprint bit frequency difference
    freq = result.reindex(
        result.bit_frequency_difference.abs()
        .sort_values(ascending=False).index
    ).head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(
        freq.feature,
        freq.bit_frequency_difference
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        f"Mean bit value ({a}) - ({b})"
    )
    ax.set_title(
        f"{subset_name}: Morgan bit prevalence\n"
        f"{a} (n={na}) vs {b} (n={nb})"
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            folder / f"{title}_bit_frequency.{ext}",
            dpi=300, bbox_inches="tight"
        )
    plt.close(fig)

    record["status"] = "COMPLETED"
    analysis_log.append(record)

for subset_name, subset_mask in subsets.items():
    for title, (a, b) in comparisons.items():
        compare(subset_name, subset_mask, title, a, b)

if all_results:
    pd.concat(
        all_results, ignore_index=True
    ).to_csv(
        OUT / "all_feature_comparisons.csv",
        index=False
    )

log = pd.DataFrame(analysis_log)
log.to_csv(
    OUT / "comparison_status.csv",
    index=False
)

# --------------------------------------------------
# 6. VALIDATION AND MANIFEST
# --------------------------------------------------
manifest = {
    "analysis": "Single-protein Morgan SHAP error analysis",
    "n_external": n,
    "n_molecular_features": len(features),
    "SHAP_probability_max_difference": max_prob_error,
    "scaffold_probability_max_difference":
        float(scaffold_prob_error),
    "model_predictions_reused": True,
    "SHAP_recomputed": False,
    "min_group_size": MIN_GROUP_SIZE,
    "TP": int((merged.group == "TP").sum()),
    "FN": int((merged.group == "FN").sum()),
    "TN": int((merged.group == "TN").sum()),
    "FP": int((merged.group == "FP").sum()),
    "comparisons_completed": int(
        (log.status == "COMPLETED").sum()
    ),
    "comparisons_skipped": int(
        (log.status == "SKIPPED_SMALL_GROUP").sum()
    ),
    "sources": {
        "predictions": str(PRED_FILE),
        "scaffolds": str(SCAFFOLD_FILE),
        "SHAP": str(SHAP_DIR)
    }
}

(OUT / "analysis_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print("\nAnalysis completed successfully")
print("External observations:", n)
print("Molecular features:", len(features))
print("SHAP probability max error:", max_prob_error)
print("\nClassification groups:")
print(merged.group.value_counts().to_string())
print("\nCompleted comparisons:",
      manifest["comparisons_completed"])
print("Skipped comparisons:",
      manifest["comparisons_skipped"])
print("\nOutput directory:", OUT)
