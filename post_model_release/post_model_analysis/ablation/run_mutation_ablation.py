
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)

# ==========================================================
# 1. FILE PATHS
# ==========================================================

BASE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Optional fixed-configuration mutation ablation; retrains development-only models.')
parser.add_argument('--repository-root', type=Path, default=BASE.parent / 'EGFR_QSAR_PUBLICATION_RELEASE')
parser.add_argument('--data-root', type=Path, default=BASE / 'data')
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
REPO = args.repository_root.expanduser().resolve()
DATA = args.data_root.expanduser().resolve()
OUTPUT = args.output_dir.expanduser().resolve()
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    parser.error('Output directory must be new or empty.')
CONFIGS = {}
for assay, model_suffix, features, development_rows, external_rows in [
    ('single_protein', 'results/final_nested_qsar/full_cutoff_7/morgan/final_models/morgan__extra_trees__budget_300.joblib', 300, 5093, 2524),
    ('cell_based', 'results/final_nested_qsar/literature_multisource_token_supported/full_cutoff_7/3d/final_models/3d__svm_rbf__budget_500.joblib', 438, 3164, 192),
]:
    CONFIGS[assay] = {
        'model': REPO / assay / model_suffix,
        'development': DATA / 'modeling' / assay / 'development/5_all_features_2d_3d_morgan_with_metadata.csv',
        'external': DATA / 'modeling' / assay / 'external/5_all_features_2d_3d_morgan_with_metadata.csv',
        'expected_features': features,
        'expected_development_rows': development_rows,
        'expected_external_rows': external_rows,
    }

# ==========================================================
# 2. EVALUATION FUNCTION
# ==========================================================

def evaluate(model, X, y, threshold):

    classes = np.asarray(model.classes_)

    if 1 not in classes:
        raise ValueError("The trained model has no class 1.")

    active_idx = np.flatnonzero(classes == 1)[0]

    probability = model.predict_proba(X)[:, active_idx]
    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()

    results = {
        "MCC": matthews_corrcoef(y, prediction),
        "ROC_AUC": roc_auc_score(y, probability),
        "AP": average_precision_score(y, probability),
        "Balanced_Accuracy": balanced_accuracy_score(
            y, prediction
        ),
        "Accuracy": accuracy_score(y, prediction),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    return results, probability, prediction


# ==========================================================
# 3. FIXED-CONFIGURATION ABLATION
# ==========================================================

def run_ablation(name, cfg):

    print("\n" + "=" * 65)
    print("RUNNING:", name.upper())
    print("=" * 65)

    for key in ["model", "development", "external"]:
        if not cfg[key].is_file():
            raise FileNotFoundError(cfg[key])

    # Load saved original full-model information
    saved = joblib.load(cfg["model"])

    if not isinstance(saved, dict):
        raise TypeError("Expected a saved model dictionary.")

    features = list(saved["selected_molecular_features"])
    mutations = list(saved["mutation_features"])
    expected_mutations = ['C797S', 'L858R', 'L861Q', 'T790M', 'Other_EGFR_mutation', 'Mutation_unspecified']
    if mutations != expected_mutations:
        raise ValueError('Saved mutation features differ from the verified six-indicator encoding.')

    assert len(features) == cfg["expected_features"]
    assert len(features) == len(set(features))

    # Clone fitted classifier:
    # keeps exact algorithm and hyperparameters,
    # but removes its fitted state.
    model = clone(saved["model"])

    print("Algorithm:", saved["model_name"])
    print("Molecular features:", len(features))
    print("Mutation features excluded:", len(mutations))
    print("Hyperparameters:", model.get_params())

    # Read original modeling matrices
    train_df = pd.read_csv(
        cfg["development"], low_memory=False
    )

    external_df = pd.read_csv(
        cfg["external"], low_memory=False
    )

    if len(train_df) != cfg["expected_development_rows"]:
        raise ValueError("Unexpected development cohort size.")

    if len(external_df) != cfg["expected_external_rows"]:
        raise ValueError("Unexpected external cohort size.")

    target = saved["target"]
    key = saved["key"]
    threshold = float(saved.get("threshold", 0.5))

    for label, df in [
        ("Development", train_df),
        ("External", external_df),
    ]:
        missing = [
            col for col in features + [target, key]
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{label}: missing columns {missing[:10]}"
            )

    # Check for compound overlap between cohorts
    train_keys = set(train_df[key].dropna().astype(str))
    external_keys = set(external_df[key].dropna().astype(str))

    overlap = train_keys & external_keys

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} overlapping compound keys "
            "between development and external cohorts."
        )

    print("Compound overlap:", len(overlap))

    # Use ONLY the selected molecular features.
    # No OHE columns are passed to the classifier.
    X_train_df = train_df.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    )

    X_external_df = external_df.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    )

    X_train_df = X_train_df.replace(
        [np.inf, -np.inf], np.nan
    )
    X_external_df = X_external_df.replace(
        [np.inf, -np.inf], np.nan
    )

    y_train = train_df[target].astype(int).to_numpy()
    y_external = external_df[target].astype(int).to_numpy()

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Development must contain both classes.")

    if set(np.unique(y_external)) != {0, 1}:
        raise ValueError("External must contain both classes.")

    print("Development observations:", len(y_train))
    print("External observations:", len(y_external))

    # ------------------------------------------------------
    # Imputation: fit medians using development data only
    # ------------------------------------------------------

    medians = X_train_df.median(axis=0)

    if medians.isna().any():
        bad = medians.index[medians.isna()].tolist()
        raise ValueError(
            f"All-missing development features: {bad[:10]}"
        )

    X_train = X_train_df.fillna(medians).to_numpy(
        dtype=np.float64
    )
    X_external = X_external_df.fillna(medians).to_numpy(
        dtype=np.float64
    )

    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_external).all()
    ):
        raise ValueError("Non-finite values after imputation.")

    # ------------------------------------------------------
    # Scaling:
    # Extra Trees = no scaler
    # SVM-RBF = StandardScaler
    # ------------------------------------------------------

    original_scaler = saved.get("scaler")

    if original_scaler is not None:
        if not isinstance(original_scaler, StandardScaler):
            raise TypeError(
                "Unexpected scaler type. Inspect preprocessing."
            )

        scaler = clone(original_scaler)
        X_train = scaler.fit_transform(X_train)
        X_external = scaler.transform(X_external)
        print("Scaler: fitted on development data")
    else:
        scaler = None
        print("Scaler: not required")

    # ------------------------------------------------------
    # Train the molecular-only model
    # ------------------------------------------------------

    print("Training without mutation OHE...")

    model.fit(X_train, y_train)

    # ------------------------------------------------------
    # Evaluate on strict external cohort
    # ------------------------------------------------------

    metrics, probabilities, predictions = evaluate(
        model, X_external, y_external, threshold
    )

    print("\nEXTERNAL VALIDATION RESULTS")
    for metric, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # ------------------------------------------------------
    # Save model with all required metadata
    # ------------------------------------------------------

    ablated_bundle = {
        "model": model,
        "scaler": scaler,
        "selected_molecular_features": features,
        "mutation_features": [],
        "removed_mutation_features": mutations,
        "imputation_medians": medians.to_dict(),
        "threshold": threshold,
        "representation": saved["representation"],
        "model_name": saved["model_name"],
        "budget": saved["budget"],
        "hyperparameters": saved["hyperparameters"],
        "target": target,
        "key": key,
        "ablation_type": "fixed_configuration_without_OHE",
    }

    joblib.dump(
        ablated_bundle,
        OUTPUT / f"{name}_without_OHE.joblib"
    )

    # Save exact features
    pd.DataFrame({
        "feature": features
    }).to_csv(
        OUTPUT / f"{name}_selected_features.csv",
        index=False
    )

    # Save external predictions with compound identifier
    prediction_df = pd.DataFrame({
        "external_row_index": np.arange(len(external_df)),
        key: external_df[key].to_numpy(),
        "observed_activity": y_external,
        "probability_active": probabilities,
        "predicted_activity": predictions,
    })

    prediction_df.to_csv(
        OUTPUT / f"{name}_external_predictions_without_OHE.csv",
        index=False
    )

    # Save metrics
    summary = {
        "Dataset": name,
        "Algorithm": saved["model_name"],
        "Molecular_features": len(features),
        "Mutation_features_removed": len(mutations),
        "Development_N": len(y_train),
        "External_N": len(y_external),
        **metrics,
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT / f"{name}_external_metrics_without_OHE.csv",
        index=False
    )

    return summary


# ==========================================================
# 4. RUN BOTH DATASETS
# ==========================================================

def main():

    for config in CONFIGS.values():
        for key in ('model', 'development', 'external'):
            if not config[key].is_file():
                raise FileNotFoundError(config[key])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name, config in CONFIGS.items():
        summaries.append(run_ablation(name, config))

    comparison = pd.DataFrame(summaries)

    comparison.to_csv(
        OUTPUT / "ablation_external_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL ABLATION SUMMARY")
    print("=" * 65)

    columns = [
        "Dataset",
        "MCC",
        "ROC_AUC",
        "AP",
        "Balanced_Accuracy",
        "Sensitivity",
        "Specificity",
    ]

    print(comparison[columns].to_string(index=False))

    print("\nSaved to:", OUTPUT)
    manifest = {'adapted_from':'mutation_ohe_ablation.py',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method':'Development-only fixed-configuration retraining without all six mutation-annotation indicators',
        'nested_cv_repeated':False,'feature_selection_repeated':False,'hyperparameter_search_repeated':False,
        'model_training_performed':True,
        'inputs':{name:{k:str(v) for k,v in cfg.items()} for name,cfg in CONFIGS.items()}}
    (OUTPUT/'ablation_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__ == "__main__":
    main()
