
#!/usr/bin/env python3
"""Molecular-only permutation SHAP for final EGFR QSAR models."""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_core(project):
    path = project / "scripts/10_train_grouped_cv_and_external_validation.py"
    spec = importlib.util.spec_from_file_location("egfr_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_inputs(core, path, bundle):
    # Original mutation encoding and selected-feature order.
    frame = core.read_data(path)
    features = list(bundle["selected_molecular_features"])
    mutations = list(bundle["mutation_features"])

    if mutations != list(core.MUTATIONS):
        raise ValueError("Mutation feature order differs from training.")

    # Use the original pipeline's numeric cleaning.
    numeric = frame[features].apply(
        pd.to_numeric, errors="coerce"
    )
    frame = frame.copy()
    frame.loc[:, features] = numeric

    X = core.matrix(
        frame, features, bundle["imputation_medians"]
    )

    expected = len(features) + len(mutations)
    if X.shape[1] != expected:
        raise ValueError("Incorrect feature count.")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite model inputs detected.")

    return frame, X, features, mutations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--development", required=True)
    p.add_argument("--external", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--explain", type=int, default=40)
    p.add_argument("--background", type=int, default=30)
    p.add_argument("--permutations", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    core = load_core(project)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    scaler = bundle.get("scaler")

    dev, Xdev, features, mutations = build_inputs(
        core, Path(args.development), bundle
    )
    ext, Xext, ext_features, ext_mutations = build_inputs(
        core, Path(args.external), bundle
    )

    assert features == ext_features
    assert mutations == ext_mutations

    n_features = len(features)
    n_total = n_features + len(mutations)

    if model.n_features_in_ != n_total:
        raise ValueError("Model input dimension mismatch.")

    if scaler is not None and scaler.n_features_in_ != n_total:
        raise ValueError("Scaler input dimension mismatch.")

    def predict_full(X):
        X = np.asarray(X, dtype=float)
        if scaler is not None:
            X = scaler.transform(X)
        return model.predict_proba(X)[:, 1]

    rng = np.random.default_rng(args.seed)

    explain_idx = np.sort(
        rng.choice(
            len(ext),
            min(args.explain, len(ext)),
            replace=False
        )
    )

    background_idx = np.sort(
        rng.choice(
            len(dev),
            min(args.background, len(dev)),
            replace=False
        )
    )

    pd.DataFrame({"external_row_index": explain_idx}).to_csv(output / "explained_row_indices.csv", index=False)
    pd.DataFrame({"development_row_index": background_idx}).to_csv(output / "background_row_indices.csv", index=False)
    Xexplain = Xext[explain_idx]
    background = Xdev[background_idx, :n_features]

    values = []
    baselines = []
    predictions = []

    masker = shap.maskers.Independent(background)

    for i, full_row in enumerate(Xexplain):
        mutation_values = full_row[n_features:].copy()

        # Only molecular descriptors/fingerprints change.
        # The mutation indicators remain fixed.
        def molecular_predict(molecular_matrix):
            molecular_matrix = np.atleast_2d(
                np.asarray(molecular_matrix, dtype=float)
            )
            fixed_mutations = np.repeat(
                mutation_values.reshape(1, -1),
                molecular_matrix.shape[0],
                axis=0
            )
            full = np.hstack([
                molecular_matrix, fixed_mutations
            ])
            return predict_full(full)

        explainer = shap.Explainer(
            molecular_predict,
            masker,
            algorithm="permutation",
            seed=args.seed
        )

        result = explainer(
            full_row[:n_features].reshape(1, -1),
            max_evals=args.permutations * (2 * n_features + 1)
        )

        shap_row = np.asarray(result.values).reshape(-1)
        baseline = float(
            np.asarray(result.base_values).reshape(-1)[0]
        )
        prediction = float(
            predict_full(full_row.reshape(1, -1))[0]
        )

        if not np.isclose(
            baseline + shap_row.sum(),
            prediction,
            atol=1e-4
        ):
            raise RuntimeError(
                f"SHAP additivity failed at external row {explain_idx[i]}"
            )

        values.append(shap_row)
        baselines.append(baseline)
        predictions.append(prediction)

        print(
            f"Explained {i + 1}/{len(Xexplain)}",
            flush=True
        )

    values = np.vstack(values)

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0)
    }).sort_values(
        "mean_abs_shap", ascending=False
    )

    importance.to_csv(
        output / "shap_molecular_importance.csv",
        index=False
    )

    pd.DataFrame(
        values, columns=features
    ).to_csv(
        output / "shap_molecular_values.csv",
        index=False
    )

    pd.DataFrame(
        Xexplain[:, :n_features],
        columns=features
    ).to_csv(
        output / "shap_molecular_inputs.csv",
        index=False
    )

    pd.DataFrame({
        "external_row_index": explain_idx,
        "baseline_probability": baselines,
        "predicted_probability": predictions,
        "shap_sum": values.sum(axis=1)
    }).to_csv(
        output / "shap_explained_predictions.csv",
        index=False
    )

    top = importance.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top["feature"], top["mean_abs_shap"]
    )
    ax.set_xlabel(
        "Mean |SHAP value| (predicted probability)"
    )
    ax.set_title("Top 20 molecular features")
    fig.tight_layout()
    fig.savefig(
        output / "shap_top20_bar.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        output / "shap_top20_bar.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    shap.summary_plot(
        values,
        pd.DataFrame(
            Xexplain[:, :n_features],
            columns=features
        ),
        max_display=20,
        show=False,
        plot_size=(10, 8)
    )
    plt.title("Molecular SHAP summary")
    plt.tight_layout()
    plt.savefig(
        output / "shap_beeswarm.png",
        dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        output / "shap_beeswarm.pdf",
        bbox_inches="tight"
    )
    plt.close("all")

    metadata = {
        "model": str(args.model),
        "development": str(args.development),
        "external": str(args.external),
        "method": "Permutation SHAP",
        "n_explained": len(explain_idx),
        "n_background": len(background_idx),
        "molecular_features": n_features,
        "mutation_features_fixed": mutations,
        "mutation_features_reported": False,
        "seed": args.seed,
        "permutations": args.permutations,
        "explanation": (
            "Molecular-only attributions on predicted active "
            "probability, with mutation indicators fixed at "
            "each explained compound's observed values."
        )
    }

    with open(
        output / "shap_metadata.json", "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nSHAP analysis completed.")
    print("Output:", output)
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
