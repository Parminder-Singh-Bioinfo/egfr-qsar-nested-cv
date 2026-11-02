#!/usr/bin/env python3
"""Nested grouped-CV QSAR optimization with frozen external/AD evaluation.

The outer StratifiedGroupKFold estimates internal generalization. Within each
outer-training partition, feature selection is independently refitted in each
inner-training fold before randomized hyperparameter comparison. InChIKey_STD
is the grouping unit throughout. The strict external set is used only after a
development-only winner and final configuration have been selected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--external", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--representation", required=True, choices=["2d", "3d", "morgan", "2d_3d", "2d_3d_morgan"])
    p.add_argument("--analysis-name", required=True)
    p.add_argument("--core-script", type=Path, default=Path(__file__).with_name("qsar_core.py"))
    p.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200, 300, 500])
    p.add_argument("--models", nargs="+", default=["logistic", "svm_rbf", "random_forest", "extra_trees", "lightgbm", "xgboost", "knn"])
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--search-iterations", type=int, default=12)
    p.add_argument("--missing-threshold", type=float, default=0.30)
    p.add_argument("--correlation-threshold", type=float, default=0.90)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--ad-column", default="Inside_AD")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_core(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("qsar_core", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_bool(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (bool, np.bool_)): return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0", "inside", "inside_ad", "in_domain"}: return True
    if text in {"false", "f", "no", "n", "0", "0.0", "outside", "outside_ad", "out_of_domain"}: return False
    return np.nan


def available_models(requested):
    available = []
    for name in requested:
        if name == "lightgbm":
            try: import lightgbm  # noqa: F401
            except ImportError:
                print("Warning: LightGBM unavailable; skipped", file=sys.stderr); continue
        if name == "xgboost":
            try: import xgboost  # noqa: F401
            except ImportError:
                print("Warning: XGBoost unavailable; skipped", file=sys.stderr); continue
        available.append(name)
    if not available: raise ValueError("No requested models are available")
    return available


def parameter_space(name):
    spaces = {
        "logistic": {"C": np.logspace(-3, 2, 30), "class_weight": ["balanced"]},
        "svm_rbf": {"C": np.logspace(-2, 2, 25), "gamma": np.logspace(-5, -1, 25), "class_weight": ["balanced"]},
        "random_forest": {"n_estimators": [300, 500, 800], "max_depth": [8, 12, 16, None], "min_samples_split": [4, 8, 12], "min_samples_leaf": [2, 3, 5], "max_features": ["sqrt", 0.3, 0.5], "class_weight": ["balanced_subsample"]},
        "extra_trees": {"n_estimators": [300, 500, 800], "max_depth": [8, 12, 16, None], "min_samples_split": [4, 8, 12], "min_samples_leaf": [2, 3, 5], "max_features": ["sqrt", 0.3, 0.5], "bootstrap": [True, False], "class_weight": ["balanced"]},
        "lightgbm": {"n_estimators": [300, 500, 800], "learning_rate": [0.01, 0.03, 0.05], "num_leaves": [7, 15, 31], "max_depth": [4, 6, 8], "min_child_samples": [20, 40, 70], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0.0, 0.5, 2.0], "reg_lambda": [2.0, 5.0, 10.0]},
        "xgboost": {"n_estimators": [300, 500, 800], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [3, 4, 6], "min_child_weight": [3, 8, 15], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0.0, 0.5, 2.0], "reg_lambda": [3.0, 8.0, 15.0], "gamma": [0.0, 0.1, 0.3]},
        "knn": {"n_neighbors": [5, 7, 11, 15, 21, 31], "weights": ["uniform", "distance"], "p": [1, 2]},
    }
    return spaces[name]


def build_estimator(name, params, args, y):
    if name == "logistic":
        return LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000, random_state=args.random_state, **params)
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=args.random_state, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=args.n_jobs, random_state=args.random_state, **params)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_jobs=args.n_jobs, random_state=args.random_state, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_jobs=args.n_jobs, random_state=args.random_state, verbosity=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        negatives = max(int((y == 0).sum()), 1); positives = max(int((y == 1).sum()), 1)
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=args.n_jobs,
                             random_state=args.random_state, scale_pos_weight=negatives / positives, **params)
    if name == "knn":
        return KNeighborsClassifier(metric="minkowski", n_jobs=args.n_jobs, **params)
    raise KeyError(name)


def scaled(name): return name in {"logistic", "svm_rbf", "knn"}


def fit_predict(name, params, X_train, y_train, X_test, args):
    scaler = StandardScaler() if scaled(name) else None
    if scaler is not None:
        X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    model = build_estimator(name, params, args, y_train).fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1], model, scaler


def candidates(name, args, seed_offset=0):
    return list(ParameterSampler(parameter_space(name), n_iter=args.search_iterations,
                                 random_state=args.random_state + seed_offset))


def jsonable_params(params):
    output = {}
    for key, value in params.items():
        if isinstance(value, np.integer): value = int(value)
        elif isinstance(value, np.floating): value = float(value)
        elif isinstance(value, np.bool_): value = bool(value)
        output[key] = value
    return output


def inner_tune(core, outer_train, groups, budget, model_name, inner_splits, args, seed_offset):
    cached = []
    for inner_train_idx, inner_valid_idx in inner_splits:
        train = outer_train.iloc[inner_train_idx]; valid = outer_train.iloc[inner_valid_idx]
        y_train = train[core.TARGET].to_numpy(dtype=int); y_valid = valid[core.TARGET].to_numpy(dtype=int)
        selected, medians, _, _ = core.select_features(train, y_train, groups, budget, args)
        cached.append((core.matrix(train, selected, medians), y_train,
                       core.matrix(valid, selected, medians), y_valid))
    rows = []
    for candidate_id, params in enumerate(candidates(model_name, args, seed_offset), 1):
        params = jsonable_params(params)
        fold_mcc = []
        for X_train, y_train, X_valid, y_valid in cached:
            probability, _, _ = fit_predict(model_name, params, X_train, y_train, X_valid, args)
            fold_mcc.append(core.metrics(y_valid, probability, args.threshold)["mcc"])
        rows.append({"candidate_id": candidate_id, "params_json": json.dumps(params, sort_keys=True),
                     "inner_mean_mcc": float(np.mean(fold_mcc)), "inner_sd_mcc": float(np.std(fold_mcc, ddof=1))})
    result = pd.DataFrame(rows).sort_values(["inner_mean_mcc", "inner_sd_mcc"], ascending=[False, True])
    best = json.loads(result.iloc[0].params_json)
    return best, result


def search_all_configurations(core, train, models, budgets, inner_splits, args, seed_base):
    """Select budget, model and hyperparameters using inner folds only."""
    rows = []
    best_record = None
    for budget in budgets:
        groups = core.feature_groups(train, args.representation)
        for model_index, model_name in enumerate(models):
            params, search = inner_tune(
                core, train, groups, budget, model_name, inner_splits, args,
                seed_offset=seed_base + budget * 10 + model_index,
            )
            search["budget"] = budget; search["model"] = model_name
            rows.append(search)
            winner = search.sort_values(["inner_mean_mcc", "inner_sd_mcc"], ascending=[False, True]).iloc[0]
            record = {
                "budget": int(budget), "model": model_name, "params": params,
                "inner_mean_mcc": float(winner.inner_mean_mcc),
                "inner_sd_mcc": float(winner.inner_sd_mcc),
            }
            if best_record is None or (record["inner_mean_mcc"], -record["inner_sd_mcc"]) > (best_record["inner_mean_mcc"], -best_record["inner_sd_mcc"]):
                best_record = record
    return best_record, pd.concat(rows, ignore_index=True)


def bootstrap_ci(core, frame, y, probability, args):
    rng = np.random.default_rng(args.random_state)
    keys = frame[core.KEY].astype(str).to_numpy(); unique = np.unique(keys)
    mapping = {key: np.flatnonzero(keys == key) for key in unique}
    names = ["mcc", "balanced_accuracy", "roc_auc", "pr_auc"]
    samples = {name: [] for name in names}
    for _ in range(args.bootstrap):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([mapping[key] for key in chosen])
        values = core.metrics(y[idx], probability[idx], args.threshold)
        for name in names:
            if np.isfinite(values[name]): samples[name].append(values[name])
    output = {}
    for name in names:
        output[name + "_ci_low"] = np.percentile(samples[name], 2.5) if samples[name] else np.nan
        output[name + "_ci_high"] = np.percentile(samples[name], 97.5) if samples[name] else np.nan
    return output


def evaluate_external_groups(core, external, probability, args):
    parsed = external[args.ad_column].map(parse_bool)
    if parsed.isna().any(): raise ValueError("Invalid/missing external AD assignments")
    masks = {"Total": np.ones(len(external), dtype=bool), "Inside_AD": parsed.to_numpy(dtype=bool),
             "Outside_AD": ~parsed.to_numpy(dtype=bool)}
    y_all = external[core.TARGET].to_numpy(dtype=int); rows = []
    for group, mask in masks.items():
        y = y_all[mask]; p = probability[mask]; subset = external.loc[mask]
        row = {"analysis_group": group, "rows": len(subset), "unique_compounds": subset[core.KEY].nunique(),
               "active": int((y == 1).sum()), "inactive": int((y == 0).sum()),
               "active_prevalence": float(y.mean()) if len(y) else np.nan,
               "ad_coverage_percent": 100 * masks["Inside_AD"].sum() / len(external)}
        row.update(core.metrics(y, p, args.threshold)); row.update(bootstrap_ci(core, subset, y, p, args))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(nested_summary, external_metrics, output, args):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar([args.representation], nested_summary.mean_outer_mcc, color="#4C78A8")
    ax.errorbar([0], nested_summary.mean_outer_mcc, yerr=nested_summary.sd_outer_mcc,
                fmt="none", color="black", capsize=4)
    ax.set(xlabel="Molecular representation", ylabel="Nested grouped-CV MCC", title="Unbiased nested-CV performance")
    fig.tight_layout()
    fig.savefig(output / "figures/nested_cv_mcc_by_model.png", dpi=args.dpi, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=external_metrics, x="analysis_group", y="mcc", palette=["#4C78A8", "#59A14F", "#E15759"], ax=ax)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Strict external subset", ylabel="MCC", title="Frozen external performance by structural AD")
    fig.tight_layout(); fig.savefig(output / "figures/strict_external_mcc_by_ad.png", dpi=args.dpi, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args(); start = time.time(); core = load_core(args.core_script)
    for folder in ["metrics", "inner_search", "selected_features", "final_models", "predictions", "figures", "audit"]:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    development = core.read_data(args.development); external = core.read_data(args.external)
    if args.ad_column not in external: raise KeyError(f"External data lack {args.ad_column}")
    overlap = set(development[core.KEY]) & set(external[core.KEY])
    if overlap: raise ValueError(f"External set overlaps development by {len(overlap)} compounds")
    models = available_models(args.models); molecular_groups = core.feature_groups(development, args.representation)
    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.random_state)
    outer_splits = list(outer.split(development, development[core.TARGET], development[core.KEY]))
    checkpoint = args.output_dir / "metrics/nested_outer_fold_metrics_checkpoint.csv"
    completed = pd.read_csv(checkpoint) if args.resume and checkpoint.is_file() else pd.DataFrame()
    outer_rows, search_rows = completed.to_dict("records"), []

    total = len(outer_splits); done = len(outer_rows)
    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer_splits, 1):
        if not completed.empty and (completed.outer_fold == outer_fold).any():
            continue
        outer_train = development.iloc[outer_train_idx].reset_index(drop=True)
        outer_valid = development.iloc[outer_valid_idx].reset_index(drop=True)
        groups = core.feature_groups(outer_train, args.representation)
        inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.random_state + outer_fold)
        inner_splits = list(inner.split(outer_train, outer_train[core.TARGET], outer_train[core.KEY]))
        chosen, search = search_all_configurations(core, outer_train, models, args.budgets, inner_splits, args,
                                                    seed_base=outer_fold * 10000)
        search["outer_fold"] = outer_fold; search_rows.append(search)
        search.to_csv(args.output_dir / f"inner_search/outer_fold_{outer_fold}_search.csv", index=False)
        budget, model_name, best_params = chosen["budget"], chosen["model"], chosen["params"]
        y_train = outer_train[core.TARGET].to_numpy(dtype=int); y_valid = outer_valid[core.TARGET].to_numpy(dtype=int)
        selected, medians, _, allocation = core.select_features(outer_train, y_train, groups, budget, args)
        X_train = core.matrix(outer_train, selected, medians); X_valid = core.matrix(outer_valid, selected, medians)
        valid_prob, model, scaler = fit_predict(model_name, best_params, X_train, y_train, X_valid, args)
        train_X = scaler.transform(X_train) if scaler else X_train
        train_prob = model.predict_proba(train_X)[:, 1]
        valid_metrics = core.metrics(y_valid, valid_prob, args.threshold)
        train_metrics = core.metrics(y_train, train_prob, args.threshold)
        row = {"representation": args.representation, "outer_fold": outer_fold, "budget": budget,
               "model": model_name, "best_params_json": json.dumps(best_params, sort_keys=True),
               "inner_selected_mean_mcc": chosen["inner_mean_mcc"], "inner_selected_sd_mcc": chosen["inner_sd_mcc"],
               "selected_molecular_features": len(selected), "mutation_features": len(core.MUTATIONS),
               "feature_allocation_json": json.dumps(allocation, sort_keys=True), "train_mcc": train_metrics["mcc"],
               "train_outer_gap": train_metrics["mcc"] - valid_metrics["mcc"]}
        row.update(valid_metrics); outer_rows.append(row); pd.DataFrame(outer_rows).to_csv(checkpoint, index=False)
        done += 1; elapsed = time.time() - start; eta = elapsed / max(done, 1) * (total - done) / 3600
        print(f"[{done}/{total}] outer={outer_fold} selected={model_name}/budget={budget} MCC={valid_metrics['mcc']:.3f} ETA={eta:.1f}h", flush=True)
    outer_df = pd.DataFrame(outer_rows); outer_df.to_csv(args.output_dir / "metrics/nested_outer_fold_metrics.csv", index=False)
    if search_rows: pd.concat(search_rows, ignore_index=True).to_csv(args.output_dir / "inner_search/all_inner_search_results.csv", index=False)
    summary = pd.DataFrame([{"representation": args.representation, "outer_folds": outer_df.outer_fold.nunique(),
        "mean_outer_mcc": outer_df.mcc.mean(), "sd_outer_mcc": outer_df.mcc.std(),
        "mean_outer_balanced_accuracy": outer_df.balanced_accuracy.mean(), "mean_outer_roc_auc": outer_df.roc_auc.mean(),
        "mean_outer_pr_auc": outer_df.pr_auc.mean(), "mean_outer_f1": outer_df.f1.mean(),
        "mean_train_mcc": outer_df.train_mcc.mean(), "mean_train_outer_gap": outer_df.train_outer_gap.mean()}])
    summary.to_csv(args.output_dir / "metrics/nested_cv_summary.csv", index=False)

    # Final configuration selection uses development only. Outer results are not
    # used here.  Restore this expensive search after an interrupted final
    # external transformation when --resume is supplied.
    final_search_path = args.output_dir / "inner_search/final_winner_development_search.csv"
    final_winner_path = args.output_dir / "metrics/development_selected_winner.csv"
    if args.resume and final_search_path.is_file() and final_winner_path.is_file():
        saved_winner = pd.read_csv(final_winner_path, low_memory=False).iloc[0]
        winner_model = str(saved_winner["model"])
        winner_budget = int(saved_winner["budget"])
        params_value = saved_winner["params"]
        final_params = json.loads(params_value) if isinstance(params_value, str) else {}
        print(
            f"Restored final development winner: {winner_model}/budget={winner_budget}",
            flush=True,
        )
    else:
        final_inner = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.random_state + 999)
        final_splits = list(final_inner.split(development, development[core.TARGET], development[core.KEY]))
        final_chosen, final_search = search_all_configurations(core, development, models, args.budgets, final_splits, args,
                                                               seed_base=99999)
        winner_model, winner_budget, final_params = final_chosen["model"], final_chosen["budget"], final_chosen["params"]
        final_search.to_csv(final_search_path, index=False)
        pd.DataFrame([{k: (json.dumps(v, sort_keys=True) if k == "params" else v) for k, v in final_chosen.items()}]).to_csv(
            final_winner_path, index=False)
    y_dev = development[core.TARGET].to_numpy(dtype=int); y_ext = external[core.TARGET].to_numpy(dtype=int)
    selected, medians, audit, allocation = core.select_features(development, y_dev, molecular_groups, winner_budget, args)
    audit.to_csv(args.output_dir / "selected_features/final_feature_audit.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv(args.output_dir / "selected_features/final_selected_features.csv", index=False)
    # Publication audit: record selected-feature values that cannot be safely
    # represented by downstream float32 tree implementations. These values are
    # converted to missing and imputed only with development-derived medians.
    invalid_records = []
    for dataset_name, frame in [("development", development), ("strict_external", external)]:
        raw = frame[selected].apply(pd.to_numeric, errors="coerce")
        invalid = raw.notna() & ((~np.isfinite(raw)) | (raw.abs() > core.FLOAT32_SAFE_ABS))
        for feature in selected:
            positions = np.flatnonzero(invalid[feature].to_numpy())
            for position in positions:
                invalid_records.append({
                    "dataset": dataset_name,
                    "row_index": int(position),
                    core.KEY: str(frame.iloc[position][core.KEY]),
                    "feature": feature,
                    "raw_value": str(raw.iloc[position][feature]),
                    "action": "set_to_missing_then_impute_with_development_median",
                })
    pd.DataFrame(
        invalid_records,
        columns=["dataset", "row_index", core.KEY, "feature", "raw_value", "action"],
    ).to_csv(args.output_dir / "audit/nonfinite_or_float32_overflow_selected_values.csv", index=False)
    X_dev = core.matrix(development, selected, medians); X_ext = core.matrix(external, selected, medians)
    ext_prob, model, scaler = fit_predict(winner_model, final_params, X_dev, y_dev, X_ext, args)
    bundle = {"model": model, "scaler": scaler, "selected_molecular_features": selected,
              "mutation_features": core.MUTATIONS, "imputation_medians": medians, "threshold": args.threshold,
              "representation": args.representation, "model_name": winner_model, "budget": winner_budget,
              "hyperparameters": final_params, "feature_group_allocation": allocation, "target": core.TARGET,
              "key": core.KEY, "ad_column": args.ad_column}
    joblib.dump(bundle, args.output_dir / f"final_models/{args.representation}__{winner_model}__budget_{winner_budget}.joblib", compress=3)
    predictions = external[[core.KEY, core.CONTEXT, args.ad_column]].copy()
    predictions["observed"] = y_ext; predictions["probability_active"] = ext_prob
    predictions["predicted"] = (ext_prob >= args.threshold).astype(int)
    predictions.to_csv(args.output_dir / "predictions/strict_external_predictions.csv", index=False)
    external_metrics = evaluate_external_groups(core, external, ext_prob, args)
    external_metrics.insert(0, "representation", args.representation); external_metrics.insert(1, "model", winner_model)
    external_metrics.insert(2, "budget", winner_budget)
    external_metrics.to_csv(args.output_dir / "metrics/strict_external_metrics_total_inside_outside_ad.csv", index=False)
    plot_outputs(summary.iloc[0], external_metrics, args.output_dir, args)
    configuration = vars(args).copy(); configuration = {k: str(v) if isinstance(v, Path) else v for k, v in configuration.items()}
    configuration.update({"winner_model": winner_model, "winner_budget": winner_budget, "final_hyperparameters": final_params})
    (args.output_dir / "run_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(f"Nested-CV winner: {winner_model}, budget={winner_budget}")
    print(external_metrics[["analysis_group", "rows", "mcc", "balanced_accuracy", "roc_auc", "pr_auc"]].to_string(index=False))
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
