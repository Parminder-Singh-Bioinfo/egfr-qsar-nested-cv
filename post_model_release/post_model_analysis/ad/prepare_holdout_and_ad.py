#!/usr/bin/env python3
"""Create a compound-disjoint external holdout and assess structural AD.

Inputs are the aggregated modeling tables created by
03_aggregate_egfr_activity.py. Any external compound whose InChIKey_STD is
present in development is removed from the strict holdout regardless of its
mutation context. Morgan radius-2/2048-bit fingerprints are used only for the
structural applicability-domain assessment.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase

try:
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # compatibility with older RDKit installations
    rdFingerprintGenerator = None
    from rdkit.Chem import AllChem


KEY = "InChIKey_STD"
CONTEXT = "Mutation_clean"
SMILES = "SMILES_STD"
TARGET = "Activity_binary"
PIC50 = "pIC50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit development/external overlap, make a compound-disjoint "
            "external holdout, and calculate Morgan-Tanimoto AD coverage."
        )
    )
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--ad-percentile",
        type=float,
        default=5.0,
        help=(
            "Percentile of development leave-one-out nearest-neighbor "
            "similarities used as the AD threshold (default: 5)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    required = [KEY, CONTEXT, SMILES, TARGET, PIC50]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing required columns {missing}")
    if df[required].isna().any().any():
        counts = df[required].isna().sum()
        raise ValueError(
            f"{path}: required columns contain missing values: "
            f"{counts[counts.gt(0)].to_dict()}"
        )
    duplicate_context = df.duplicated([KEY, CONTEXT], keep=False)
    if duplicate_context.any():
        raise ValueError(
            f"{path}: {int(duplicate_context.sum())} rows duplicate "
            f"{KEY}+{CONTEXT}; aggregation must produce one context row."
        )
    return df


def unique_compounds(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    smiles_per_key = df.groupby(KEY)[SMILES].nunique(dropna=True)
    inconsistent = smiles_per_key[smiles_per_key.gt(1)]
    if not inconsistent.empty:
        raise ValueError(
            f"{dataset}: {len(inconsistent)} standardized keys have multiple "
            "standardized SMILES values."
        )
    compounds = (
        df[[KEY, SMILES]]
        .drop_duplicates(KEY)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    compounds["dataset"] = dataset
    return compounds


def fingerprint(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse standardized SMILES: {smiles}")
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        return generator.GetFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )


def make_fingerprints(
    compounds: pd.DataFrame, radius: int, n_bits: int
) -> List[object]:
    fps = []
    errors = []
    for row_number, row in compounds.iterrows():
        try:
            fps.append(fingerprint(row[SMILES], radius, n_bits))
        except Exception as exc:
            errors.append(
                {
                    KEY: row[KEY],
                    SMILES: row[SMILES],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            fps.append(None)
    if errors:
        raise ValueError(
            f"Fingerprint generation failed for {len(errors)} compounds; "
            f"first examples: {errors[:3]}"
        )
    return fps


def development_loo_nearest(fps: Sequence[object]) -> np.ndarray:
    if len(fps) < 2:
        raise ValueError("At least two development compounds are required")
    nearest = np.zeros(len(fps), dtype=float)
    for index, query in enumerate(fps):
        # Excluding the query itself makes this a leave-one-compound-out density.
        similarities = DataStructs.BulkTanimotoSimilarity(query, fps)
        similarities[index] = -1.0
        nearest[index] = float(max(similarities))
    return nearest


def external_nearest(
    external_fps: Sequence[object],
    development_fps: Sequence[object],
    development_keys: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    maximum = np.zeros(len(external_fps), dtype=float)
    nearest_keys: List[str] = []
    for index, query in enumerate(external_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(query, development_fps)
        best_index = int(np.argmax(similarities))
        maximum[index] = float(similarities[best_index])
        nearest_keys.append(str(development_keys[best_index]))
    return maximum, nearest_keys


def exact_context_comparison(
    development: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    dev_columns = [KEY, CONTEXT, SMILES, PIC50, TARGET]
    ext_columns = [KEY, CONTEXT, SMILES, PIC50, TARGET]
    comparison = development[dev_columns].merge(
        external[ext_columns],
        on=[KEY, CONTEXT],
        how="inner",
        suffixes=("_development", "_external"),
        validate="one_to_one",
    )
    if comparison.empty:
        comparison["absolute_pIC50_difference"] = pd.Series(dtype=float)
        comparison["activity_label_agreement"] = pd.Series(dtype=bool)
        comparison["standardized_smiles_identical"] = pd.Series(dtype=bool)
        return comparison
    comparison["absolute_pIC50_difference"] = (
        comparison[f"{PIC50}_development"]
        - comparison[f"{PIC50}_external"]
    ).abs()
    comparison["activity_label_agreement"] = (
        comparison[f"{TARGET}_development"]
        == comparison[f"{TARGET}_external"]
    )
    comparison["standardized_smiles_identical"] = (
        comparison[f"{SMILES}_development"]
        == comparison[f"{SMILES}_external"]
    )
    return comparison.sort_values(
        ["activity_label_agreement", "absolute_pIC50_difference"],
        ascending=[True, False],
    )


def save_similarity_plots(
    dev_nearest: np.ndarray,
    external_maximum: np.ndarray,
    ad_threshold: float,
    output_dir: Path,
    dpi: int,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.2, 5.5))
    bins = np.linspace(0, 1, 41)
    plt.hist(
        dev_nearest,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Development LOO nearest neighbor",
        color="#2878B5",
    )
    plt.hist(
        external_maximum,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Strict external nearest development",
        color="#E07A1F",
    )
    plt.axvline(
        ad_threshold,
        color="black",
        linestyle="--",
        linewidth=1.7,
        label=f"AD threshold = {ad_threshold:.3f}",
    )
    plt.xlabel("Maximum Morgan Tanimoto similarity")
    plt.ylabel("Density")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "tanimoto_nearest_neighbor_distributions.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8.2, 5.5))
    sorted_values = np.sort(external_maximum)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cumulative, color="#009E73", linewidth=2)
    plt.axvline(ad_threshold, color="black", linestyle="--", linewidth=1.7)
    plt.xlabel("Maximum external-to-development Tanimoto similarity")
    plt.ylabel("Cumulative proportion of strict external compounds")
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "strict_external_tanimoto_ecdf.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    args = parse_args()
    if not 0 <= args.ad_percentile <= 100:
        raise ValueError("--ad-percentile must be between 0 and 100")
    if args.radius < 0 or args.n_bits <= 0:
        raise ValueError("Fingerprint radius must be >=0 and n-bits must be >0")

    development_path = args.development.resolve()
    external_path = args.external.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development = read_table(development_path)
    external = read_table(external_path)
    dev_compounds = unique_compounds(development, "development")
    ext_compounds = unique_compounds(external, "external_before_overlap_removal")

    development_keys = set(dev_compounds[KEY])
    overlapping_keys = development_keys.intersection(set(ext_compounds[KEY]))
    external_overlap_rows = external.loc[external[KEY].isin(overlapping_keys)].copy()
    external_strict = external.loc[~external[KEY].isin(overlapping_keys)].copy()
    if external_strict.empty:
        raise ValueError(
            "No external rows remain after compound-overlap removal; "
            "a strict external holdout cannot be constructed."
        )
    ext_strict_compounds = unique_compounds(external_strict, "strict_external")

    comparison = exact_context_comparison(development, external)
    overlap_compounds = ext_compounds.loc[
        ext_compounds[KEY].isin(overlapping_keys)
    ].copy()
    overlap_context_counts = (
        external_overlap_rows.groupby(KEY)
        .size()
        .rename("external_context_rows")
        .reset_index()
    )
    overlap_compounds = overlap_compounds.merge(
        overlap_context_counts, on=KEY, how="left", validate="one_to_one"
    )

    dev_fps = make_fingerprints(dev_compounds, args.radius, args.n_bits)
    ext_fps = make_fingerprints(ext_strict_compounds, args.radius, args.n_bits)
    dev_nearest = development_loo_nearest(dev_fps)
    ad_threshold = float(np.percentile(dev_nearest, args.ad_percentile))
    ext_maximum, nearest_keys = external_nearest(
        ext_fps, dev_fps, dev_compounds[KEY].astype(str).tolist()
    )

    development_ad = dev_compounds.copy()
    development_ad["development_loo_max_tanimoto"] = dev_nearest
    ext_strict_compounds["max_tanimoto_to_development"] = ext_maximum
    ext_strict_compounds["nearest_development_InChIKey_STD"] = nearest_keys
    ext_strict_compounds["AD_threshold"] = ad_threshold
    ext_strict_compounds["Inside_AD"] = ext_strict_compounds[
        "max_tanimoto_to_development"
    ].ge(ad_threshold)

    annotations = ext_strict_compounds[
        [
            KEY,
            "max_tanimoto_to_development",
            "nearest_development_InChIKey_STD",
            "AD_threshold",
            "Inside_AD",
        ]
    ]
    external_strict_annotated = external_strict.merge(
        annotations, on=KEY, how="left", validate="many_to_one"
    )
    external_inside = external_strict_annotated.loc[
        external_strict_annotated["Inside_AD"]
    ].copy()
    external_outside = external_strict_annotated.loc[
        ~external_strict_annotated["Inside_AD"]
    ].copy()

    thresholds = np.round(np.arange(0.0, 1.01, 0.05), 2)
    coverage = pd.DataFrame(
        {
            "tanimoto_threshold": thresholds,
            "external_compounds_at_or_above": [
                int(np.sum(ext_maximum >= threshold)) for threshold in thresholds
            ],
            "external_compound_coverage_percent": [
                float(np.mean(ext_maximum >= threshold) * 100)
                for threshold in thresholds
            ],
        }
    )

    paths: Dict[str, Path] = {
        "strict_external": output_dir / "external_strict_no_compound_overlap.csv",
        "strict_external_ad": output_dir / "external_strict_with_ad.csv",
        "inside_ad": output_dir / "external_strict_inside_ad.csv",
        "outside_ad": output_dir / "external_strict_outside_ad.csv",
        "overlap_rows": output_dir / "external_rows_removed_compound_overlap.csv",
        "overlap_compounds": output_dir / "overlapping_compounds.csv",
        "exact_context": output_dir / "exact_context_overlap_comparison.csv",
        "development_ad": output_dir / "development_loo_nearest_neighbors.csv",
        "external_compound_ad": output_dir / "external_compound_ad_annotations.csv",
        "coverage": output_dir / "external_ad_coverage_by_threshold.csv",
        "summary_csv": output_dir / "holdout_ad_summary.csv",
        "summary_json": output_dir / "holdout_ad_manifest.json",
    }

    external_strict.to_csv(paths["strict_external"], index=False)
    external_strict_annotated.to_csv(paths["strict_external_ad"], index=False)
    external_inside.to_csv(paths["inside_ad"], index=False)
    external_outside.to_csv(paths["outside_ad"], index=False)
    external_overlap_rows.to_csv(paths["overlap_rows"], index=False)
    overlap_compounds.to_csv(paths["overlap_compounds"], index=False)
    comparison.to_csv(paths["exact_context"], index=False)
    development_ad.to_csv(paths["development_ad"], index=False)
    ext_strict_compounds.to_csv(paths["external_compound_ad"], index=False)
    coverage.to_csv(paths["coverage"], index=False)

    inside_compounds = int(ext_strict_compounds["Inside_AD"].sum())
    outside_compounds = int(len(ext_strict_compounds) - inside_compounds)
    summary = {
        "development_rows": int(len(development)),
        "development_unique_compounds": int(len(dev_compounds)),
        "external_rows_before_overlap_removal": int(len(external)),
        "external_unique_compounds_before_overlap_removal": int(len(ext_compounds)),
        "overlapping_compounds_removed": int(len(overlapping_keys)),
        "external_rows_removed_for_compound_overlap": int(len(external_overlap_rows)),
        "exact_compound_mutation_context_overlaps": int(len(comparison)),
        "strict_external_rows": int(len(external_strict)),
        "strict_external_unique_compounds": int(len(ext_strict_compounds)),
        "morgan_radius": int(args.radius),
        "morgan_bits": int(args.n_bits),
        "ad_threshold_percentile": float(args.ad_percentile),
        "ad_threshold": ad_threshold,
        "external_median_max_tanimoto": float(np.median(ext_maximum)),
        "inside_ad_unique_compounds": inside_compounds,
        "outside_ad_unique_compounds": outside_compounds,
        "inside_ad_compound_coverage_percent": float(
            inside_compounds / len(ext_strict_compounds) * 100
        ),
        "strict_external_active_rows": int(
            external_strict[TARGET].astype(int).sum()
        ),
        "strict_external_inactive_rows": int(
            len(external_strict) - external_strict[TARGET].astype(int).sum()
        ),
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    ).to_csv(paths["summary_csv"], index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdBase.rdkitVersion,
        "development_input": str(development_path),
        "external_input": str(external_path),
        "compound_overlap_policy": (
            "Remove every external row when its InChIKey_STD occurs anywhere "
            "in development, regardless of mutation context."
        ),
        "ad_definition": (
            "Inside AD when external maximum Morgan Tanimoto similarity to "
            f"development is >= the {args.ad_percentile:g}th percentile of "
            "development leave-one-compound-out nearest-neighbor similarities."
        ),
        "summary": summary,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    save_similarity_plots(
        dev_nearest, ext_maximum, ad_threshold, output_dir, args.dpi
    )

    print("Strict external holdout and AD assessment completed.")
    print(
        f"Development: {len(development):,} rows, "
        f"{len(dev_compounds):,} unique compounds"
    )
    print(
        f"External before removal: {len(external):,} rows, "
        f"{len(ext_compounds):,} unique compounds"
    )
    print(
        f"Overlapping compounds removed: {len(overlapping_keys):,} "
        f"({len(external_overlap_rows):,} external rows)"
    )
    print(
        f"Strict external: {len(external_strict):,} rows, "
        f"{len(ext_strict_compounds):,} unique compounds"
    )
    print(
        f"Median max Tanimoto: {np.median(ext_maximum):.3f} | "
        f"AD threshold: {ad_threshold:.3f} | "
        f"coverage: {inside_compounds / len(ext_strict_compounds) * 100:.1f}%"
    )
    print(f"Strict external dataset: {paths['strict_external_ad']}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
