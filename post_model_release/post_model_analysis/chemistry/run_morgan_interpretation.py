
#!/usr/bin/env python3
"""
EGFR Morgan fingerprint chemical interpretation.

Outputs:
  audit/       fingerprint validation and row-matching reports
  tables/      bit environments, collisions, TP/FN statistics
  figures/     highlighted molecular structures and summary plots
  input/       input manifest
  run_configuration.json

IMPORTANT:
  - Requires a verified structure-to-external-row mapping.
  - Does not assume a bit has a unique chemical meaning.
  - Stops before chemical interpretation if fingerprint
    reconstruction fails.
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


BITS = [361, 130, 959, 1366, 489, 491, 1816, 969, 694, 708]
RADIUS = 2
NBITS = 2048

TP_COLOR = "#2474A6"
FN_COLOR = "#D97935"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--smiles-csv", required=True)
    p.add_argument("--smiles-column", required=True)

    p.add_argument("--root", default=str(
        Path.home() / "projects/EGFR_ALL_SHAP_RESULTS"
        / "01_Single_Protein_Morgan"
    ))

    p.add_argument("--output", default=str(
        Path.home() / "projects/EGFR_MORGAN_CHEMICAL_INTERPRETATION"
    ))

    p.add_argument(
        "--mapping",
        choices=["row_id", "inchikey"],
        default="row_id"
    )

    p.add_argument("--chirality", action="store_true")
    p.add_argument("--use-features", action="store_true")
    p.add_argument(
        "--use-bond-types",
        dest="use_bond_types",
        action="store_true",
        default=True,
        help="Use bond types in Morgan fingerprint generation (default)"
    )

    p.add_argument(
        "--no-use-bond-types",
        dest="use_bond_types",
        action="store_false",
        help="Ignore bond types in Morgan fingerprint generation"
    )

    p.add_argument("--max-examples", type=int, default=3)
    p.add_argument("--bits", nargs="+", type=int, default=BITS)

    return p.parse_args()


def fingerprint(mol, args):
    info = {}

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=NBITS,
        useChirality=args.chirality,
        useFeatures=args.use_features,
        useBondTypes=args.use_bond_types,
        bitInfo=info
    )

    return fp, info


def environment_fragment(mol, center, radius):
    """
    Extract an atom-centered environment with bond topology.

    Returns:
      fragment SMILES
      atom indices
      bond indices

    Radius-zero environments are represented by the center atom.
    """
    if radius == 0:
        atom = mol.GetAtomWithIdx(center)

        # Include atom properties in the radius-zero identity.
        frag = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=[center],
            canonical=True,
            isomericSmiles=True
        )

        return frag, [center], []

    bonds = list(
        Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    )

    atoms = {center}

    for bidx in bonds:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())

    atoms = sorted(atoms)

    frag = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atoms,
        bondsToUse=bonds,
        canonical=True,
        isomericSmiles=True
    )

    return frag, atoms, bonds


def save_molecule(mol, atoms, bonds, path, legend):
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 550)

    options = drawer.drawOptions()
    options.addAtomIndices = False

    drawer.DrawMolecule(
        mol,
        legend=legend,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors={
            a: (1.0, 0.65, 0.25) for a in atoms
        },
        highlightBondColors={
            b: (1.0, 0.65, 0.25) for b in bonds
        }
    )

    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def save_plot(fig, folder, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            folder / f"{stem}.{ext}",
            dpi=400,
            bbox_inches="tight"
        )
    plt.close(fig)


def main():
    args = parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.output).expanduser()

    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be new or empty")

    audit = out / "audit"
    tables = out / "tables"
    figures = out / "figures"
    individual = figures / "individual_bits"
    manuscript = figures / "manuscript"
    input_dir = out / "input"

    for folder in [
        audit, tables, figures, individual, manuscript, input_dir
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    smiles_path = Path(args.smiles_csv).expanduser()

    labels_path = (
        root / "Error_Analysis"
        / "external_predictions_with_error_groups.csv"
    )

    matrix_path = (
        root / "Main_SHAP"
        / "shap_molecular_inputs.csv"
    )

    comparison_path = (
        root / "Error_Analysis/Full_External"
        / "TP_vs_FN_feature_comparison.csv"
    )

    for path in [
        smiles_path, labels_path, matrix_path, comparison_path
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    structures = pd.read_csv(smiles_path)
    labels = pd.read_csv(labels_path)
    matrix = pd.read_csv(matrix_path)
    comparison = pd.read_csv(comparison_path)

    if args.smiles_column not in structures.columns:
        raise ValueError(
            f"SMILES column not found: {args.smiles_column}"
        )

    if "external_row_index" not in labels.columns:
        raise ValueError("Classification file lacks external_row_index")

    if not labels["external_row_index"].is_unique:
        raise ValueError("Duplicate external_row_index in labels")

    if args.mapping == "row_id":
        if "external_row_index" not in structures.columns:
            raise ValueError(
                "SMILES CSV must contain external_row_index "
                "for row_id mapping"
            )

        if not structures["external_row_index"].is_unique:
            raise ValueError(
                "SMILES CSV has duplicate external_row_index"
            )

        joined = labels.merge(
            structures[
                ["external_row_index", args.smiles_column]
            ],
            on="external_row_index",
            how="left",
            validate="one_to_one"
        )

    else:
        if "InChIKey_STD" not in structures.columns:
            raise ValueError(
                "SMILES CSV must contain InChIKey_STD"
            )

        # An InChIKey can have multiple mutation-context observations.
        # We therefore require one unambiguous standardized SMILES
        # per InChIKey rather than merging all assay rows directly.
        unique_structures = structures[
            ["InChIKey_STD", args.smiles_column]
        ].dropna().drop_duplicates()

        counts = unique_structures.groupby(
            "InChIKey_STD"
        )[args.smiles_column].nunique()

        ambiguous = counts[counts > 1]

        if len(ambiguous):
            raise ValueError(
                f"{len(ambiguous)} InChIKeys map to multiple "
                "SMILES strings. Resolve them before proceeding."
            )

        unique_structures = unique_structures.drop_duplicates(
            "InChIKey_STD"
        )

        joined = labels.merge(
            unique_structures,
            on="InChIKey_STD",
            how="left",
            validate="many_to_one"
        )

    if joined[args.smiles_column].isna().any():
        raise ValueError(
            "Some external observations have no matched SMILES"
        )

    if len(joined) != len(matrix):
        raise ValueError(
            "External observation and feature-matrix counts differ"
        )

    # Matrix is assumed to follow external_row_index 0..N-1.
    # This assumption must be supported by the original
    # feature-matrix export workflow.
    expected_ids = np.arange(len(matrix))

    if not np.array_equal(
        joined["external_row_index"].to_numpy(),
        expected_ids
    ):
        raise ValueError(
            "External IDs do not match feature-matrix row positions. "
            "A verified feature-matrix row-ID mapping is required."
        )

    selected_columns = {
        bit: f"rdkit_morgan_r2_2048bit_{bit:04d}"
        for bit in args.bits
    }

    missing = [
        col for col in selected_columns.values()
        if col not in matrix.columns
    ]

    if missing:
        raise ValueError(
            f"Selected fingerprint columns missing: {missing}"
        )

    # ------------------------------------------------------------
    # Reconstruct fingerprints and audit against saved model inputs
    # ------------------------------------------------------------

    mols = []
    infos = []
    audit_rows = []

    for i, row in joined.iterrows():
        smi = str(row[args.smiles_column])
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES at external row {i}: {smi}"
            )

        fp, info = fingerprint(mol, args)

        for bit, col in selected_columns.items():
            observed = matrix.iloc[i][col]
            if observed not in (0, 1):
                raise ValueError("Saved bit value must be exactly zero or one")
            reconstructed = int(fp.GetBit(bit))

            audit_rows.append({
                "external_row_index": int(row["external_row_index"]),
                "bit": bit,
                "saved_value": observed,
                "reconstructed_value": reconstructed,
                "match": observed == reconstructed
            })

        mols.append(mol)
        infos.append(info)

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(
        audit / "selected_bit_reconstruction_audit.csv",
        index=False
    )

    mismatch_count = int((~audit_df["match"]).sum())

    print("Selected-bit reconstruction mismatches:", mismatch_count)

    if mismatch_count:
        raise RuntimeError(
            "Fingerprint reconstruction FAILED. "
            "Inspect audit/selected_bit_reconstruction_audit.csv "
            "and verify the original fingerprint settings, "
            "standardization, and row mapping."
        )

    # ------------------------------------------------------------
    # Extract environments and possible hash collisions
    # ------------------------------------------------------------

    env_rows = []
    occurrences = defaultdict(list)

    for i, (mol, info) in enumerate(zip(mols, infos)):
        row = joined.iloc[i]

        for bit in args.bits:
            for center, radius in info.get(bit, []):
                fragment, atoms, bonds = environment_fragment(
                    mol, center, radius
                )

                # This key identifies a distinct extracted
                # environment representation, not an original
                # un-hashed Morgan identifier.
                env_key = f"r{radius}:{fragment}"

                record = {
                    "external_row_index": int(
                        row["external_row_index"]
                    ),
                    "InChIKey_STD": row.get("InChIKey_STD", ""),
                    "Mutation_clean": row.get("Mutation_clean", ""),
                    "group": row["group"],
                    "Inside_AD": row.get("Inside_AD", ""),
                    "Scaffold_status": row.get(
                        "Scaffold_status", ""
                    ),
                    "bit": bit,
                    "center_atom": center,
                    "radius": radius,
                    "fragment_smiles": fragment,
                    "environment_key": env_key
                }

                env_rows.append(record)

                occurrences[(bit, env_key)].append(
                    (i, center, radius, atoms, bonds)
                )

    env_df = pd.DataFrame(env_rows)

    env_df.to_csv(
        tables / "morgan_atom_environments.csv",
        index=False
    )

    if env_df.empty:
        raise RuntimeError(
            "No environments found despite matching selected bits"
        )

    summary = (
        env_df.groupby(
            ["bit", "environment_key", "fragment_smiles", "radius"]
        )
        .agg(
            environment_occurrences=(
                "external_row_index", "size"
            ),
            unique_observations=(
                "external_row_index", "nunique"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        tables / "environment_frequency.csv",
        index=False
    )

    collision = (
        summary.groupby("bit")
        .agg(
            distinct_environment_representations=(
                "environment_key", "nunique"
            ),
            total_environment_occurrences=(
                "environment_occurrences", "sum"
            )
        )
        .reset_index()
    )

    collision["multiple_environments_detected"] = (
        collision["distinct_environment_representations"] > 1
    )

    collision.to_csv(
        tables / "bit_environment_diversity.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Bit prevalence in TP and FN
    # ------------------------------------------------------------

    prevalence_rows = []

    for bit, col in selected_columns.items():
        for group in ["TP", "FN"]:
            mask = joined["group"].eq(group).to_numpy()
            values = matrix.loc[mask, col].to_numpy()

            prevalence_rows.append({
                "bit": bit,
                "group": group,
                "n": int(mask.sum()),
                "bit_present_n": int((values == 1).sum()),
                "prevalence_percent": float(
                    100 * np.mean(values == 1)
                )
            })

    prevalence = pd.DataFrame(prevalence_rows)

    prevalence.to_csv(
        tables / "tp_fn_bit_prevalence.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Highlight representative molecular environments
    # ------------------------------------------------------------

    representative_rows = []

    for bit in args.bits:
        bit_summary = summary[
            summary["bit"] == bit
        ].sort_values(
            "unique_observations",
            ascending=False
        )

        for rank, (_, env) in enumerate(
            bit_summary.head(args.max_examples).iterrows(),
            start=1
        ):
            key = (bit, env["environment_key"])
            candidates = occurrences[key]

            # Prefer different compound IDs when possible.
            used = set()
            chosen = None

            for item in candidates:
                idx = item[0]
                compound = joined.iloc[idx].get(
                    "InChIKey_STD", idx
                )

                if compound not in used:
                    chosen = item
                    used.add(compound)
                    break

            if chosen is None:
                continue

            idx, center, radius, atoms, bonds = chosen

            filename = (
                f"bit_{bit:04d}_environment_{rank:02d}.png"
            )

            save_molecule(
                mols[idx],
                atoms,
                bonds,
                individual / filename,
                legend=(
                    f"Bit {bit:04d} | radius {radius} | "
                    f"{joined.iloc[idx]['group']}"
                )
            )

            representative_rows.append({
                "bit": bit,
                "environment_rank": rank,
                "external_row_index": int(
                    joined.iloc[idx]["external_row_index"]
                ),
                "group": joined.iloc[idx]["group"],
                "fragment_smiles": env["fragment_smiles"],
                "radius": radius,
                "image_file": filename
            })

    pd.DataFrame(representative_rows).to_csv(
        tables / "representative_environment_images.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Manuscript figure: TP/FN prevalence
    # ------------------------------------------------------------

    wide = prevalence.pivot(
        index="bit",
        columns="group",
        values="prevalence_percent"
    ).reindex(args.bits)

    y = np.arange(len(wide))
    h = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.barh(
        y + h/2, wide["TP"],
        height=h, color=TP_COLOR, label="TP"
    )

    ax.barh(
        y - h/2, wide["FN"],
        height=h, color=FN_COLOR, label="FN"
    )

    ax.set_yticks(y)
    ax.set_yticklabels([
        f"Bit {bit:04d}" for bit in wide.index
    ])

    ax.invert_yaxis()
    ax.set_xlabel("Bit prevalence (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False)

    fig.tight_layout()

    save_plot(
        fig,
        manuscript,
        "morgan_tp_fn_prevalence"
    )

    # ------------------------------------------------------------
    # Save run configuration and source manifest
    # ------------------------------------------------------------

    config = {
        "rdkit_version": rdBase.rdkitVersion,
        "radius": RADIUS,
        "nBits": NBITS,
        "useChirality": args.chirality,
        "useFeatures": args.use_features,
        "useBondTypes": args.use_bond_types,
        "selected_bits": args.bits,
        "smiles_column": args.smiles_column,
        "mapping_method": args.mapping,
        "fingerprint_reconstruction_mismatches": mismatch_count,
        "note": (
            "Environment diversity is a conservative indicator "
            "of possible hashed-bit ambiguity. It is not proof "
            "that every distinct representation is an unrelated "
            "chemical feature."
        )
    }

    (out / "run_configuration.json").write_text(
        json.dumps(config, indent=2)
    )

    manifest = []

    for path in [
        smiles_path, labels_path, matrix_path, comparison_path
    ]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest.append({
            "path": str(path),
            "sha256": digest
        })

    pd.DataFrame(manifest).to_csv(
        input_dir / "input_manifest.csv",
        index=False
    )

    print("\nAnalysis completed.")
    print("Output:", out)
    print("Environment records:", len(env_df))
    print("\nDistinct environments per bit:")
    print(collision.to_string(index=False))


if __name__ == "__main__":
    main()
