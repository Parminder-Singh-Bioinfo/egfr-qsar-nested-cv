#!/usr/bin/env python3
"""Reconstruct EGFR QSAR modeling CSVs from the four-matrix frozen ZIP.

This script reads the archive produced by the project's frozen-data packaging
step and uses representation_columns.json to restore original column order.
It never changes the archive or the original modeling matrices.

Requires Python 3.8+; standard library only.

Example:
    python reconstruct_modeling_matrices.py \
      --archive ~/projects/EGFR_QSAR_FROZEN_DATA.zip \
      --output ~/projects/EGFR_QSAR_RECONSTRUCTED \
      --project cell_based --split external \
      --representations 2d
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack
from zipfile import ZipFile

REPRESENTATIONS = ("2d", "3d", "morgan", "2d_3d", "2d_3d_morgan")
PROJECTS = ("single_protein", "cell_based")
SPLITS = ("development", "external")


def sha256_zip_member(archive, member):
    digest = hashlib.sha256()
    with archive.open(member, "r") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_csv_values(created, original):
    """Compare every CSV value and column in order; ignore CSV quoting differences."""
    with created.open("r", encoding="utf-8-sig", newline="") as a, original.open(
        "r", encoding="utf-8-sig", newline=""
    ) as b:
        a_reader, b_reader = csv.reader(a), csv.reader(b)
        for row_number, (row_a, row_b) in enumerate(
            __import__("itertools").zip_longest(a_reader, b_reader), start=1
        ):
            if row_a != row_b:
                return False, row_number
    return True, None


def verify_column_manifest(master_columns, descriptions):
    if len(set(master_columns)) != len(master_columns):
        raise ValueError("Master CSV has duplicate column names")
    positions = {column: index for index, column in enumerate(master_columns)}
    plans = {}
    for representation, metadata in descriptions.items():
        if representation not in REPRESENTATIONS:
            raise ValueError("Unrecognized representation: " + representation)
        columns = metadata["columns"]
        filename = metadata["filename"]
        if not isinstance(columns, list) or not columns:
            raise ValueError("Empty or invalid columns for " + representation)
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate feature names for " + representation)
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("Unsafe CSV filename: " + filename)
        missing = [column for column in columns if column not in positions]
        if missing:
            raise ValueError("Columns absent from master for {}: {}".format(representation, missing[:10]))
        plans[representation] = ([positions[column] for column in columns], columns, filename)
    return plans


def reconstruct(archive_path, output, project, split, representations, compare_to):
    with ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("representation_columns.json").decode("utf-8"))
        entry = manifest[project][split]
        member = entry["master_file"]
        expected_member = "{}/{}/5_all_features_2d_3d_morgan_with_metadata.csv".format(project, split)
        if member != expected_member:
            raise ValueError("Unexpected archive master-file path: " + member)
        expected_hash = entry["sha256"].lower()
        if sha256_zip_member(archive, member).lower() != expected_hash:
            raise ValueError("Master file SHA-256 mismatch: " + member)
        print("Verified master SHA-256:", member)

        with archive.open(member, "r") as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.reader(text_stream)
            master_columns = next(reader)
            plans = verify_column_manifest(master_columns, entry["representations"])
            missing_reps = [rep for rep in representations if rep not in plans]
            if missing_reps:
                raise ValueError("Representations missing from manifest: " + str(missing_reps))
            if compare_to and len(representations) != 1:
                raise ValueError("--compare-to needs exactly one representation")

            out_dir = output / project / split
            targets = {rep: out_dir / plans[rep][2] for rep in representations}
            existing = [str(target) for target in targets.values() if target.exists()]
            if existing:
                raise FileExistsError("Refusing to overwrite existing files: " + ", ".join(existing))

            # Write to temporary files first; only publish complete outputs.
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_paths = {}
            row_count = 0
            try:
                with ExitStack() as stack:
                    writers = {}
                    for rep in representations:
                        descriptor, path = tempfile.mkstemp(
                            prefix="." + plans[rep][2] + ".", suffix=".tmp", dir=str(out_dir)
                        )
                        temp_paths[rep] = Path(path)
                        target_file = stack.enter_context(
                            os.fdopen(descriptor, "w", encoding="utf-8", newline="")
                        )
                        writers[rep] = csv.writer(target_file, lineterminator="\n")
                        writers[rep].writerow(plans[rep][1])
                    for row in reader:
                        if len(row) != len(master_columns):
                            raise ValueError(
                                "Row {} has {} columns; master header has {}".format(
                                    row_count + 2, len(row), len(master_columns)
                                )
                            )
                        for rep in representations:
                            writers[rep].writerow([row[idx] for idx in plans[rep][0]])
                        row_count += 1
                if compare_to:
                    same, first_differing_row = compare_csv_values(temp_paths[representations[0]], compare_to)
                    if not same:
                        raise ValueError(
                            "Reconstructed values differ from reference at CSV row "
                            + str(first_differing_row)
                        )
                    print("Reference comparison: PASS (all columns and values in row order)")
                # No existing file is ever overwritten; avoid rename-replace semantics.
                for rep, temp_path in temp_paths.items():
                    target = targets[rep]
                    if target.exists():
                        raise FileExistsError(str(target))
                    os.link(str(temp_path), str(target))
                    temp_path.unlink()
                temp_paths.clear()
            finally:
                for temp_path in temp_paths.values():
                    try:
                        temp_path.unlink()
                    except FileNotFoundError:
                        pass

    print("Reconstructed rows:", row_count)
    for rep, path in targets.items():
        print("{}: {} ({:.2f} MB)".format(rep, path, path.stat().st_size / 1024**2))
    print("Original files and ZIP archive unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="Path to EGFR_QSAR_FROZEN_DATA.zip")
    parser.add_argument("--output", type=Path, required=True, help="New output directory (never use original project directory)")
    parser.add_argument("--project", required=True, choices=PROJECTS)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=["2d"])
    parser.add_argument("--compare-to", type=Path, help="Original CSV for value-by-value check (single representation)")
    args = parser.parse_args()
    if args.compare_to and not args.compare_to.is_file():
        parser.error("--compare-to file does not exist: " + str(args.compare_to))
    if len(args.representations) != len(set(args.representations)):
        parser.error("Duplicate --representations")
    if not args.archive.is_file():
        parser.error("Archive not found: " + str(args.archive))
    reconstruct(
        args.archive.expanduser(), args.output.expanduser(),
        args.project, args.split, args.representations, args.compare_to.expanduser() if args.compare_to else None
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)

