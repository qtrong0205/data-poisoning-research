#!/usr/bin/env python3
"""Recover poison_manifest.json from existing poisoned manifests and metadata."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if first_line.count("\t") > first_line.count(",") else ","


def read_manifest(path: Path) -> Tuple[List[Dict[str, str]], int]:
    delimiter = detect_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        required = {"sample_id", "is_poisoned"}
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                f"{path}: missing required columns: {', '.join(sorted(missing))}"
            )

        rows = list(reader)

    return rows, len(rows)


def parse_poison_flag(value: str, row_number: int, path: Path) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        f"{path}: invalid is_poisoned value at data row {row_number}: {value!r}"
    )


def build_poison_manifest(
    experiment_dir: Path,
    expected_attack: str,
) -> Tuple[dict, int, int]:
    metadata_path = experiment_dir / "metadata.json"
    csv_path = experiment_dir / "manifest_train.csv"

    if not metadata_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(
            f"{experiment_dir}: expected both manifest_train.csv and metadata.json"
        )

    with metadata_path.open("r", encoding="utf-8-sig") as f:
        metadata = json.load(f)

    rows, total_rows = read_manifest(csv_path)

    sample_ids = []
    seen = set()

    for row_number, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"{csv_path}: empty sample_id at CSV row {row_number}")
        if sample_id in seen:
            raise ValueError(f"{csv_path}: duplicate sample_id: {sample_id}")
        seen.add(sample_id)

        if parse_poison_flag(row.get("is_poisoned", ""), row_number, csv_path):
            sample_ids.append(sample_id)

    if "poison_count" not in metadata:
        raise ValueError(f"{metadata_path}: missing poison_count")

    expected_poison_count = int(metadata["poison_count"])
    if len(sample_ids) != expected_poison_count:
        raise ValueError(
            f"{experiment_dir}: metadata poison_count={expected_poison_count}, "
            f"but manifest contains {len(sample_ids)} rows marked TRUE"
        )

    if "total_samples" in metadata:
        expected_total = int(metadata["total_samples"])
        if total_rows != expected_total:
            raise ValueError(
                f"{experiment_dir}: metadata total_samples={expected_total}, "
                f"but manifest has {total_rows} rows"
            )

    result = dict(metadata)
    result["attack"] = expected_attack
    result["poisoned_samples"] = sample_ids

    return result, total_rows, len(sample_ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create poison_manifest.json for every experiment/rate folder "
            "that contains manifest_train.csv and metadata.json."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Input directory. Defaults to <project-root>/data/poisoned/ATTACK.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing poison_manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without writing JSON files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_root = args.input_root or (
        project_root / "data" / "poisoned" / "label_flipping"
    )
    input_root = input_root.resolve()

    if not input_root.is_dir():
        print(f"ERROR: input directory does not exist: {input_root}", file=sys.stderr)
        return 1

    experiment_dirs = sorted(
        p for p in input_root.glob("*/*")
        if p.is_dir()
        and (p / "manifest_train.csv").is_file()
        and (p / "metadata.json").is_file()
    )

    if not experiment_dirs:
        print(f"No experiment/rate folders found under: {input_root}")
        return 0

    failures = 0
    created = 0

    for experiment_dir in experiment_dirs:
        output_path = experiment_dir / "poison_manifest.json"
        try:
            result, total_count, poison_count = build_poison_manifest(
                experiment_dir, "label_flipping"
            )

            if output_path.exists() and not args.overwrite:
                print(f"SKIP (already exists): {output_path}")
                continue

            if args.dry_run:
                print(
                    f"VALID (dry-run): {experiment_dir} | "
                    f"total={total_count}, poisoned={poison_count}"
                )
                continue

            temp_path = output_path.with_suffix(".json.tmp")
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                f.write("\n")
            temp_path.replace(output_path)

            created += 1
            print(
                f"CREATED: {output_path} | "
                f"total={total_count}, poisoned={poison_count}"
            )

        except Exception as exc:
            failures += 1
            print(f"ERROR: {experiment_dir}: {exc}", file=sys.stderr)

    print(
        f"\nFinished. Created={created}, "
        f"failed={failures}, scanned={len(experiment_dirs)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
