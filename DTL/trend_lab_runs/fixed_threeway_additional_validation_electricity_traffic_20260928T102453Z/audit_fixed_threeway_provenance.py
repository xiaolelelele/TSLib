#!/usr/bin/env python3
"""Independent source-data and split-provenance audit for fixed-threeway runs.

This companion checks configuration/data identity only. It does not load NPZ,
checkpoints, predictions, or any test metric. New Electricity/Traffic CSV values
are compared row-by-row only to verify the frozen first-eight-column derivation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_ROOT = WORKSPACE / "DTL/trend_lab_fusion_runs/triple_mean_revalidation_20260928T092723Z"
DEFAULT_OLD_ROOT = WORKSPACE / "DTL/trend_lab_runs/fixed_threeway_exploratory_old_8groups_20260928T102002Z"
DEFAULT_FRESH_ROOT = WORKSPACE / "DTL/trend_lab_runs/fixed_threeway_additional_validation_electricity_traffic_20260928T102453Z"
EXPECTED_NEW_BOUNDS = {
    "Electricity": [0, 15782, 19728, 22358, 26304],
    "Traffic": [0, 10526, 13158, 14912, 17544],
}
RATIOS = [0.6, 0.15, 0.1, 0.15]
NEW_COLUMNS = [str(i) for i in range(8)]
FROZEN_PROTOCOL_SHA256 = "d1b90f4005927c36558d7aa38cb71e50385dbdc08a0eed8ec6e7e10832581852"
CHECK_COUNT = 0


def require(condition: bool, message: str) -> None:
    global CHECK_COUNT
    CHECK_COUNT += 1
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def verify_file(path_value: str | Path, expected_hash: str, label: str) -> tuple[Path, str]:
    path = Path(path_value).expanduser().resolve()
    require(path.is_file(), f"{label}: missing file {path}")
    actual = sha256_file(path)
    require(actual == expected_hash, f"{label}: SHA256 mismatch for {path}")
    return path, actual


def manifest_hashes(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            digest, name = parts
            output[str(Path(name).expanduser().resolve())] = digest
    return output


def verify_line_hash_manifest(manifest_path: Path, base: Path, label: str) -> int:
    count = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split(maxsplit=1)
        require(len(parts) == 2, f"{label}:{line_number}: malformed hash row")
        digest, name = parts
        candidate = Path(name).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        verify_file(candidate, digest, f"{label}:{line_number}")
        count += 1
    require(count > 0, f"{label}: empty hash manifest")
    return count


def csv_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and len(header) >= 2, f"{path}: missing CSV header/data columns")
        count = 0
        for row in reader:
            require(len(row) == len(header), f"{path}: malformed row {count + 2}: {len(row)} fields vs {len(header)}")
            count += 1
    return [str(value) for value in header], count


def compare_first_eight(original: Path, derived: Path, dataset: str) -> dict[str, Any]:
    expected_header = ["date", *NEW_COLUMNS]
    with original.open("r", newline="", encoding="utf-8-sig") as source_stream, \
            derived.open("r", newline="", encoding="utf-8-sig") as derived_stream:
        source_reader = csv.reader(source_stream)
        derived_reader = csv.reader(derived_stream)
        source_header = next(source_reader, None)
        derived_header = next(derived_reader, None)
        require(source_header is not None and derived_header is not None,
                f"{dataset}: one CSV has no header")
        require(source_header[0] == "date", f"{dataset}: first source column is not date")
        require(source_header[1:9] == NEW_COLUMNS,
                f"{dataset}: source first eight value columns are {source_header[1:9]}, expected {NEW_COLUMNS}")
        require(source_header[-1] == "OT" and all(value.isdigit() for value in source_header[1:-1]),
                f"{dataset}: source numeric-column header layout is unexpected")
        require(derived_header == expected_header,
                f"{dataset}: derived header {derived_header} differs from {expected_header}")
        rows = 0
        for rows, (source_row, derived_row) in enumerate(zip(source_reader, derived_reader), start=1):
            require(len(source_row) == len(source_header), f"{dataset}: malformed source row {rows + 1}")
            require(len(derived_row) == len(expected_header), f"{dataset}: malformed derived row {rows + 1}")
            for column_index, (derived_value, source_value) in enumerate(zip(derived_row, source_row[:9])):
                require(derived_value == source_value,
                        f"{dataset}: derived row {rows + 1}, column {column_index} differs from source")
        # Detect a length mismatch without loading either file into memory.
        require(next(source_reader, None) is None and next(derived_reader, None) is None,
                f"{dataset}: original and derived row counts differ")
    return {"rows": rows, "source_columns": len(source_header),
            "derived_columns": len(derived_header), "columns": expected_header,
            "all_rows_and_values_exact": True}


def split_bounds(n: int, ratios: list[float]) -> list[int]:
    cumulative = 0.0
    result = [0]
    for ratio in ratios:
        cumulative += float(ratio)
        result.append(int(math.floor(cumulative * n)))
    result[-1] = n
    return result


def verify_cell_metadata(group_key: str, group: dict[str, Any], expected_seeds: list[int] | None = None
                         ) -> dict[str, Any]:
    source = group.get("source")
    require(isinstance(source, dict), f"{group_key}: missing source provenance")
    config_path, config_sha = verify_file(
        source.get("source_config_path", ""), source.get("config_sha256", ""), f"{group_key}/config")
    results_path, results_sha = verify_file(
        source.get("source_results_path", ""), source.get("results_sha256", ""), f"{group_key}/results")
    data_path, data_sha = verify_file(source.get("data_file", ""), source.get("data_sha256", ""),
                                      f"{group_key}/source data")
    config = read_json(config_path)
    source_results = read_json(results_path)
    data_config = config.get("data", {})
    data_info = source_results.get("data", {})
    split = group.get("split_metadata", {})
    columns = list(data_config.get("value_columns", []))
    require(columns == list(group.get("input_columns", [])),
            f"{group_key}: config value_columns differ from fusion group input_columns")
    require(int(data_config.get("channels", -1)) == len(columns),
            f"{group_key}: config channel count differs from value_columns")
    require(int(data_config.get("train_stride", -1)) == 4,
            f"{group_key}: config train_stride is not 4")
    require(int(data_config.get("eval_stride", -1)) == 24,
            f"{group_key}: config eval_stride is not 24")
    require([float(v) for v in data_config.get("split_ratios", [])] == RATIOS,
            f"{group_key}: config split ratios differ from protocol")
    require(int(data_config.get("lookback", -1)) == 96 and int(data_config.get("horizon", -1)) == 24,
            f"{group_key}: config lookback/horizon differ from protocol")
    require(data_info.get("train_stride") == 4 and data_info.get("eval_stride") == 24,
            f"{group_key}: source results train/eval stride differ from protocol")
    require([float(v) for v in data_info.get("split_ratios", [])] == RATIOS,
            f"{group_key}: source results split ratios differ from protocol")
    require(int(data_info.get("lookback", -1)) == 96 and int(data_info.get("horizon", -1)) == 24,
            f"{group_key}: source results lookback/horizon differ from protocol")
    require(data_info.get("smoke_truncated", False) is False,
            f"{group_key}: source result says data were smoke truncated")

    data_header, row_count = csv_header_and_count(data_path)
    time_column = data_config.get("time_column")
    if time_column:
        require(time_column in data_header, f"{group_key}: configured time column absent from source data")
    require(all(column in data_header for column in columns),
            f"{group_key}: one or more configured value columns are absent from source data")
    source_bounds = [int(v) for v in data_info.get("split_bounds", [])]
    group_bounds = [int(v) for v in split.get("split_bounds", [])]
    expected_bounds = split_bounds(row_count, RATIOS)
    require(source_bounds == expected_bounds,
            f"{group_key}: source-result bounds {source_bounds} differ from row-derived {expected_bounds}")
    require(group_bounds == source_bounds,
            f"{group_key}: fusion group bounds {group_bounds} differ from source results {source_bounds}")
    require(int(data_info.get("length", -1)) == row_count
            and int(data_info.get("source_rows", -1)) == row_count
            and int(data_info.get("rows_retained", -1)) == row_count,
            f"{group_key}: source result row counts differ from CSV row count {row_count}")
    require(int(split.get("train_stride", -1)) == 4 and int(split.get("eval_stride", -1)) == 24,
            f"{group_key}: fusion group train/eval stride differs from protocol")
    require([float(v) for v in split.get("split_ratios", [])] == RATIOS,
            f"{group_key}: fusion group split ratios differ from protocol")
    if expected_seeds is not None:
        require(list(map(int, config.get("seeds", []))) == expected_seeds,
                f"{group_key}: training config seed list differs from expected {expected_seeds}")

    source_checkpoint_sha = source.get("checkpoint_sha256")
    source_prediction_sha = source.get("source_prediction_npz_sha256")
    for path_key, hash_value, label in (
        ("source_checkpoint_path", source_checkpoint_sha, "checkpoint"),
        ("source_prediction_npz_path", source_prediction_sha, "source predictions"),
    ):
        require(isinstance(path_key, str) and isinstance(hash_value, str),
                f"{group_key}: missing {label} path/hash metadata")
        verify_file(source[path_key], hash_value, f"{group_key}/{label}")
    require(source.get("reload_verified") is True
            and source_results.get("checkpoint_reload_verified") is True,
            f"{group_key}: checkpoint reload verification is not true")

    return {
        "group_key": group_key,
        "dataset": group.get("dataset"),
        "backbone": group.get("backbone"),
        "source_config_sha256": config_sha,
        "source_results_sha256": results_sha,
        "source_data_sha256": data_sha,
        "source_data_path": str(data_path),
        "csv_rows": row_count,
        "csv_columns": len(data_header),
        "input_columns": columns,
        "split_bounds": source_bounds,
        "train_stride": 4,
        "eval_stride": 24,
        "lookback": 96,
        "horizon": 24,
        "checkpoint_sha256": source_checkpoint_sha,
        "source_prediction_npz_sha256": source_prediction_sha,
    }


def verify_training_matrix(training_root: Path, fresh_groups: dict[str, Any]) -> dict[str, Any]:
    matrix_path = training_root / "matrix_verification.json"
    require(matrix_path.is_file(), f"missing training matrix manifest {matrix_path}")
    matrix = read_json(matrix_path)
    matrix_sha = sha256_file(matrix_path)
    require(matrix.get("run_root") == str(training_root.resolve()), "training matrix run_root mismatch")
    require(matrix.get("phase") == "full_matrix" and matrix.get("wrapper_exit_code") == 0,
            "training matrix did not complete successfully")
    require(matrix.get("protocol_sha256") == FROZEN_PROTOCOL_SHA256
            and matrix.get("protocol_snapshot_sha256") == FROZEN_PROTOCOL_SHA256
            and matrix.get("protocol_snapshot_matches_frozen_hash") is True,
            "training matrix protocol hash/snapshot mismatch")
    require(matrix.get("test_metrics_requested_or_computed") is not True
            and matrix.get("preflight", {}).get("test_metrics_requested_or_computed") is False,
            "training preflight requested or computed test metrics")
    require(matrix.get("training_process_exit_codes") == {"Electricity": 0, "Traffic": 0},
            "training matrix process status incomplete")
    require(matrix.get("source_snapshot_matches_live_python_files") is True,
            "training matrix source snapshot check was not successful")
    source_hash_entries = verify_line_hash_manifest(training_root / "source_sha256.txt", training_root,
                                                    "training source hash manifest")
    config_hash_entries = verify_line_hash_manifest(training_root / "config_sha256.txt", training_root,
                                                    "training config hash manifest")
    reference_hash_entries = verify_line_hash_manifest(training_root / "reference_sha256.txt", training_root,
                                                       "training reference hash manifest")
    protocol_hash_entries = verify_line_hash_manifest(training_root / "protocol_sha256.txt", training_root,
                                                      "training protocol hash manifest")

    design = matrix.get("fixed_training_design", {})
    require(design.get("datasets") == ["Electricity", "Traffic"], "training design dataset list mismatch")
    require(design.get("backbones") == ["linear", "dlinear"], "training design backbone list mismatch")
    require(design.get("seeds") == [0, 1, 2, 3, 4], "training design seeds mismatch")
    require(design.get("train_stride") == 4 and design.get("eval_stride") == 24,
            "training design stride mismatch")
    require(design.get("all_variables") == "first 8 numeric columns after date, in source order",
            "training design column-selection rule mismatch")

    dataset_rows: dict[str, Any] = {}
    expected_cell_keys = {f"{dataset}_{backbone}" for dataset in ("Electricity", "Traffic")
                          for backbone in ("linear", "dlinear")}
    require(set(fresh_groups) == expected_cell_keys,
            f"fresh fusion group set differs: {sorted(fresh_groups)}")
    train_manifest = manifest_hashes(training_root / "data_sha256.txt")
    for dataset, expected_bounds in EXPECTED_NEW_BOUNDS.items():
        preflight = matrix["preflight"]["datasets"].get(dataset)
        require(isinstance(preflight, dict), f"matrix preflight missing {dataset}")
        require(preflight.get("selected_columns") == NEW_COLUMNS,
                f"{dataset}: matrix preflight selected columns differ from 0..7")
        require(preflight.get("prepared_raw_correspondence_verified") is True,
                f"{dataset}: matrix preflight raw correspondence flag is false")
        require(preflight.get("test_numeric_values_parsed_by_preflight") is False,
                f"{dataset}: preflight parsed test numeric values")
        require(preflight.get("source_rows") == preflight.get("prepared_rows"),
                f"{dataset}: preflight source/prepared row counts differ")
        require(preflight.get("adjacent_duplicates") == 0
                and preflight.get("backwards_intervals") == 0
                and preflight.get("off_nominal_intervals") == 0,
                f"{dataset}: source timestamp preflight failed")

        lower = dataset.lower()
        original = (WORKSPACE / f"TSLib/Time-Series-Library/dataset/{lower}/{lower}.csv").resolve()
        derived = (training_root / "prepared" / f"{lower}_first8.csv").resolve()
        original_sha = train_manifest.get(str(original))
        derived_sha = train_manifest.get(str(derived))
        require(isinstance(original_sha, str) and isinstance(derived_sha, str),
                f"{dataset}: data_sha256.txt misses original or prepared CSV")
        verify_file(original, original_sha, f"{dataset}/original CSV")
        verify_file(derived, derived_sha, f"{dataset}/prepared CSV")
        correspondence = compare_first_eight(original, derived, dataset)
        require(correspondence["rows"] == EXPECTED_NEW_BOUNDS[dataset][-1],
                f"{dataset}: raw CSV row count differs from fixed expected count")
        require(preflight.get("source_rows") == correspondence["rows"]
                and preflight.get("prepared_rows") == correspondence["rows"],
                f"{dataset}: matrix preflight rows disagree with actual CSVs")
        expected_groups = [f"{dataset}_{backbone}" for backbone in ("linear", "dlinear")]
        for group_key in expected_groups:
            group = fresh_groups[group_key]
            source = group["source"]
            cell_dir = training_root / "results" / dataset / f"data__{group['backbone']}"
            require(Path(source["source_cell_dir"]).resolve() == cell_dir.resolve(),
                    f"{group_key}: source cell dir differs from fixed training matrix")
            require(source.get("training_matrix_manifest_path") == str(matrix_path.resolve())
                    and source.get("training_matrix_manifest_sha256") == matrix_sha
                    and source.get("training_matrix_manifest_verified") is True
                    and source.get("source_matrix_verified") is True,
                    f"{group_key}: fusion provenance does not match verified training matrix manifest")
            require(source.get("original_data_path") == str(original)
                    and source.get("original_data_sha256") == original_sha,
                    f"{group_key}: original source CSV identity/hash mismatch")
            require(source.get("derived_data_path") == str(derived)
                    and source.get("derived_data_sha256") == derived_sha
                    and source.get("derived_data_columns") == NEW_COLUMNS,
                    f"{group_key}: prepared source CSV identity/columns mismatch")
            source_manifest_hash = source.get("source_data_hash_manifest_sha256")
            require(source.get("source_data_hash_manifest_path") == str((training_root / "data_sha256.txt").resolve())
                    and source_manifest_hash == sha256_file(training_root / "data_sha256.txt"),
                    f"{group_key}: source data hash manifest provenance mismatch")

            config_path = cell_dir / "config.json"
            results_path = cell_dir / "results.json"
            audit_path = cell_dir / "independent_audit.json"
            checkpoint_path = cell_dir / "checkpoint.pt"
            prediction_path = cell_dir / "predictions.npz"
            config = read_json(config_path)
            cell_results = read_json(results_path)
            train_audit = read_json(audit_path)
            for path, key in ((config_path, "config_sha256"), (results_path, "results_sha256"),
                              (checkpoint_path, "checkpoint_sha256"),
                              (prediction_path, "source_prediction_npz_sha256"),
                              (audit_path, "independent_training_audit_sha256")):
                expected_hash = source.get(key)
                require(isinstance(expected_hash, str), f"{group_key}: missing source hash {key}")
                verify_file(path, expected_hash, f"{group_key}/{key}")
            require(config.get("data", {}).get("value_columns") == NEW_COLUMNS
                    and config.get("data", {}).get("channels") == 8,
                    f"{group_key}: training config is not bound to columns 0..7")
            require(config.get("data", {}).get("csv_path") == str(derived),
                    f"{group_key}: training config CSV path differs from prepared CSV")
            require(config.get("data", {}).get("train_stride") == 4
                    and config.get("data", {}).get("eval_stride") == 24,
                    f"{group_key}: training config stride mismatch")
            require(config.get("seeds") == [0, 1, 2, 3, 4], f"{group_key}: training seeds mismatch")
            require(cell_results.get("checkpoint_reload_verified") is True
                    and cell_results.get("data", {}).get("smoke_truncated") is False,
                    f"{group_key}: source results reload/smoke state mismatch")
            require(train_audit.get("verified") is True
                    and train_audit.get("seeds") == [0, 1, 2, 3, 4]
                    and train_audit.get("source_sha256") == derived_sha,
                    f"{group_key}: independent training audit source/seeds mismatch")
            dataset_rows[group_key] = {
                "training_cell": str(cell_dir),
                "input_columns": NEW_COLUMNS,
                "source_rows": correspondence["rows"],
                "expected_split_bounds": expected_bounds,
                "checkpoint_reload_verified": True,
                "training_audit_verified": True,
            }
    return {"matrix_manifest_sha256": matrix_sha, "protocol_sha256": matrix["protocol_sha256"],
            "source_hash_manifest_entries_verified": source_hash_entries,
            "config_hash_manifest_entries_verified": config_hash_entries,
            "reference_hash_manifest_entries_verified": reference_hash_entries,
            "protocol_hash_manifest_entries_verified": protocol_hash_entries,
            "new_data": {dataset: {**correspondence_summary(dataset, training_root),
                                   "expected_split_bounds": EXPECTED_NEW_BOUNDS[dataset]}
                         for dataset in EXPECTED_NEW_BOUNDS},
            "cells": dataset_rows}


def correspondence_summary(dataset: str, training_root: Path) -> dict[str, Any]:
    lower = dataset.lower()
    original = (WORKSPACE / f"TSLib/Time-Series-Library/dataset/{lower}/{lower}.csv").resolve()
    derived = (training_root / "prepared" / f"{lower}_first8.csv").resolve()
    source_header, source_rows = csv_header_and_count(original)
    derived_header, derived_rows = csv_header_and_count(derived)
    return {"original_path": str(original), "original_sha256": sha256_file(original),
            "derived_path": str(derived), "derived_sha256": sha256_file(derived),
            "original_rows": source_rows, "derived_rows": derived_rows,
            "original_columns": len(source_header), "derived_columns": len(derived_header),
            "columns": ["date", *NEW_COLUMNS], "all_rows_and_values_exact": True}


def audit_roots(training_root: Path, old_root: Path, fresh_root: Path) -> dict[str, Any]:
    global CHECK_COUNT
    CHECK_COUNT = 0
    training_root, old_root, fresh_root = map(lambda p: p.expanduser().resolve(),
                                               (training_root, old_root, fresh_root))
    fresh_results = read_json(fresh_root / "results.json")
    old_results = read_json(old_root / "results.json")
    fresh_groups = fresh_results.get("groups", {})
    old_groups = old_results.get("groups", {})
    require(set(fresh_groups) == {"Electricity_linear", "Electricity_dlinear", "Traffic_linear", "Traffic_dlinear"},
            "fresh result groups are incomplete or unexpected")
    require(len(old_groups) == 8, f"expected 8 exploratory groups, got {len(old_groups)}")

    fresh_reports = []
    for key, group in sorted(fresh_groups.items()):
        info = verify_cell_metadata(key, group, [0, 1, 2, 3, 4])
        dataset = group.get("dataset")
        require(dataset in EXPECTED_NEW_BOUNDS, f"{key}: unexpected fresh dataset {dataset}")
        require(info["input_columns"] == NEW_COLUMNS,
                f"{key}: fresh fusion group input columns differ from 0..7")
        require(info["split_bounds"] == EXPECTED_NEW_BOUNDS[dataset],
                f"{key}: split bounds differ from fixed {dataset} boundaries")
        expected_data_path = (training_root / "prepared" / f"{dataset.lower()}_first8.csv").resolve()
        require(Path(info["source_data_path"]).resolve() == expected_data_path,
                f"{key}: fusion source data is not the expected prepared CSV")
        fresh_reports.append(info)

    old_reports = []
    for key, group in sorted(old_groups.items()):
        info = verify_cell_metadata(key, group, [0, 1, 2])
        old_reports.append(info)

    matrix_report = verify_training_matrix(training_root, fresh_groups)
    require(set(m["dataset"] + "_" + m["backbone"] for m in fresh_reports) == set(matrix_report["cells"]),
            "fresh result groups and verified training matrix cells differ")
    return {
        "passed": True,
        "audit": "fixed_threeway_source_data_and_split_provenance",
        "fresh_test_metrics_examined": False,
        "scope": ["training stride and split-boundary consistency",
                  "source config/results/data file hashes and row counts",
                  "training matrix manifest/preflight provenance",
                  "exact first-eight-column CSV derivation, all rows and values"],
        "training_root": str(training_root),
        "old_exploratory_root": str(old_root),
        "fresh_evaluation_root": str(fresh_root),
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "training_matrix": matrix_report,
        "old_group_count": len(old_reports),
        "old_groups": old_reports,
        "fresh_group_count": len(fresh_reports),
        "fresh_groups": fresh_reports,
        "train_stride": 4,
        "fresh_expected_split_bounds": EXPECTED_NEW_BOUNDS,
        "checks": CHECK_COUNT,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--fresh-root", type=Path, default=DEFAULT_FRESH_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    destination = args.report or args.fresh_root / "supplementary_provenance_audit.json"
    auditor = Path(__file__).resolve()
    snapshot = destination.parent / "audit_fixed_threeway_provenance.py"
    try:
        report = audit_roots(args.training_root, args.old_root, args.fresh_root)
        if snapshot.resolve() != auditor:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(auditor, snapshot)
        report["auditor"] = {"source": str(auditor), "snapshot": str(snapshot.resolve()),
                             "sha256": sha256_file(auditor)}
        report["report_path"] = str(destination.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(json.dumps({"passed": True, "checks": report["checks"], "report": str(destination.resolve())}))
        return 0
    except Exception as exc:
        failure = {"passed": False, "audit": "fixed_threeway_source_data_and_split_provenance",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "training_root": str(args.training_root.resolve()),
                   "old_exploratory_root": str(args.old_root.resolve()),
                   "fresh_evaluation_root": str(args.fresh_root.resolve()),
                   "fresh_test_metrics_examined": False}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
