#!/usr/bin/env python3
"""Prepare exact first-eight-channel CSVs and report train-only quality."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/lgc/TSLib/TSLib/Time-Series-Library/dataset")
SPLIT_RATIOS = [0.60, 0.15, 0.10, 0.15]
VALUE_COUNT = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(raw: str) -> tuple[datetime, str]:
    try:
        return datetime.fromisoformat(raw), "isoformat"
    except ValueError:
        pass
    formats = (
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    )
    for date_format in formats:
        try:
            return datetime.strptime(raw, date_format), date_format
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp {raw!r}")


def inspect_time_and_schema(path: Path) -> dict:
    timestamps: list[datetime] = []
    row_count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or header[0] != "date":
            raise ValueError(f"Expected first CSV column to be exactly 'date': {path}")
        if len(header) < VALUE_COUNT + 1 or len(set(header)) != len(header):
            raise ValueError(f"CSV needs at least 8 uniquely named value columns: {path}")
        formats = Counter()
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Row width mismatch at CSV line {line}: {len(row)} != {len(header)}")
            timestamp, fmt = parse_time(row[0])
            timestamps.append(timestamp)
            formats[fmt] += 1
            row_count += 1
    if not row_count:
        raise ValueError(f"No data rows in {path}")
    deltas = [(current - previous).total_seconds()
              for previous, current in zip(timestamps, timestamps[1:])]
    positive = [delta for delta in deltas if delta > 0]
    nominal = Counter(positive).most_common(1)[0][0] if positive else None
    duplicate_examples = []
    reversal_examples = []
    interval_examples = []
    for index, delta in enumerate(deltas, start=1):
        if delta == 0 and len(duplicate_examples) < 20:
            duplicate_examples.append({
                "previous_data_row_index_zero_based": index - 1,
                "current_data_row_index_zero_based": index,
                "date": timestamps[index].isoformat(sep=" "),
            })
        elif delta < 0 and len(reversal_examples) < 20:
            reversal_examples.append({
                "previous_data_row_index_zero_based": index - 1,
                "current_data_row_index_zero_based": index,
                "delta_seconds": delta,
                "previous_date": timestamps[index - 1].isoformat(sep=" "),
                "current_date": timestamps[index].isoformat(sep=" "),
            })
        elif nominal is not None and delta != nominal and len(interval_examples) < 20:
            interval_examples.append({
                "previous_data_row_index_zero_based": index - 1,
                "current_data_row_index_zero_based": index,
                "delta_seconds": delta,
                "nominal_delta_seconds": nominal,
                "previous_date": timestamps[index - 1].isoformat(sep=" "),
                "current_date": timestamps[index].isoformat(sep=" "),
            })
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": row_count,
        "header": header,
        "all_value_column_count": len(header) - 1,
        "selected_value_columns_first_8_in_source_order": header[1:VALUE_COUNT + 1],
        "timestamp_parse_formats": dict(formats),
        "first_time": timestamps[0].isoformat(sep=" "),
        "last_time": timestamps[-1].isoformat(sep=" "),
        "adjacent_duplicate_count": sum(delta == 0 for delta in deltas),
        "backwards_interval_count": sum(delta < 0 for delta in deltas),
        "nominal_positive_interval_seconds": nominal,
        "positive_interval_min_seconds": min(positive) if positive else None,
        "positive_interval_median_seconds": median(positive) if positive else None,
        "positive_interval_max_seconds": max(positive) if positive else None,
        "interval_not_equal_to_nominal_count": sum(nominal is not None and delta != nominal for delta in deltas),
        "interval_examples_first_20": interval_examples,
        "duplicate_examples_first_20": duplicate_examples,
        "reversal_examples_first_20": reversal_examples,
        "split_ratios": SPLIT_RATIOS,
        "split_bounds_floor": [math.floor(ratio * row_count) for ratio in (0.0, 0.60, 0.75, 0.85, 1.0)],
    }


def prepare_dataset(name: str, relative: str) -> dict:
    source = DATA_ROOT / relative
    prepared = ROOT / "prepared" / f"{name.lower()}_first8.csv"
    source_meta = inspect_time_and_schema(source)
    train_end = source_meta["split_bounds_floor"][1]
    header = source_meta["header"]
    value_columns = source_meta["selected_value_columns_first_8_in_source_order"]
    train_values: list[list[float]] = [[] for _ in value_columns]
    nonfinite = []
    invalid = []
    with source.open("r", newline="", encoding="utf-8-sig") as in_stream, \
         prepared.open("w", newline="", encoding="utf-8") as out_stream:
        reader = csv.reader(in_stream)
        writer = csv.writer(out_stream, lineterminator="\n")
        source_header = next(reader)
        if source_header != header:
            raise AssertionError("CSV header changed between preflight passes")
        writer.writerow([header[0], *value_columns])
        for row_index, row in enumerate(reader):
            selected = row[1:VALUE_COUNT + 1]
            writer.writerow([row[0], *selected])
            if row_index >= train_end:
                continue
            for column_index, (column, raw) in enumerate(zip(value_columns, selected)):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    invalid.append({"data_row_index_zero_based": row_index, "column": column, "raw": raw})
                    continue
                if not math.isfinite(value):
                    nonfinite.append({"data_row_index_zero_based": row_index, "column": column, "raw": raw})
                    continue
                train_values[column_index].append(value)

    derived_meta = inspect_time_and_schema(prepared)
    if derived_meta["rows"] != source_meta["rows"]:
        raise AssertionError("Prepared CSV changed row count")
    if derived_meta["header"] != [header[0], *value_columns]:
        raise AssertionError("Prepared CSV changed selected header or column order")
    # Verify every date and selected raw numeric string across all rows without parsing test values.
    with source.open("r", newline="", encoding="utf-8-sig") as old_stream, \
         prepared.open("r", newline="", encoding="utf-8-sig") as new_stream:
        old_reader, new_reader = csv.reader(old_stream), csv.reader(new_stream)
        next(old_reader)
        next(new_reader)
        for row_index, (old_row, new_row) in enumerate(zip(old_reader, new_reader)):
            if [old_row[0], *old_row[1:VALUE_COUNT + 1]] != new_row:
                raise AssertionError(f"Selected raw strings/order differ at row {row_index}")

    train_columns = []
    constant_columns = []
    for column, values in zip(value_columns, train_values):
        finite_count = len(values)
        unique_count = len(set(values))
        is_constant = finite_count == train_end and unique_count <= 1
        if is_constant:
            constant_columns.append(column)
        train_columns.append({
            "column": column,
            "train_finite_value_count": finite_count,
            "train_unique_finite_value_count": unique_count,
            "exactly_constant_over_train": is_constant,
        })
    return {
        "source": source_meta,
        "prepared": derived_meta,
        "prepared_sha256": sha256(prepared),
        "prepared_row_and_selected_raw_string_correspondence_verified": True,
        "train_rows_for_quality_check": train_end,
        "train_invalid_numeric_count": len(invalid),
        "train_invalid_numeric_examples_first_20": invalid[:20],
        "train_nonfinite_count": len(nonfinite),
        "train_nonfinite_examples_first_20": nonfinite[:20],
        "selected_training_columns": train_columns,
        "exactly_constant_selected_train_columns": constant_columns,
        "train_quality_requires_review": bool(invalid or nonfinite),
        "test_numeric_values_parsed_by_preflight": False,
    }


def main() -> None:
    report = {
        "protocol_note": "8-channel pilot: first eight numeric columns after date in source order; full row sequence retained.",
        "test_metrics_requested_or_computed_by_preflight": False,
        "datasets": {
            "Electricity": prepare_dataset("Electricity", "electricity/electricity.csv"),
            "Traffic": prepare_dataset("Traffic", "traffic/traffic.csv"),
        },
    }
    destination = ROOT / "preflight.json"
    destination.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: {
            "rows": item["source"]["rows"],
            "all_column_count_including_date": len(item["source"]["header"]),
            "selected_first_8": item["source"]["selected_value_columns_first_8_in_source_order"],
            "timestamp_parse_formats": item["source"]["timestamp_parse_formats"],
            "duplicate_count": item["source"]["adjacent_duplicate_count"],
            "backwards_count": item["source"]["backwards_interval_count"],
            "nominal_interval_seconds": item["source"]["nominal_positive_interval_seconds"],
            "interval_not_nominal_count": item["source"]["interval_not_equal_to_nominal_count"],
            "train_rows": item["train_rows_for_quality_check"],
            "exactly_constant_train_columns": item["exactly_constant_selected_train_columns"],
            "train_invalid_numeric_count": item["train_invalid_numeric_count"],
            "train_nonfinite_count": item["train_nonfinite_count"],
            "derived_rows_and_selected_raw_strings_verified": item["prepared_row_and_selected_raw_string_correspondence_verified"],
        }
        for name, item in report["datasets"].items()
    }, ensure_ascii=True, indent=2))
    print(f"Full preflight report: {destination}")
    problems = {
        name: {
            "invalid_numeric_count": item["train_invalid_numeric_count"],
            "nonfinite_count": item["train_nonfinite_count"],
        }
        for name, item in report["datasets"].items()
        if item["train_quality_requires_review"]
    }
    if problems:
        raise ValueError(f"Training-segment values require review; no imputations were applied: {problems}")


if __name__ == "__main__":
    main()
