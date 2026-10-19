"""Reproduce stored audits/tables in a fresh directory; never modify source runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source" / "research"))
from audit_fixed_threeway import audit_root
from summarize_fixed_threeway import load_cells, build_tables, csv_write

BASE = Path("/home/lgc/TSLib/DTL")
ROOTS = [
    BASE / "trend_lab_runs/fixed_threeway_exploratory_old_8groups_20260928T102002Z",
    BASE / "trend_lab_runs/fixed_threeway_additional_validation_electricity_traffic_20260928T102453Z",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"audits": {}, "tables": {}}
    for label, root in zip(("old", "new"), ROOTS):
        audit = audit_root(root)
        (args.output / (label + "_audit.json")).write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n")
        summary["audits"][label] = {"passed": audit["passed"], "checks": audit["checks"]}
    cells, _ = load_cells(ROOTS)
    tables = build_tables(cells)
    names = ("method_summary", "per_seed", "comparisons", "selections", "cohort_counts")
    for name, rows in zip(names, tables):
        target = args.output / (name + ".csv")
        csv_write(target, rows)
        original = BASE / "research/analysis/fixed_threeway_20260928" / target.name
        equal = target.read_bytes() == original.read_bytes()
        summary["tables"][name] = {
            "rows": len(rows), "byte_identical": equal,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        if not equal:
            raise AssertionError("CSV differs from saved original: " + name)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
