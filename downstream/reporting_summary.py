from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACTIVE_TASK_DIRS = {"program_report", "msi_triage", "subtype_report"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect digital molecular reporting downstream CSV outputs.")
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    metric_files = sorted(path for path in root.glob("*/*metrics.csv") if path.parent.name in ACTIVE_TASK_DIRS)
    frames = []
    for path in metric_files:
        frames.append(pd.read_csv(path))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = root / "digital_molecular_reporting_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Wrote {len(summary)} rows to {out}")


if __name__ == "__main__":
    main()
