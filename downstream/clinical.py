from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import MissingInputError, MissingItem


def load_msi_table(resource_root: str | Path, cancer: str) -> pd.DataFrame:
    path = Path(resource_root) / "clinical" / "cbioportal" / cancer / "clinical_sample_long.tsv"
    if not path.is_file():
        raise MissingInputError(f"Missing cBioPortal sample clinical table: {path}", [MissingItem(str(path), "missing file", "MSI labels")])
    long = pd.read_csv(path, sep="\t")
    wanted = long[long["clinicalAttributeId"].isin(["MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE"])].copy()
    if wanted.empty:
        raise MissingInputError(f"No MSI labels found in {path}", [MissingItem(str(path), "missing MSI_SCORE_MANTIS/MSI_SENSOR_SCORE", "MSI labels")])
    wanted["patient_id"] = wanted["patientId"].astype(str)
    wanted["value"] = pd.to_numeric(wanted["value"], errors="coerce")
    wide = wanted.pivot_table(index="patient_id", columns="clinicalAttributeId", values="value", aggfunc="mean")
    wide["MSI_H_MANTIS"] = (wide["MSI_SCORE_MANTIS"] > 0.4).astype(float)
    if "MSI_SENSOR_SCORE" in wide.columns:
        wide["MSI_H_MSIsensor"] = (wide["MSI_SENSOR_SCORE"] >= 10.0).astype(float)
    return wide
