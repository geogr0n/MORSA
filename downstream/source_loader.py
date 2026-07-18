from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SOURCE_BY_NAME, SOURCE_NAMES
from .errors import MissingInputError, MissingItem


@dataclass
class ExpressionFrame:
    cohort: str
    cancer: str
    source: str
    source_role: str
    fold: int
    expr: pd.DataFrame

    @property
    def n_patients(self) -> int:
        return int(self.expr.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expr.shape[1])


def cohort_paths(data_root: str | Path, cancer: str, cohort: str) -> tuple[Path, Path]:
    root = Path(data_root)
    cohort_upper = cohort.upper()
    if cohort_upper == "TCGA":
        base = root / cancer
        return base / "ref_file.csv", base / "output" / "TCGA"
    if cohort_upper == "CPTAC":
        base = root / "cross-cohort" / f"CPTAC-{cancer}"
        return base / "ref_file.csv", base / "output" / f"{cancer}_TCGA_models"
    raise ValueError(f"Unsupported cohort: {cohort}")


def load_ref_table(ref_file: Path) -> pd.DataFrame:
    if not ref_file.is_file():
        raise MissingInputError(f"Missing ref_file: {ref_file}", [MissingItem(str(ref_file), "missing file", "ref_file")])
    ref = pd.read_csv(ref_file)
    required = {"wsi_file_name", "patient_id"}
    missing = required - set(ref.columns)
    if missing:
        raise MissingInputError(
            f"ref_file is missing required columns: {sorted(missing)}",
            [MissingItem(str(ref_file), f"missing columns: {sorted(missing)}", "ref_file")],
        )
    ref["wsi_file_name"] = ref["wsi_file_name"].astype(str)
    ref["patient_id"] = ref["patient_id"].astype(str)
    return ref


def load_all_source_splits(
    data_root: str | Path,
    cancer: str,
    cohort: str,
    sources: tuple[str, ...] = SOURCE_NAMES,
) -> dict[int, dict[str, ExpressionFrame]]:
    ref_file, model_root = cohort_paths(data_root, cancer, cohort)
    ref = load_ref_table(ref_file)
    wsi_to_patient = dict(zip(ref["wsi_file_name"], ref["patient_id"]))
    missing = _missing_source_paths(model_root, sources)
    if missing:
        raise MissingInputError(f"Missing model prediction files under {model_root}", missing)

    anchor = _load_prediction_pickle(model_root / "mean" / "test_results.pkl")
    split_keys = sorted([key for key in anchor if key.startswith("split_")], key=lambda x: int(x.split("_")[1]))
    frames: dict[int, dict[str, ExpressionFrame]] = {}
    for split_key in split_keys:
        fold = int(split_key.split("_")[1])
        frames[fold] = {}
        for source_name in sources:
            spec = SOURCE_BY_NAME[source_name]
            if source_name == "TrueRNA":
                result = anchor[split_key]
                genes = list(anchor["genes"])
                matrix = result["real"]
            else:
                result_data = _load_prediction_pickle(model_root / str(spec.experiment) / "test_results.pkl")
                if split_key not in result_data:
                    raise MissingInputError(
                        f"Prediction file for {source_name} does not contain {split_key}",
                        [MissingItem(str(model_root / str(spec.experiment) / "test_results.pkl"), f"missing {split_key}", source_name)],
                    )
                result = result_data[split_key]
                genes = list(result_data["genes"])
                matrix = result["preds"]
            expr = _patient_expression(result["wsi_file_name"], matrix, genes, wsi_to_patient, source_name)
            frames[fold][source_name] = ExpressionFrame(
                cohort=cohort.upper(),
                cancer=cancer,
                source=source_name,
                source_role=spec.role,
                fold=fold,
                expr=expr,
            )
    return frames


def load_tcga_fold_patients(data_root: str | Path, cancer: str, fold: int, anchor_experiment: str = "mean") -> tuple[list[str], list[str]]:
    _, model_root = cohort_paths(data_root, cancer, "TCGA")
    train_path = model_root / anchor_experiment / f"train_{fold}.npy"
    test_path = model_root / anchor_experiment / f"test_{fold}.npy"
    missing = []
    if not train_path.is_file():
        missing.append(MissingItem(str(train_path), "missing fold split", "train patients"))
    if not test_path.is_file():
        missing.append(MissingItem(str(test_path), "missing fold split", "test patients"))
    if missing:
        raise MissingInputError(f"Missing train/test split files for {cancer} fold {fold}", missing)
    train = [str(x) for x in np.load(train_path, allow_pickle=True).tolist()]
    test = [str(x) for x in np.load(test_path, allow_pickle=True).tolist()]
    return train, test


def load_real_expression_from_ref(
    data_root: str | Path,
    cancer: str,
    patients: list[str] | None = None,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    ref_file, _ = cohort_paths(data_root, cancer, "TCGA")
    ref = load_ref_table(ref_file)
    rna_cols = [col for col in ref.columns if col.startswith("rna_")]
    if genes is not None:
        wanted = {f"rna_{gene}" for gene in genes}
        rna_cols = [col for col in rna_cols if col in wanted]
    if not rna_cols:
        raise MissingInputError(
            f"No RNA columns found in {ref_file}",
            [MissingItem(str(ref_file), "missing rna_* columns", "TrueRNA expression")],
        )
    if patients is not None:
        patient_set = set(map(str, patients))
        ref = ref[ref["patient_id"].isin(patient_set)]
    expr = ref[["patient_id", *rna_cols]].copy()
    expr = expr.rename(columns={col: col[4:] for col in rna_cols})
    gene_cols = [col[4:] for col in rna_cols]
    expr[gene_cols] = expr[gene_cols].apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby("patient_id", sort=True)[gene_cols].mean()
    expr.index = expr.index.astype(str)
    expr.columns = [str(col).upper() for col in expr.columns]
    return expr


def align_source_to_reference(reference: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = reference.index.intersection(source.index)
    genes = reference.columns.intersection(source.columns)
    return reference.loc[patients, genes].sort_index(), source.loc[patients, genes].sort_index()
def _patient_expression(
    wsi_names: np.ndarray | list[str],
    matrix: np.ndarray,
    genes: list[str],
    wsi_to_patient: dict[str, str],
    context: str,
) -> pd.DataFrame:
    missing_wsi = [str(wsi) for wsi in wsi_names if str(wsi) not in wsi_to_patient]
    if missing_wsi:
        preview = ", ".join(missing_wsi[:5])
        raise MissingInputError(
            f"Some WSI names in predictions cannot be mapped to patient_id: {preview}",
            [MissingItem(preview, "missing wsi_file_name in ref_file", context)],
        )
    patients = [wsi_to_patient[str(wsi)] for wsi in wsi_names]
    genes_upper = [str(gene).upper() for gene in genes]
    expr = pd.DataFrame(matrix, index=pd.Index(patients, name="patient_id"), columns=genes_upper)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr.groupby(level=0, sort=True).mean()


def _missing_source_paths(model_root: Path, sources: tuple[str, ...]) -> list[MissingItem]:
    missing = []
    for source_name in sources:
        if source_name == "TrueRNA":
            continue
        spec = SOURCE_BY_NAME[source_name]
        path = model_root / str(spec.experiment) / "test_results.pkl"
        if not path.is_file():
            missing.append(MissingItem(str(path), "missing prediction pkl", source_name))
    return missing


def _load_prediction_pickle(path: Path) -> dict:
    if not path.is_file():
        raise MissingInputError(f"Missing prediction file: {path}", [MissingItem(str(path), "missing prediction pkl", "prediction")])
    with path.open("rb") as handle:
        return pickle.load(handle)
