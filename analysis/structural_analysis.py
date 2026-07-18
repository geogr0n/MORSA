from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
EVAL_DIR = PROJECT_DIR / "evaluation"
for p in (SRC_DIR, EVAL_DIR, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_config import DEFAULT_RANK_K  # noqa: E402
from analysis.subspace_similarity import run_learned_vs_pca_subspace  # noqa: E402
from downstream.source_loader import (
    align_source_to_reference,
    load_all_source_splits,
    load_real_expression_from_ref,
    load_tcga_fold_patients,
)


SOURCE_TO_EXPERIMENT = {
    "TrueRNA": None,
    "Mean": "mean",
    "HE2RNA": "he2rna",
    "ViS": "vis",
    "MORSA-Enc": "morsa_enc",
    "MORSA-Mean": "morsa_mean",
    "MORSA-HE2RNA": "morsa_he2rna",
    "MORSA-ViS": "morsa_vis",
    "MORSA": "morsa",
}
SOURCE_ORDER = tuple(SOURCE_TO_EXPERIMENT.keys())
TASK_ORDER = (
    "rna_structure",
    "basis_stability",
    "coordinate_recovery",
    "morphology_structure",
    "component_attribution",
    "learnedrank_alignment",
    "efficiency",
)


def _parse_tasks(value: str) -> tuple[str, ...]:
    if not value.strip():
        return TASK_ORDER
    tasks = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(tasks) - set(TASK_ORDER))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return tasks


def _task_dir(results_root: str, task: str) -> Path:
    out = Path(results_root) / task
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tcga_ref_file(data_root: str, cancer: str) -> Path:
    return Path(data_root) / cancer / "ref_file.csv"


def _tcga_feature_path(data_root: str, cancer: str) -> Path:
    return Path(data_root) / cancer / "features"


def _pc90(matrix: np.ndarray) -> int:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return 0
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0)
    max_comp = int(min(x.shape[0] - 1, x.shape[1]))
    if max_comp <= 0:
        return 0
    pca = PCA(n_components=max_comp)
    pca.fit(x)
    cum = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_, nan=0.0))
    idx = np.searchsorted(cum, 0.90, side="left")
    return int(min(max_comp, idx + 1))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return math.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return math.nan
    return float(stats.pearsonr(x, y).statistic)


def _as_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (x - mean) / std
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _discover_experiments(model_root: Path) -> list[str]:
    experiments: list[str] = []
    if not model_root.is_dir():
        return experiments
    for item in sorted(model_root.iterdir()):
        if not item.is_dir():
            continue
        if item.name == "results":
            continue
        if (item / "model_best.pt").is_file():
            experiments.append(item.name)
    return experiments


def _load_test_results(model_root: Path, experiment: str) -> dict:
    path = model_root / experiment / "test_results.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test_results.pkl: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_metadata(model_root: Path, experiment: str) -> dict:
    path = model_root / experiment / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_summary_row(model_root: Path, experiment: str) -> dict[str, object]:
    row_file = model_root / "results" / experiment / "summary_row.csv"
    if not row_file.is_file():
        raise FileNotFoundError(f"Missing summary_row.csv: {row_file}")
    row = pd.read_csv(row_file).iloc[0].to_dict()
    return row


def run_rna_structure(data_root: str, cancer: str, results_root: str, seed: int) -> None:
    out = _task_dir(results_root, "rna_structure")
    expr = load_real_expression_from_ref(data_root, cancer)
    if expr.empty:
        raise ValueError("RNA structure analysis received an empty patient-level expression matrix.")
    x_real = expr.to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    x_perm = x_real.copy()
    for j in range(x_perm.shape[1]):
        x_perm[:, j] = rng.permutation(x_perm[:, j])
    pc90_real = _pc90(x_real)
    pc90_perm = _pc90(x_perm)
    ratio = float(pc90_real / pc90_perm) if pc90_perm > 0 else math.nan
    row = {
        "cancer": cancer,
        "n_patients": int(expr.shape[0]),
        "n_genes": int(expr.shape[1]),
        "PC90_real": pc90_real,
        "PC90_permuted": pc90_perm,
        "PC90_ratio_real_over_permuted": ratio,
    }
    pd.DataFrame([row]).to_csv(out / "rna_pca_structure.csv", index=False)


def run_basis_stability(cancer: str, model_root: Path, results_root: str) -> None:
    out = _task_dir(results_root, "basis_stability")
    experiment_dir = model_root / "morsa"
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Canonical MORSA directory not found: {experiment_dir}")
    fold = 0
    basis_list: list[np.ndarray] = []
    fold_ids: list[int] = []
    while (experiment_dir / f"train_{fold}.npy").is_file():
        basis_path = experiment_dir / f"basis_fold_{fold}.npz"
        if not basis_path.is_file():
            raise FileNotFoundError(
                f"Basis stability analysis requires the actual fold basis: {basis_path}. "
                "Run the canonical SPEX experiment before the structure analysis."
            )
        with np.load(basis_path, allow_pickle=False) as payload:
            if "U_k" not in payload:
                raise KeyError(f"{basis_path} does not contain U_k.")
            basis = np.asarray(payload["U_k"], dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != DEFAULT_RANK_K or not np.isfinite(basis).all():
            raise ValueError(
                f"Expected a finite G x {DEFAULT_RANK_K} basis in {basis_path}, got {basis.shape}."
            )
        basis_list.append(basis)
        fold_ids.append(fold)
        fold += 1

    if len(basis_list) < 2:
        raise ValueError("Basis stability analysis requires at least two folds with valid train samples.")

    rows = []
    for (i, ui), (j, uj) in itertools.combinations(list(zip(fold_ids, basis_list)), 2):
        if ui.shape != uj.shape:
            raise ValueError(f"Fold bases have inconsistent shapes: {ui.shape} and {uj.shape}.")
        s = np.linalg.svd(ui.T @ uj, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        proj_sim = float(np.mean(s**2))
        mean_cos = float(np.mean(s))
        max_angle_deg = float(np.degrees(np.max(np.arccos(np.clip(s, -1.0, 1.0)))))

        rows.append(
            {
                "cancer": cancer,
                "fold_i": i,
                "fold_j": j,
                "basis_source": "checkpoint",
                "projection_similarity": proj_sim,
                "mean_cosine_principal_vectors": mean_cos,
                "max_principal_angle_deg": max_angle_deg,
            }
        )

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(out / "fold_pair_subspace_similarity.csv", index=False)
    summary = {
        "cancer": cancer,
        "n_fold_pairs": int(len(pair_df)),
        "mean_projection_similarity": float(pair_df["projection_similarity"].mean()),
        "median_projection_similarity": float(pair_df["projection_similarity"].median()),
        "basis_source": "checkpoint",
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)


def run_coordinate_recovery(data_root: str, cancer: str, results_root: str) -> None:
    out = _task_dir(results_root, "coordinate_recovery")
    fold_frames = load_all_source_splits(data_root, cancer, "TCGA", sources=SOURCE_ORDER)

    pc_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for fold, source_frames in sorted(fold_frames.items()):
        train_patients, _ = load_tcga_fold_patients(data_root, cancer, fold)
        train_expr = load_real_expression_from_ref(data_root, cancer, patients=train_patients)
        train_genes = [g for g in train_expr.columns if pd.notna(g)]
        if len(train_genes) <= DEFAULT_RANK_K:
            continue
        z_train = _zscore_cols(train_expr[train_genes]).fillna(0.0)
        pca = PCA(n_components=DEFAULT_RANK_K)
        pca.fit(z_train.to_numpy(dtype=float))
        mu = pca.mean_.astype(np.float64)
        U = pca.components_.T.astype(np.float64)

        ref_expr = source_frames["TrueRNA"].expr
        for source in SOURCE_ORDER:
            src_expr = source_frames[source].expr
            ref_aligned, src_aligned = align_source_to_reference(ref_expr, src_expr)
            common_genes = [g for g in train_genes if g in ref_aligned.columns and g in src_aligned.columns]
            if len(common_genes) <= DEFAULT_RANK_K:
                continue
            col_idx = [train_genes.index(g) for g in common_genes]
            mu_sub = mu[col_idx]
            U_sub = U[col_idx, :]

            ref_x = ref_aligned[common_genes].to_numpy(dtype=float)
            src_x = src_aligned[common_genes].to_numpy(dtype=float)
            ref_pc = (ref_x - mu_sub) @ U_sub
            src_pc = (src_x - mu_sub) @ U_sub

            per_pc = []
            for pc in range(DEFAULT_RANK_K):
                corr = _safe_corr(ref_pc[:, pc], src_pc[:, pc])
                per_pc.append(corr)
                pc_rows.append(
                    {
                        "cancer": cancer,
                        "fold": fold,
                        "source": source,
                        "pc": pc + 1,
                        "pc_coord_pcc": corr,
                        "n_patients": int(ref_pc.shape[0]),
                        "n_genes": int(len(common_genes)),
                    }
                )
            summary_rows.append(
                {
                    "cancer": cancer,
                    "fold": fold,
                    "source": source,
                    "mean_pc1_16_pcc": float(np.nanmean(per_pc)),
                    "n_patients": int(ref_pc.shape[0]),
                    "n_genes": int(len(common_genes)),
                }
            )

    pc_df = pd.DataFrame(pc_rows)
    sum_df = pd.DataFrame(summary_rows)
    if pc_df.empty or sum_df.empty:
        raise ValueError("No valid fold/source coordinate-recovery rows were computed.")
    pc_df.to_csv(out / "pc_recovery_per_coordinate.csv", index=False)
    sum_df.to_csv(out / "pc_recovery_summary.csv", index=False)

    pair_defs = [
        ("Mean", "MORSA-Mean"),
        ("HE2RNA", "MORSA-HE2RNA"),
        ("ViS", "MORSA-ViS"),
        ("MORSA-Enc", "MORSA"),
    ]
    gain_rows = []
    for base, target in pair_defs:
        merged = (
            sum_df[sum_df["source"] == base][["fold", "mean_pc1_16_pcc"]]
            .merge(sum_df[sum_df["source"] == target][["fold", "mean_pc1_16_pcc"]], on="fold", suffixes=("_base", "_target"))
        )
        if merged.empty:
            continue
        merged["gain"] = merged["mean_pc1_16_pcc_target"] - merged["mean_pc1_16_pcc_base"]
        for _, row in merged.iterrows():
            gain_rows.append(
                {
                    "cancer": cancer,
                    "pair": f"{base}->{target}",
                    "fold": int(row["fold"]),
                    "base_mean_pc1_16_pcc": float(row["mean_pc1_16_pcc_base"]),
                    "target_mean_pc1_16_pcc": float(row["mean_pc1_16_pcc_target"]),
                    "gain": float(row["gain"]),
                }
            )
    pd.DataFrame(gain_rows).to_csv(out / "morsa_gain.csv", index=False)


def _feature_file(feature_root: Path, project: str, wsi_name: str) -> Path:
    wsi = str(wsi_name).replace(".svs", "")
    return feature_root / str(project) / wsi / f"{wsi}.h5"


def _covariance_metrics(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return {"effective_rank": math.nan, "pc90": math.nan, "offdiag_abs_mean": math.nan, "spectrum_top5_ratio": math.nan}
    xc = x - np.mean(x, axis=0, keepdims=True)
    cov = (xc.T @ xc) / float(max(x.shape[0] - 1, 1))
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(np.real(evals), a_min=0.0, a_max=None)
    total = float(np.sum(evals))
    if total <= 0:
        return {"effective_rank": math.nan, "pc90": math.nan, "offdiag_abs_mean": math.nan, "spectrum_top5_ratio": math.nan}
    p = evals / total
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    evals_desc = np.sort(evals)[::-1]
    cum = np.cumsum(evals_desc) / total
    pc90 = float(np.searchsorted(cum, 0.90, side="left") + 1)
    d = cov.shape[0]
    mask = ~np.eye(d, dtype=bool)
    offdiag = float(np.mean(np.abs(cov[mask]))) if np.any(mask) else 0.0
    topk = min(5, len(evals_desc))
    concentration = float(np.sum(evals_desc[:topk]) / total)
    return {
        "effective_rank": eff_rank,
        "pc90": pc90,
        "offdiag_abs_mean": offdiag,
        "spectrum_top5_ratio": concentration,
    }


def run_morphology_structure(data_root: str, cancer: str, results_root: str, seed: int) -> None:
    out = _task_dir(results_root, "morphology_structure")
    ref_file = _tcga_ref_file(data_root, cancer)
    feature_path = _tcga_feature_path(data_root, cancer)
    if not ref_file.is_file():
        raise FileNotFoundError(f"Missing ref_file: {ref_file}")
    if not feature_path.is_dir():
        raise FileNotFoundError(f"Missing feature_path: {feature_path}")
    ref = pd.read_csv(ref_file)
    needed = {"wsi_file_name", "tcga_project"}
    if not needed.issubset(ref.columns):
        raise ValueError(f"ref_file is missing required columns: {sorted(needed - set(ref.columns))}")

    rows = []
    rng = np.random.default_rng(int(seed) + 404)
    for _, row in ref.iterrows():
        f = _feature_file(feature_path, str(row["tcga_project"]), str(row["wsi_file_name"]))
        if not f.is_file():
            continue
        with h5py.File(f, "r") as handle:
            if "cluster_features" not in handle:
                continue
            feats = np.asarray(handle["cluster_features"][:], dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] < 2 or feats.shape[1] < 2:
            continue
        perm = feats.copy()
        for j in range(perm.shape[1]):
            perm[:, j] = rng.permutation(perm[:, j])

        real_m = _covariance_metrics(feats)
        perm_m = _covariance_metrics(perm)
        rows.append(
            {
                "cancer": cancer,
                "wsi_file_name": str(row["wsi_file_name"]),
                "tcga_project": str(row["tcga_project"]),
                "n_tokens": int(feats.shape[0]),
                "feat_dim": int(feats.shape[1]),
                "real_effective_rank": real_m["effective_rank"],
                "perm_effective_rank": perm_m["effective_rank"],
                "real_pc90": real_m["pc90"],
                "perm_pc90": perm_m["pc90"],
                "real_offdiag_abs_mean": real_m["offdiag_abs_mean"],
                "perm_offdiag_abs_mean": perm_m["offdiag_abs_mean"],
                "real_spectrum_top5_ratio": real_m["spectrum_top5_ratio"],
                "perm_spectrum_top5_ratio": perm_m["spectrum_top5_ratio"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid WSI features were found for covariance analysis.")
    df["delta_effective_rank"] = df["real_effective_rank"] - df["perm_effective_rank"]
    df["delta_pc90"] = df["real_pc90"] - df["perm_pc90"]
    df["delta_offdiag_abs_mean"] = df["real_offdiag_abs_mean"] - df["perm_offdiag_abs_mean"]
    df["delta_spectrum_top5_ratio"] = df["real_spectrum_top5_ratio"] - df["perm_spectrum_top5_ratio"]
    df.to_csv(out / "wsi_covariance_per_wsi.csv", index=False)
    summary = {
        "cancer": cancer,
        "n_wsi": int(len(df)),
        "real_effective_rank_mean": float(df["real_effective_rank"].mean()),
        "perm_effective_rank_mean": float(df["perm_effective_rank"].mean()),
        "real_pc90_mean": float(df["real_pc90"].mean()),
        "perm_pc90_mean": float(df["perm_pc90"].mean()),
        "real_offdiag_abs_mean": float(df["real_offdiag_abs_mean"].mean()),
        "perm_offdiag_abs_mean": float(df["perm_offdiag_abs_mean"].mean()),
        "real_spectrum_top5_ratio_mean": float(df["real_spectrum_top5_ratio"].mean()),
        "perm_spectrum_top5_ratio_mean": float(df["perm_spectrum_top5_ratio"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)


def run_component_attribution(model_root: Path, cancer: str, results_root: str) -> None:
    out = _task_dir(results_root, "component_attribution")
    required_experiments = (
        "mean",
        "morsa_mean",
        "he2rna",
        "morsa_he2rna",
        "vis",
        "morsa_vis",
        "morsa_enc",
        "morsa",
        "diag_morsa",
    )
    raw_rows = []
    for exp in required_experiments:
        row = _load_summary_row(model_root, exp)
        row["experiment"] = exp
        row["cancer"] = cancer
        raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out / "raw_metrics.csv", index=False)

    pair_defs = [
        ("Mean->MORSA-Mean", "mean", "morsa_mean"),
        ("HE2RNA->MORSA-HE2RNA", "he2rna", "morsa_he2rna"),
        ("ViS->MORSA-ViS", "vis", "morsa_vis"),
        ("MORSA-Enc->MORSA", "morsa_enc", "morsa"),
        ("Mean->MORSA-Enc", "mean", "morsa_enc"),
        ("MORSA-Mean->MORSA", "morsa_mean", "morsa"),
        ("DiagMORSA->MORSA", "diag_morsa", "morsa"),
    ]
    metric_cols = (
        "all_mean_pcc",
        "top1000_mean_pcc",
        "num_sig_genes",
        "avg_fold_train_seconds",
    )
    gains = []
    for pair_name, base_exp, target_exp in pair_defs:
        base = raw_df[raw_df["experiment"] == base_exp].iloc[0]
        target = raw_df[raw_df["experiment"] == target_exp].iloc[0]
        for metric in metric_cols:
            gains.append(
                {
                    "cancer": cancer,
                    "pair": pair_name,
                    "base_experiment": base_exp,
                    "target_experiment": target_exp,
                    "metric": metric,
                    "base_value": float(base[metric]),
                    "target_value": float(target[metric]),
                    "gain": float(target[metric] - base[metric]),
                }
            )
    pd.DataFrame(gains).to_csv(out / "structural_contributions.csv", index=False)


def _require_finite_metric(value: object, *, experiment: str, metric_name: str) -> float:
    number = _as_float_or_nan(value)
    if not np.isfinite(number):
        raise ValueError(f"Missing or invalid `{metric_name}` in metadata/summary for experiment `{experiment}`.")
    return float(number)


def run_efficiency_analysis(model_root: Path, results_root: str) -> None:
    out = _task_dir(results_root, "efficiency")
    experiments = _discover_experiments(model_root)
    rows = []
    for exp in experiments:
        summary = _load_summary_row(model_root, exp)
        meta = _load_metadata(model_root, exp)
        model_type = str(meta.get("model_type", "")).lower()
        head_type = str(meta.get("head_type", "")).lower()
        if not model_type or not head_type:
            raise ValueError(f"Metadata is missing `model_type` or `head_type` for `{exp}`.")
        rows.append(
            {
                "experiment": exp,
                "model_type": model_type,
                "head_type": head_type,
                "training_time_seconds_avg_fold": _require_finite_metric(
                    summary.get("avg_fold_train_seconds"),
                    experiment=exp,
                    metric_name="avg_fold_train_seconds",
                ),
                "parameter_count": _require_finite_metric(
                    meta.get("parameter_count_excluding_output_head"),
                    experiment=exp,
                    metric_name="parameter_count_excluding_output_head",
                ),
                "parameter_count_excluding_output_head": _require_finite_metric(
                    meta.get("parameter_count_excluding_output_head"),
                    experiment=exp,
                    metric_name="parameter_count_excluding_output_head",
                ),
                "parameter_count_including_output_head": _require_finite_metric(
                    meta.get("parameter_count_including_output_head"),
                    experiment=exp,
                    metric_name="parameter_count_including_output_head",
                ),
                "peak_gpu_memory_mb": _require_finite_metric(
                    meta.get("peak_gpu_memory_mb"),
                    experiment=exp,
                    metric_name="peak_gpu_memory_mb",
                ),
                "inference_time_per_wsi_seconds": _require_finite_metric(
                    meta.get("avg_inference_time_per_wsi_seconds"),
                    experiment=exp,
                    metric_name="avg_inference_time_per_wsi_seconds",
                ),
                "profiled_samples": 0,
                "metric_source": "training_metadata_only",
            }
        )
    pd.DataFrame(rows).to_csv(out / "model_efficiency.csv", index=False)


def _append_metric_rows(
    rows: list[dict[str, object]],
    *,
    cancer: str,
    task: str,
    source_file: Path,
    metrics: dict[str, object],
    comparison: str = "",
    experiment: str = "",
    extra: dict[str, object] | None = None,
) -> None:
    for metric, value in metrics.items():
        row = {
            "cancer": cancer,
            "task": task,
            "comparison": comparison,
            "experiment": experiment,
            "metric": metric,
            "value": value,
            "source_file": str(source_file),
        }
        if extra:
            row.update(extra)
        rows.append(row)


def write_analysis_summary(results_root: str | Path, cancer: str) -> Path:
    root = Path(results_root)
    rows: list[dict[str, object]] = []

    rna_structure = root / "rna_structure" / "rna_pca_structure.csv"
    if rna_structure.is_file():
        for record in pd.read_csv(rna_structure).to_dict("records"):
            metrics = {k: v for k, v in record.items() if k != "cancer"}
            _append_metric_rows(rows, cancer=cancer, task="rna_structure", source_file=rna_structure, metrics=metrics)

    basis_stability = root / "basis_stability" / "summary.csv"
    if basis_stability.is_file():
        for record in pd.read_csv(basis_stability).to_dict("records"):
            metrics = {k: v for k, v in record.items() if k != "cancer"}
            _append_metric_rows(
                rows,
                cancer=cancer,
                task="basis_stability",
                source_file=basis_stability,
                metrics=metrics,
            )

    coordinate_recovery = root / "coordinate_recovery" / "morsa_gain.csv"
    if coordinate_recovery.is_file():
        df = pd.read_csv(coordinate_recovery)
        if not df.empty:
            grouped = df.groupby("pair", dropna=False).agg(
                base_mean_pc1_16_pcc=("base_mean_pc1_16_pcc", "mean"),
                target_mean_pc1_16_pcc=("target_mean_pc1_16_pcc", "mean"),
                mean_pc1_16_pcc_gain=("gain", "mean"),
                min_pc1_16_pcc_gain=("gain", "min"),
                improvement_consistency=("gain", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0))),
            )
            for pair, record in grouped.reset_index().set_index("pair").to_dict("index").items():
                _append_metric_rows(
                    rows,
                    cancer=cancer,
                    task="coordinate_recovery",
                    comparison=str(pair),
                    source_file=coordinate_recovery,
                    metrics=record,
                )

    morphology_structure = root / "morphology_structure" / "summary.csv"
    if morphology_structure.is_file():
        for record in pd.read_csv(morphology_structure).to_dict("records"):
            metrics = {k: v for k, v in record.items() if k != "cancer"}
            _append_metric_rows(rows, cancer=cancer, task="morphology_structure", source_file=morphology_structure, metrics=metrics)

    component_attribution = root / "component_attribution" / "structural_contributions.csv"
    if component_attribution.is_file():
        for record in pd.read_csv(component_attribution).to_dict("records"):
            _append_metric_rows(
                rows,
                cancer=cancer,
                task="component_attribution",
                comparison=str(record.get("pair", "")),
                source_file=component_attribution,
                metrics={
                    f"{record.get('metric', 'metric')}_base": record.get("base_value"),
                    f"{record.get('metric', 'metric')}_target": record.get("target_value"),
                    f"{record.get('metric', 'metric')}_gain": record.get("gain"),
                },
                extra={
                    "base_experiment": record.get("base_experiment", ""),
                    "target_experiment": record.get("target_experiment", ""),
                },
            )

    learnedrank_alignment = root / "learnedrank_alignment" / "summary.csv"
    if learnedrank_alignment.is_file():
        for record in pd.read_csv(learnedrank_alignment).to_dict("records"):
            metrics = {k: v for k, v in record.items() if k != "cancer"}
            _append_metric_rows(
                rows,
                cancer=cancer,
                task="learnedrank_alignment",
                source_file=learnedrank_alignment,
                metrics=metrics,
            )

    efficiency = root / "efficiency" / "model_efficiency.csv"
    if efficiency.is_file():
        df = pd.read_csv(efficiency)
        for record in df.to_dict("records"):
            experiment = str(record.get("experiment", ""))
            metrics = {
                k: v
                for k, v in record.items()
                if k not in {"experiment", "model_type", "head_type"}
            }
            _append_metric_rows(
                rows,
                cancer=cancer,
                task="efficiency",
                experiment=experiment,
                source_file=efficiency,
                metrics=metrics,
                extra={"model_type": record.get("model_type", ""), "head_type": record.get("head_type", "")},
            )

    out = root / "structural_analysis_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MORSA analysis tasks.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--seed", type=int, default=29)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_root = Path(args.model_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    tasks = _parse_tasks(args.tasks)
    print(f"[analysis] cancer={args.cancer} tasks={','.join(tasks)}", flush=True)

    if "rna_structure" in tasks:
        run_rna_structure(args.data_root, args.cancer, args.results_root, seed=args.seed)
    if "basis_stability" in tasks:
        run_basis_stability(args.cancer, model_root, args.results_root)
    if "coordinate_recovery" in tasks:
        run_coordinate_recovery(args.data_root, args.cancer, args.results_root)
    if "morphology_structure" in tasks:
        run_morphology_structure(args.data_root, args.cancer, args.results_root, seed=args.seed)
    if "component_attribution" in tasks:
        run_component_attribution(model_root, args.cancer, args.results_root)
    if "learnedrank_alignment" in tasks:
        run_learned_vs_pca_subspace(
            cancer=args.cancer,
            model_root=model_root,
            output_dir=_task_dir(args.results_root, "learnedrank_alignment"),
        )
    if "efficiency" in tasks:
        run_efficiency_analysis(model_root=model_root, results_root=args.results_root)

    summary_path = write_analysis_summary(args.results_root, args.cancer)
    print(f"[analysis] summary -> {summary_path}", flush=True)
    print(f"[analysis] done -> {args.results_root}", flush=True)


if __name__ == "__main__":
    main()
